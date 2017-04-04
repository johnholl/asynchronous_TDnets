from __future__ import print_function
import distutils.version
import threading
import numpy as np
import tensorflow as tf
from models.QA_model import BasicQTDNet
import random
import sys

use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')



class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, policy):
        threading.Thread.__init__(self)
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        env_runner(self.env, self.policy, self.summary_writer)





def env_runner(env, policy, summary_writer):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0
    prev_reward = 0
    prev_action = np.zeros(shape=[env.num_actions])

    while True:

        fetched = policy.act(last_state, prev_action, prev_reward, last_features[0], last_features[1], last_features[2])
        action, value_, features = fetched[0], fetched[1], fetched[2]
        action = np.array([float(int(i == action)) for i in range(env.num_actions)])
        state, reward, terminal = env.step(action.argmax())

        # collect the experience
        policy.update_replay_memory((last_state, action, reward, terminal,
                                     last_features, prev_action, [prev_reward],
                                     state, features
                                     ))
        length += 1
        rewards += reward

        prev_action = action
        prev_reward = reward
        last_state = state
        last_features = features

        if terminal:
            summary = tf.Summary()
            summary.value.add(tag='episode_reward', simple_value=float(rewards))
            summary.value.add(tag='reward_per_timestep', simple_value=float(rewards)/float(length))
            summary_writer.add_summary(summary, policy.global_step.eval())
            summary_writer.flush()
            # if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
            last_state = env.reset()
            last_features = policy.get_initial_features()
            print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
            length = 0
            rewards = 0


class ATDNet(object):
    def __init__(self, env, task, grid_size=20):

        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.TDnetwork = BasicQTDNet(env.observation_space_shape, env.num_actions)
                self.network = self.TDnetwork.anet
                self.qnetwork = self.TDnetwork.qnet
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_TDnetwork = BasicQTDNet(env.observation_space_shape, env.num_actions)
                self.local_network = pi = self.local_TDnetwork.anet
                self.local_qnetwork = self.local_TDnetwork.qnet
                pi.global_step = self.global_step

                self.Q_target = tf.placeholder(tf.float32, [None])
                self.action = tf.placeholder(tf.float32, [None, env.num_actions])



            # Define Answer network's Q-value loss

            if use_tf12_api:
                Q_readout = tf.reduce_sum(tf.multiply(pi.Q, self.action), reduction_indices=[1])
            else:
                Q_readout = tf.reduce_sum(tf.mul(pi.Q, self.action), reduction_indices=[1])

            q_delta = Q_readout - self.Q_target

            self.agent_loss = tf.reduce_mean(tf.where(
                                    tf.abs(q_delta) < 1,
                                    0.5 * tf.square(q_delta),
                                    tf.abs(q_delta) - 0.5),
                                    axis=0)

            self.avg_Q = tf.reduce_mean(Q_readout)
            self.max_Q = tf.reduce_max(Q_readout)


            # Define Answer network's prediction loss
            pred_loss_weight = 0.05

            self.prediction_target = tf.placeholder(tf.float32, [None, self.TDnetwork.num_predictions])

            pix_delta =  pi.predictions - self.prediction_target

            self.prediction_loss = pred_loss_weight*tf.reduce_sum(tf.reduce_mean(tf.where(
                                    tf.abs(pix_delta) < 1,
                                    0.5 * tf.square(pix_delta),
                                    tf.abs(pix_delta) - 0.5),
                                    axis=0))

            self.avg_prediction = tf.reduce_mean(pi.predictions)
            self.max_prediction = tf.reduce_max(pi.predictions)


            # Define Question network's loss

            self.question_target = tf.placeholder(tf.float32, [None])

            if use_tf12_api:
                questionq_readout = tf.reduce_sum(tf.multiply(self.local_TDnetwork.Q, self.action), reduction_indices=[1])
            else:
                questionq_readout = tf.reduce_sum(tf.mul(self.local_TDnetwork.Q, self.action), reduction_indices=[1])

            questionq_delta = questionq_readout - self.question_target

            self.question_loss = tf.reduce_mean(tf.where(
                                    tf.abs(questionq_delta) < 1,
                                    0.5 * tf.square(questionq_delta),
                                    tf.abs(questionq_delta) - 0.5),
                                    axis=0)

            self.avg_questionQ = tf.reduce_mean(questionq_readout)
            self.max_questionQ = tf.reduce_max(questionq_readout)


            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi)


            agent_grads = tf.gradients(self.agent_loss, pi.var_list)
            prediction_grads = tf.gradients(self.prediction_loss, pi.var_list)
            question_grads = tf.gradients(self.question_loss, self.local_TDnetwork.qnet.var_list)

            # main task summaries

            if use_tf12_api:
                tf.summary.scalar("model/Q_loss", self.agent_loss)
                tf.summary.scalar("model/Q_val", self.avg_Q)
                tf.summary.scalar("model/Q_max", self.max_Q)
                tf.summary.image("model/state", pi.x*255, max_outputs=12)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(agent_grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                for var in pi.var_list:
                    tf.summary.scalar("model/" + var.name + "_norm", tf.global_norm([var]))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/Q_loss", self.agent_loss)
                tf.scalar_summary("model/Q_val", self.avg_Q)
                tf.scalar_summary("model/Q_max", self.max_Q)
                tf.image_summary("model/state", pi.x*255, max_images=12)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(agent_grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                for var in pi.var_list:
                    tf.scalar_summary("model/" + var.name + "_norm", tf.global_norm([var]))
                self.summary_op = tf.merge_all_summaries()

            # auxiliary answer summaries

            if use_tf12_api:
                self.predloss_sum = tf.summary.scalar("model/pixel_loss", self.prediction_loss)
                self.predval = tf.summary.scalar("model/pixel_value", self.avg_prediction)
                self.predmax = tf.summary.scalar("model/pixel_max", self.max_prediction)
            else:
                self.predloss_sum = tf.scalar_summary("model/pixel_loss", self.prediction_loss)
                self.predval = tf.scalar_summary("model/pixel_value", self.avg_prediction)
                self.predmax = tf.scalar_summary("model/pixel_max", self.max_prediction)

            # auxiliary question summaries
            # ......

            agent_grads, _ = tf.clip_by_global_norm(agent_grads, 40.0)
            prediction_grads, _ = tf.clip_by_global_norm(prediction_grads, 40.0)
            question_grads, _ = tf.clip_by_global_norm(question_grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.local_TDnetwork.var_list, self.TDnetwork.var_list)])

            agent_grads_and_vars = list(zip(agent_grads, self.network.var_list))
            prediction_grads_and_vars = list(zip(prediction_grads, self.network.var_list))
            question_grads_and_vars = list(zip(question_grads, self.qnetwork.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.RMSPropOptimizer(1e-4)
            self.agent_train_op = tf.group(opt.apply_gradients(agent_grads_and_vars), inc_step)
            self.prediction_train_op = opt.apply_gradients(prediction_grads_and_vars)
            self.question_train_op = opt.apply_gradients(question_grads_and_vars)
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.sync)  # copy weights from shared to local

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        """
        Grab a minibatch from the replay memory. Set targets and update parameters
        from the Q-loss and Prediction-loss functions"""

        if len(self.runner.policy.replay_memory) > 100:
            Qbatch = self.get_minibatch()

            pentultimate_prediction, predictions, fetched = self.train_main(
                Qbatch, sess, should_compute_summary=should_compute_summary)

            final_pred = self.train_answer(Qbatch, sess, should_compute_summary, fetched, predictions)

            self.train_question(Qbatch, sess, should_compute_summary, final_pred, pentultimate_prediction)

        self.local_steps += 1



    def get_minibatch(self, length=20):
        '''Grab a minibatch of length = length'''

        Qbatch = []
        starting_pos = random.choice(range(len(self.runner.policy.replay_memory)-length))
        terminal = False
        batchsize = 0
        while not terminal and batchsize < length:
            Qbatch.append(self.runner.policy.replay_memory[starting_pos + batchsize])
            batchsize += 1
            terminal = Qbatch[-1][3]
        return Qbatch

    def train_main(self, Qbatch, sess, should_compute_summary):
        '''Train the forward network on the main task and also output predictions that will be used
        to update the question network'''

        last_states = [m[0] for m in Qbatch]
        actions = [m[1] for m in Qbatch]
        rewards = [m[2] for m in Qbatch]
        start_features = Qbatch[0][4]
        last_actions = [m[5] for m in Qbatch]
        last_rewards = [m[6] for m in Qbatch]
        states = [m[7] for m in Qbatch]
        features = [m[8] for m in Qbatch]
        next_features = features[0]

        feed_dict={self.local_network.x: states,
                   self.local_network.action: actions,
                   self.local_network.reward: np.expand_dims(rewards, axis=1),
                   self.local_network.state_in[0]: next_features[0],
                   self.local_network.state_in[1]: next_features[1],
                   self.local_network.state_in[2]: next_features[2],
                   self.local_network.bs: len(Qbatch)
                   }

        feed_dict.update(zip(self.local_network.var_list, self.local_network.target_weights))

        Q_values, values, predictions = sess.run((self.local_network.Q, self.local_network.vf, self.local_network.predictions),
                                                 feed_dict=feed_dict)

        Q_targets = []
        for i in range(len(Qbatch)):
            new_Q_target = 0.
            for j in range(len(Qbatch)-i):
                new_Q_target += (.99**j)*rewards[i+j]
            Q_targets.append(new_Q_target)
            if not Qbatch[-1][3]:
                Q_targets[-1] += (.99**(len(Qbatch)-i))*values[-1]

        # put fetches logic here
        if should_compute_summary:
            fetches = [self.summary_op, self.agent_train_op, self.local_network.predictions, self.global_step]
        else:
            fetches = [self.agent_train_op, self.local_network.predictions, self.global_step]

        fetched = sess.run(fetches, feed_dict={self.local_network.x: last_states,
                                                      self.local_network.action: last_actions,
                                                      self.local_network.reward: last_rewards,
                                                      self.local_network.state_in[0]: start_features[0],
                                                      self.local_network.state_in[1]: start_features[1],
                                                      self.local_network.state_in[2]: start_features[2],
                                                      self.Q_target: Q_targets,
                                                      self.action: actions})
        early_predictions = fetched[-2]
        pentultimate_prediction = early_predictions[-1]

        return pentultimate_prediction, predictions, fetched

    def train_answer(self, Qbatch, sess, should_compute_summary, fetched, predictions):
        last_states = [m[0] for m in Qbatch]
        actions = [m[1] for m in Qbatch]
        rewards = [m[2] for m in Qbatch]
        start_features = Qbatch[0][4]
        last_actions = [m[5] for m in Qbatch]
        last_rewards = [m[6] for m in Qbatch]
        states = [m[7] for m in Qbatch]
        features = [m[8] for m in Qbatch]
        next_features = features[0]

        predfeed_dict = {self.local_network.x: states,
                                  self.local_network.action: actions,
                                  self.local_network.reward: np.expand_dims(rewards, axis=1),
                                  self.local_network.state_in[0]: start_features[0],
                                  self.local_network.state_in[1]: start_features[1],
                                  self.local_network.state_in[2]: start_features[2],
                                  self.local_network.bs: len(Qbatch)
                                    }


        predfeed_dict.update(zip(self.local_network.var_list, self.local_network.target_weights))

        final_prediction = predictions



        targetfeed_dict = {self.local_qnetwork.x: states,
                           self.local_qnetwork.action: actions,
                           self.local_qnetwork.reward: np.expand_dims(rewards, axis=1),
                           self.local_qnetwork.pred_in: final_prediction,
                           self.local_qnetwork.bs: len(Qbatch)}

        pred_targs = sess.run(self.local_qnetwork.prediction_targs, feed_dict=targetfeed_dict)


        if should_compute_summary:
            predfetches = [self.predloss_sum, self.predmax, self.predval, self.prediction_train_op, self.global_step]
        else:
            predfetches = [self.prediction_train_op, self.global_step]

        predfetched = sess.run(predfetches, feed_dict={self.local_network.x: last_states,
                                                      self.local_network.action: last_actions,
                                                      self.local_network.reward: last_rewards,
                                                      self.local_network.state_in[0]: start_features[0],
                                                      self.local_network.state_in[1]: start_features[1],
                                                      self.local_network.state_in[2]: start_features[2],
                                                      self.prediction_target: pred_targs,
                                                      self.action: actions,
                                                      self.local_network.bs: len(Qbatch)})

        # adds summary for both main task and answer network
        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.add_summary(tf.Summary.FromString(predfetched[0]), predfetched[-1])
            self.summary_writer.add_summary(tf.Summary.FromString(predfetched[1]), predfetched[-1])
            self.summary_writer.add_summary(tf.Summary.FromString(predfetched[2]), predfetched[-1])
            self.summary_writer.flush()

        return final_prediction


    def train_question(self, Qbatch, sess, should_compute_summary, final_prediction, pentultimate_prediction):
        last_states = [m[0] for m in Qbatch]
        actions = [m[1] for m in Qbatch]
        rewards = [m[2] for m in Qbatch]
        last_actions = [m[5] for m in Qbatch]
        last_rewards = [m[6] for m in Qbatch]
        states = [m[7] for m in Qbatch]
        features = [m[8] for m in Qbatch]
        start_features = Qbatch[0][4]


        questionfeed_dict = {self.local_TDnetwork.x: states,
                             self.local_TDnetwork.action: actions,
                             self.local_TDnetwork.reward: np.expand_dims(rewards, axis=1),
                             self.local_TDnetwork.pred_in: final_prediction,
                             self.local_TDnetwork.bs: len(Qbatch),
                             self.local_TDnetwork.state_in[0]:start_features[0],
                             self.local_TDnetwork.state_in[1]:start_features[1]}

        questionfeed_dict.update(zip(self.local_network.var_list, self.local_network.target_weights))

        questionQvals, questionvals = sess.run((self.local_TDnetwork.Q, self.local_TDnetwork.vf), feed_dict= questionfeed_dict)

        questionQ_targets = []
        for i in range(len(Qbatch)):
            new_questionQ_target = 0.
            for j in range(len(Qbatch)-i):
                new_questionQ_target += (.99**j)*rewards[i+j]
            questionQ_targets.append(new_questionQ_target)
            if not Qbatch[-1][3]:
                questionQ_targets[-1] += (.99**(len(Qbatch)-i))*questionvals[-1]


        questionfeed_dict = {self.local_TDnetwork.x: last_states,
                             self.local_TDnetwork.action: last_actions,
                             self.local_TDnetwork.reward: last_rewards,
                             self.local_TDnetwork.pred_in: [pentultimate_prediction],
                             self.question_target: questionQ_targets,
                             self.action: actions,
                             self.local_TDnetwork.bs: len(Qbatch),
                             self.local_TDnetwork.state_in[0]:start_features[0],
                             self.local_TDnetwork.state_in[1]:start_features[1]}

        questionfetches = [self.question_train_op, self.global_step]

        sess.run(questionfetches, feed_dict=questionfeed_dict)

