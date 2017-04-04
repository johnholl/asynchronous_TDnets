from __future__ import print_function
import distutils.version
import scipy.signal
import threading
from collections import namedtuple
import numpy as np
import six.moves.queue as queue
import tensorflow as tf
from models.QA_model import ConvTDNet
from models.trained_models import DeterministicBehaviorAgent
from helpers.checkpoint_helpers import load_obj
import pickle


use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')


MC_value_dict = load_obj("MC_10000_det_val_seed0")


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_pa = np.asarray(rollout.prev_action)
    batch_pr = np.asarray(rollout.prev_reward)
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features, batch_pa, batch_pr)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features", "pa", "pr"])

class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.prev_action = []
        self.prev_reward = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = []
        self.last_predictions = []
        self.next_states = []
        self.predictions = []
        self.next_values = []
        self.next_predictions = []

    def add(self, state, action, reward, value, terminal, features, prev_action, prev_reward, next_state,
            next_feature, next_value, next_prediction):
        self.prev_action += [prev_action]
        self.prev_reward += [prev_reward]
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal += [terminal]
        self.last_predictions += [features]
        self.next_states += [next_state]
        self.predictions += [next_feature]
        self.next_values += [next_value]
        self.next_predictions += [next_prediction]

    def extend(self, other):
        assert not self.terminal
        self.prev_action.extend(other.prev_action)
        self.prev_reward.extend(other.prev_reward)
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.last_predictions.extend(other.last_predictions)
        self.next_states.extend(other.next_states)
        self.predictions.extend(other.predictions)
        self.next_values.extend(other.next_values)
        self.next_predictions.extend(other.next_predictions)


class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, network, beh_policy, num_local_steps):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.network = network
        self.beh_policy = beh_policy
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
        rollout_provider = tdnet_env_runner(self.env, self.network, self.beh_policy, self.num_local_steps, self.summary_writer)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)




def tdnet_env_runner(env, network, beh_policy, num_local_steps, summary_writer):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    last_state, last_pc = env.reset(seed=0)
    last_beh_features = beh_policy.get_initial_features()
    last_prediction = network.get_initial_features()
    length = 0
    rewards = 0
    prev_reward = 0
    value_ = 0
    prev_action = np.zeros(shape=[env.num_actions])

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = beh_policy.act(last_state, prev_action, prev_reward, *last_beh_features)
            action, beh_features = fetched[0], fetched[2:]
            value_, prediction = network.value(last_state, prev_action, prev_reward, last_prediction)

            action_index = np.where(action==1)
            state, reward, terminal, pix_change = env.step(action_index[0][0])

            if length > 158:
                terminal = True

            if not terminal:
                next_value, next_prediction = network.value(state, action, reward, prediction)

            else:
                next_value = 0.
                next_prediction = np.zeros(shape=[1,400])

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_prediction, prev_action, [prev_reward],
                        state, prediction, next_value, next_prediction)

            length += 1
            rewards += reward

            prev_action = action
            prev_reward = reward
            last_state = state
            last_beh_features = beh_features
            last_prediction = prediction

            if terminal:
                summary = tf.Summary()
                summary.value.add(tag='episode_reward', simple_value=float(rewards))
                summary.value.add(tag='reward_per_timestep', simple_value=float(rewards)/float(length))
                summary_writer.add_summary(summary, network.global_step.eval())
                summary_writer.flush()
                terminal_end = True
                # if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                last_state, last_pc = env.reset(seed=0)
                last_beh_features = beh_policy.get_initial_features()
                last_prediction = network.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = value_

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class FFAgent(object):
    def __init__(self, env, task, grid_size=20):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            s = tf.Session()
            self.beh_pi = DeterministicBehaviorAgent(ob_space=env.observation_space_shape, ac_space=env.num_actions,
                                   sess=s,
                                   ckpt_file="/home/john/tmp/train/model.ckpt-11471320")

            with tf.variable_scope("learner"):
                self.network = ConvTDNet(env.observation_space_shape, env.num_actions)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = ConvTDNet(env.observation_space_shape, env.num_actions)
                pi.global_step = self.global_step

                self.val_target = tf.placeholder(tf.float32, [None])


            val_delta = pi.val - self.val_target

            self.agent_loss = tf.reduce_sum(tf.where(
                                    tf.abs(val_delta) < 1,
                                    0.5 * tf.square(val_delta),
                                    tf.abs(val_delta) - 0.5),
                                    axis=0)

            pixel_loss_weight = 0.05

            self.prediction_target = tf.placeholder(tf.float32, [None, grid_size * grid_size])

            pix_delta = pi.predictions - self.prediction_target

            self.prediction_loss = pixel_loss_weight*tf.reduce_sum(tf.where(
                                    tf.abs(pix_delta) < 1,
                                    0.5 * tf.square(pix_delta),
                                    tf.abs(pix_delta) - 0.5),
                                    axis=0)


            self.avg_prediction = tf.reduce_mean(pi.predictions)
            self.max_prediction = tf.reduce_max(pi.predictions)


            self.ground_value = tf.placeholder(tf.float32, [None])

            self.mc_valueloss = tf.reduce_mean(tf.square(pi.val - self.ground_value))

            self.loss = self.agent_loss + self.prediction_loss

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.


            self.runner = RunnerThread(env, pi, self.beh_pi, 20)


            bs = tf.to_float(tf.shape(pi.x)[0])

            agent_grads = tf.gradients(self.agent_loss, pi.var_list)
            prediction_grads = tf.gradients(self.prediction_loss, pi.var_list)
            all_grads = np.concatenate((agent_grads, prediction_grads), axis=0)


            # Summary info

            tf.summary.scalar("model/Q_loss", self.agent_loss / bs)
            tf.summary.scalar("model/avg_V", tf.reduce_mean(pi.val))
            tf.summary.scalar("model/avg_groundvalue_loss", self.mc_valueloss)
            tf.summary.image("model/state", pi.x*255, max_outputs=12)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(agent_grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
            for var in pi.var_list:
                tf.summary.scalar("model/" + var.name + "_norm", tf.global_norm([var]))
            self.summary_op = tf.summary.merge_all()

            self.pixloss_sum = tf.summary.scalar("model/pixel_loss", self.prediction_loss / bs)
            self.pixelval = tf.summary.scalar("model/pixel_value", self.avg_prediction)
            self.pixelmax = tf.summary.scalar("model/pixel_max", self.max_prediction )



            agent_grads, _ = tf.clip_by_global_norm(agent_grads, 40.0)
            prediction_grads, _ = tf.clip_by_global_norm(prediction_grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            agent_grads_and_vars = list(zip(agent_grads, self.network.var_list))
            prediction_grads_and_vars = list(zip(prediction_grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            # thread_lr = 10.**(np.random.uniform(-4, -2.30103))
            thread_lr = .001
            opt = tf.train.RMSPropOptimizer(thread_lr)
            self.agent_train_op = tf.group(opt.apply_gradients(agent_grads_and_vars), inc_step)
            self.prediction_train_op = opt.apply_gradients(prediction_grads_and_vars)
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.sync)  # copy weights from shared to local
        print(sess.run(self.normalized_discount_factors))

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        if should_compute_summary:
            fetches = [self.summary_op, self.pixelmax, self.pixelval, self.agent_train_op, self.prediction_train_op, self.global_step]
        else:
            fetches = [self.agent_train_op, self.global_step]

        rollout = self.pull_batch_from_queue()



        batch_pa = np.asarray(rollout.prev_action)
        batch_pr = np.asarray(rollout.prev_reward)
        batch_si = np.asarray(rollout.states)
        batch_rewards = np.asarray(rollout.rewards)
        batch_pixc = np.asarray(rollout.pix_changes)
        batch_pixv = np.asarray(rollout.next_features)
        batch_features = np.asarray(rollout.features)
        batch_features = np.squeeze(batch_features, axis=[1])
        batch_done = np.asarray(rollout.terminal)
        batch_val = np.asarray(rollout.values)
        batch_nextval = np.asarray(rollout.next_values)
        batch_nextpred = np.asarray(rollout.next_predictions)
        batch_nextpred = np.squeeze(batch_nextpred, axis=1)
        batch_action = np.asarray(rollout.actions)
        batch_lastpixc = np.asarray(rollout.last_pix_changes)

        batch_pixv = np.squeeze(batch_pixv, axis=[1])
        next_value, next_prediction = self.network.value_with_target(batch_si, batch_action, batch_rewards, batch_pixv)

        batch_pixc = np.reshape(batch_pixc, [np.shape(batch_pixc)[0], np.shape(batch_pixc)[1]*np.shape(batch_pixc)[2]])
        batch_lastpixc = np.reshape(batch_lastpixc, [np.shape(batch_lastpixc)[0], np.shape(batch_lastpixc)[1]*np.shape(batch_lastpixc)[2]])


        value_targets = discount(batch_rewards, .99)
        prediction_targets = discount(batch_pixc, .99)
        for i in range(len(batch_si)):
            value_targets[i] += (.99**(20-i))*batch_nextval[-1]
            prediction_targets[i] += (.99**(20-i))*batch_nextpred[-1]
            if batch_done[i] and (i != 19):
                print("o crap")
        # raw_input("Press Enter")
        # print(batch_val)
        # print(value_targets)



        feed_dict = {
            self.local_network.x: batch_si,
            self.local_network.action: batch_pa,
            self.local_network.reward: batch_pr,
            self.local_network.obs_features: batch_pixc,
            self.local_network.initial_pred: [batch_nextpred[-1]],
            self.val_target: value_targets,
            self.prediction_target: prediction_targets,
            self.local_network.bs: len(batch_si),
            self.ground_value: [MC_value_dict[str(last_state)][1] for last_state in batch_si],

        }

        fetched = sess.run(fetches, feed_dict=feed_dict)



        # feed_dict = {
        #     self.local_network.x: batch_si[:-1],
        #     self.local_network.action: batch_pa[:-1],
        #     self.local_network.reward: batch_pr[:-1],
        #     self.local_network.pred: batch_features[:-1],
        #     self.l
        # }
        #
        # fetched = sess.run(tf.shape(self.local_network.testy), feed_dict=feed_dict)
        # print(fetched)


        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[1]), fetched[-1])
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[2]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
        print(self.local_steps)
