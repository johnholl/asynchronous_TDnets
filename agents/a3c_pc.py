from __future__ import print_function
import distutils.version
import scipy.signal
import threading
from collections import namedtuple
import numpy as np
import six.moves.queue as queue
import tensorflow as tf
from models.model import GridLSTMPolicy
from helpers.pixel_helpers import calculate_intensity_change
import random
import sys

use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

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
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features, prev_action, prev_reward):
        self.prev_action += [prev_action]
        self.prev_reward += [prev_reward]
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

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
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, policy, num_local_steps):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
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
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)




def env_runner(env, policy, num_local_steps, summary_writer):
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
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            fetched = policy.act(last_state, prev_action, prev_reward, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]
            # argmax to convert from one-hot
            state, reward, terminal = env.step(action.argmax())
            pix_change = calculate_intensity_change(last_state, state, num_cuts=policy.grid_size)

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features, prev_action, [prev_reward])
            policy.update_replay_memory((last_state, action, reward, terminal,
                                         last_features, prev_action, [prev_reward],
                                         pix_change, state
                                         ))
            length += 1
            rewards += reward

            prev_action = action
            prev_reward = reward
            last_state = state
            last_features = features


            # summary = tf.Summary()
            # summary_writer.add_summary(summary, policy.global_step.eval())
            # summary_writer.flush()

            if terminal:
                summary = tf.Summary()
                summary.value.add(tag='episode_reward', simple_value=float(rewards))
                summary.value.add(tag='reward_per_timestep', simple_value=float(rewards)/float(length))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()
                terminal_end = True
                # if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                last_state = env.reset()
                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, prev_action, prev_reward, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class A3CPC(object):
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
            with tf.variable_scope("global"):
                self.network = GridLSTMPolicy(env.observation_space_shape, env.num_actions, grid_size=grid_size)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = GridLSTMPolicy(env.observation_space_shape, env.num_actions, grid_size=grid_size)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.num_actions], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)
            # entropy_reg = 10.**(np.random.uniform(-3.30103,-2.))
            entropy_reg = 0.001

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.agent_loss = pi_loss + 0.5 * vf_loss - entropy * entropy_reg

            # additionally define prediction loss
            # pixel_loss_weight = 10**(np.random.uniform(-2., -1.))
            pixel_loss_weight = 0.05

            self.prediction_target = tf.placeholder(tf.float32, [None, grid_size, grid_size])
            self.action = tf.placeholder(tf.float32, [None, env.num_actions])
            if use_tf12_api:
                prediction_readout = tf.reduce_sum(
                    tf.transpose(tf.multiply(
                    tf.transpose(pi.predictions, perm=[1,2,0,3]), self.action),
                    perm=[2,0,1,3]), reduction_indices=[3])
            else:
                prediction_readout = tf.reduce_sum(
                    tf.transpose(tf.mul(
                    tf.transpose(pi.predictions, perm=[1,2,0,3]), self.action),
                    perm=[2,0,1,3]), reduction_indices=[3])

            delta =  prediction_readout - self.prediction_target


            # clipping gradient to [-1, 1] amounts to using Huber loss
            self.prediction_loss = pixel_loss_weight*tf.reduce_sum(tf.reduce_mean(tf.select(
                                    tf.abs(delta) < 1,
                                    0.5 * tf.square(delta),
                                    tf.abs(delta) - 0.5),
                                    axis=0))

            self.avg_prediction = tf.reduce_mean(pi.predictions)
            self.max_prediction = tf.reduce_max(pi.predictions)

            self.loss = self.agent_loss + self.prediction_loss

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, 20)


            agent_grads = tf.gradients(self.agent_loss, pi.var_list)
            prediction_grads = tf.gradients(self.prediction_loss, pi.var_list)
            all_grads = np.concatenate((agent_grads, prediction_grads), axis=0)

            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                tf.summary.image("model/state", pi.x)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(agent_grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                for var in pi.var_list:
                    tf.summary.scalar("model/" + var.name + "_norm", tf.global_norm([var]))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.image_summary("model/state", pi.x)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(agent_grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                for var in pi.var_list:
                    tf.scalar_summary("model/" + var.name + "_norm", tf.global_norm([var]))
                self.summary_op = tf.merge_all_summaries()

            if use_tf12_api:
                self.pixloss_sum = tf.summary.scalar("model/pixel_loss", self.prediction_loss / bs)
                self.pixelval = tf.summary.scalar("model/pixel_value", self.avg_prediction)
                self.pixelmax = tf.summary.scalar("model/pixel_max", self.max_prediction )
            else:
                self.pixloss_sum = tf.scalar_summary("model/pixel_loss", self.prediction_loss / bs)
                self.pixelval = tf.scalar_summary("model/pixel_value", self.avg_prediction)
                self.pixelmax = tf.scalar_summary("model/pixel_max", self.max_prediction )

            agent_grads, _ = tf.clip_by_global_norm(agent_grads, 40.0)
            prediction_grads, _ = tf.clip_by_global_norm(prediction_grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            agent_grads_and_vars = list(zip(agent_grads, self.network.var_list))
            prediction_grads_and_vars = list(zip(prediction_grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.RMSPropOptimizer(1e-4)
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
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.agent_train_op, self.global_step]
        else:
            fetches = [self.agent_train_op, self.global_step]

        feed_dict = {
            self.local_network.x: batch.si,
            self.local_network.action: batch.pa,
            self.local_network.reward: batch.pr,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
        }

        fetched = sess.run(fetches, feed_dict=feed_dict)


        """
        Grab a minibatch from the replay memory. Set targets and update parameters
        from the prediction loss function"""
        if len(self.runner.policy.replay_memory) > 100:
            pixelbatch = []
            starting_pos = random.choice(range(len(self.runner.policy.replay_memory)-20))
            terminal = False
            bs = 0
            while not terminal and bs < 20:
                pixelbatch.append(self.runner.policy.replay_memory[starting_pos + bs])
                bs += 1
                terminal = pixelbatch[-1][3]

            last_states = [m[0] for m in pixelbatch]
            last_actions = [m[5] for m in pixelbatch]
            last_rewards = [m[6] for m in pixelbatch]
            rewards = [m[2] for m in pixelbatch]
            pixel_changes = [m[7] for m in pixelbatch]
            actions = [m[1] for m in pixelbatch]
            start_features = pixelbatch[0][4]
            states = [m[8] for m in pixelbatch]

            pixfeed_dict = {self.local_network.x: states,
                                      self.local_network.action: actions,
                                      self.local_network.reward: np.expand_dims(rewards, axis=1),
                                      self.local_network.state_in[0]: start_features[0],
                                      self.local_network.state_in[1]: start_features[1],
                                      self.local_network.bs: len(pixelbatch)
                                        }

            pixfeed_dict.update(zip(self.local_network.var_list, self.local_network.target_weights))

            prediction_values = sess.run(self.local_network.predictions, feed_dict=pixfeed_dict)

            max_prediction_value = np.max(prediction_values, axis=3)[-1]

            pred_targets = []
            pred_target_shape = np.shape(pixel_changes)[1:]
            for i in range(len(pixelbatch)):
                new_pred_target = np.zeros(shape=pred_target_shape)
                for j in range(len(pixelbatch)-i):
                    new_pred_target += (.99**j)*pixel_changes[i+j]
                pred_targets.append(new_pred_target)
                if not pixelbatch[-1][3]:
                    pred_targets[-1] += (.99**(len(pixelbatch)-i))*max_prediction_value

            if should_compute_summary:
                pixfetches = [self.pixloss_sum, self.pixelval, self.pixelmax, self.prediction_train_op, self.global_step]
            else:
                pixfetches = [self.prediction_train_op, self.global_step]

            pixfetched = sess.run(pixfetches, feed_dict={self.local_network.x: last_states,
                                                          self.local_network.action: last_actions,
                                                          self.local_network.reward: last_rewards,
                                                          self.local_network.state_in[0]: start_features[0],
                                                          self.local_network.state_in[1]: start_features[1],
                                                          self.prediction_target: pred_targets,
                                                          self.action: actions,
                                                          self.local_network.bs: len(pixelbatch)})

            if should_compute_summary:
                self.summary_writer.add_summary(tf.Summary.FromString(pixfetched[0]), pixfetched[-1])
                self.summary_writer.add_summary(tf.Summary.FromString(pixfetched[1]), pixfetched[-1])
                self.summary_writer.add_summary(tf.Summary.FromString(pixfetched[2]), pixfetched[-1])
                self.summary_writer.flush()


        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
