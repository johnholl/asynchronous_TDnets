from __future__ import print_function
import distutils.version
import scipy.signal
import threading
from collections import namedtuple
import numpy as np
import six.moves.queue as queue
import tensorflow as tf
from pixel_helpers import calculate_intensity_change

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
            fetched = policy.act(last_state, prev_action, prev_reward, last_features[0], last_features[1], last_features[2])
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
    last_pix_change = np.zeros(shape=[20,20])

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
                        pix_change, state, prediction, next_value, next_prediction, last_pix_change)

            length += 1
            rewards += reward

            prev_action = action
            prev_reward = reward
            last_state = state
            last_beh_features = beh_features
            last_prediction = prediction
            last_pix_change = pix_change

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
