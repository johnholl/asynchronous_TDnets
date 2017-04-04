import tensorflow as tf
from helpers.layer_helpers import conv2d, flatten, linear, normalized_columns_initializer
import numpy as np


class FFQPCR_Policy(object):
    def __init__(self, ob_space, ac_space, replay_size=2000, grid_size=20, ckpt_file=None):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.pred = tf.placeholder(tf.float32, [None, grid_size, grid_size, ac_space])

        self.ac_space = ac_space
        self.bs = tf.placeholder(dtype=tf.int32)
        self.replay_memory = []
        self.replay_size = replay_size
        self.grid_size = grid_size

        self.prob = 1.
        self.final_prob = 0.1
        self.anneal_rate = .00000018

        self.num_actions = ac_space

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        pred = flatten(self.pred)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(axis=1, values=[x, self.action, self.reward, self.pred])

        x = tf.nn.relu(linear(x, 256, "l4", normalized_columns_initializer(0.1)))

        self.Q = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reduce_max(self.Q, axis=[1])

        # Auxiliary branch
        y = linear(x, 32*(self.grid_size-10)*(self.grid_size-10), 'auxbranch', normalized_columns_initializer(0.1))
        y = tf.reshape(y, shape=[-1, self.grid_size-10, self.grid_size-10, 32])
        deconv_weights = tf.get_variable("deconv" + "/w", [4, 4, ac_space, 32])
        self.predictions = tf.nn.conv2d_transpose(y, deconv_weights,
                                                output_shape=[1, self.grid_size, self.grid_size, self.ac_space],
                                                strides=[1,2,2,1], padding='SAME')

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.target_weights = []


    def value(self, ob, prev_a, prev_r, pred):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]], self.pred: pred})[0]
