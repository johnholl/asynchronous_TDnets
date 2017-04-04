from deepmind_lab import Lab
import numpy as np
import time
import tensorflow as tf
from pixel_helpers import calculate_intensity_change

class LabInterface():

    def __init__(self, level, observations=['RGB_INTERLACED'], config={'width': '84', 'height': '84', }):

        self.env = Lab(level=level, observations=observations, config=config)
        self.observation_space_shape = (int(config['height']), int(config['width']), 3)


        # For now, hardcoding number of discrete actions to 8:
        # look left, look right, look up, look down, strafe left, strafe right, forward, backward
        self.obs = np.zeros(shape=[84,84,3])
        print("interface built")

        self.ACTIONS = [self._action(-20, 0, 0, 0, 0, 0, 0),
          self._action(20, 0, 0, 0, 0, 0, 0),
          self._action(0, 10, 0, 0, 0, 0, 0),
          self._action(0, -10, 0, 0, 0, 0, 0),
          self._action(0, 0, -1, 0, 0, 0, 0),
          self._action(0, 0, 1, 0, 0, 0, 0),
          self._action(0, 0, 0, 1, 0, 0, 0),
          self._action(0, 0, 0, -1, 0, 0, 0)]
          # self._action(0, 0, 0, 0, 1, 0, 0),
          # self._action(0, 0, 0, 0, 0, 1, 0),
          # self._action(0, 0, 0, 0, 0, 0, 1)]

        self.num_actions = len(self.ACTIONS)



    def reset(self, seed=None):
        self.env.reset(seed=seed)
        obs = self.env.observations()['RGB_INTERLACED']/255.
        self.prev_obs = np.zeros(shape=[84,84,3])
        self.obs = obs
        return obs

    def step(self, action):
        rew = self.env.step(self.convert_int_to_action(action), num_steps=4)
        if self.env.is_running():
            self.obs = self.env.observations()['RGB_INTERLACED']/255.
        done = not self.env.is_running()

        return self.obs, rew, done

    def convert_int_to_action(self, index):
        action = self.ACTIONS[index]
        return action

    def _action(self, *entries):
        return np.array(entries, dtype=np.intc)



# class FeatureLabInterface():
#
#     def __init__(self, level, feature_calculator, observations=['RGB_INTERLACED'], config={'width': '84', 'height': '84', }):
#
#         self.env = Lab(level=level, observations=observations, config=config)
#         self.feature_calculator = feature_calculator
#
#         self.features = np.zeros(feature_calculator.num_features)
#         self.feature_change = np.zeros(feature_calculator.num_features)
#
#         self.observation_space_shape = (int(config['height']), int(config['width']), 3)
#
#
#         # For now, hardcoding number of discrete actions to 8:
#         # look left, look right, look up, look down, strafe left, strafe right, forward, backward
#         self.obs = np.zeros(shape=[84,84,3])
#         print("interface built")
#
#         self.ACTIONS = [self._action(-20, 0, 0, 0, 0, 0, 0),
#           self._action(20, 0, 0, 0, 0, 0, 0),
#           self._action(0, 10, 0, 0, 0, 0, 0),
#           self._action(0, -10, 0, 0, 0, 0, 0),
#           self._action(0, 0, -1, 0, 0, 0, 0),
#           self._action(0, 0, 1, 0, 0, 0, 0),
#           self._action(0, 0, 0, 1, 0, 0, 0),
#           self._action(0, 0, 0, -1, 0, 0, 0)]
#           # self._action(0, 0, 0, 0, 1, 0, 0),
#           # self._action(0, 0, 0, 0, 0, 1, 0),
#           # self._action(0, 0, 0, 0, 0, 0, 1)]
#
#         self.num_actions = len(self.ACTIONS)
#
#
#
#     def reset(self):
#         self.env.reset()
#         obs = self.env.observations()['RGB_INTERLACED']/255.
#         self.obs = obs
#
#         return obs
#
#     def step(self, action):
#         rew = self.env.step(self.convert_int_to_action(action), num_steps=4)
#         if self.env.is_running():
#             self.obs = self.env.observations()['RGB_INTERLACED']/255.
#             self.compute_feature_change()
#         done = not self.env.is_running()
#
#         return self.obs, self.feature_change, rew, done
#
#     def convert_int_to_action(self, index):
#         action = self.ACTIONS[index]
#         return action
#
#     def _action(self, *entries):
#         return np.array(entries, dtype=np.intc)
#
#     def compute_feature_change(self):
#         new_features = self.feature_calculator.calculate(self.obs, self.features)
#         self.feature_change = np.abs(new_features - self.features)
#

# class feature_calculator:
#     def __init__(self, restore_file):
#         # self.obs = tf.placeholder(), self.action, self.rew
#         # description of computation
#         # feature = result of computation
#         pass
#
#     def calculate(self, obs):
#         with tf.Session as sess:
#             sess.run(feature, feed_dict={self.obs: obs})


