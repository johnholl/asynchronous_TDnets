from helpers.custom_rnn_cells import PredictionLSTMStateTuple, GridPredictionLSTMCell, BasicPredictionLSTMCell, _linear, DiscountRNNCell
from helpers.layer_helpers import normalized_columns_initializer, linear, conv2d, flatten, categorical_sample
from helpers.rnn import old_dynamic_rnn
# from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicRNNCell
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops.math_ops import tanh, sigmoid
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs





class GridQAnswerNet:
    def __init__(self, ob_space, ac_space, replay_size=2000, grid_size=20):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
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
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(axis=1, values=[x, self.action, self.reward])
        x = tf.expand_dims(x, [0])

        size = 256
        lstm = GridPredictionLSTMCell(size, state_is_tuple=True, ac_space=ac_space,
                                       grid_size=20)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        pred_init = np.zeros((1, lstm.state_size.pred), np.float32)
        self.state_init = [c_init, h_init, pred_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        pred_in = tf.placeholder(tf.float32, [1, lstm.state_size.pred])
        self.state_in = [c_in, h_in, pred_in]

        state_in = PredictionLSTMStateTuple(c_in, h_in, pred_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h, lstm_pred = lstm_state

        # Q learning branch
        x = tf.reshape(lstm_outputs, [-1, size])
        self.Q = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reduce_max(self.Q, axis=[1])

        self.state_out = [lstm_c[:1, :], lstm_h[:1, :], lstm_pred[:1, :]]

        # Auxiliary branch
        self.predictions = tf.reshape(lstm_pred, shape=[-1, grid_size, grid_size, ac_space])
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.target_weights = []

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, prev_a, prev_r, c, h, pred):
        sess = tf.get_default_session()
        self.update_exploration()
        fetched = sess.run([self.Q, self.vf] + self.state_out,
                                {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                 self.state_in[0]: c, self.state_in[1]: h,
                                 self.state_in[2]: pred})
        qvals = fetched[0]

        if np.random.uniform > self.prob:
            action = np.argmax(qvals)
        else:
            action = np.random.choice(range(self.num_actions))

        return action, fetched[1], fetched[2:]

    def value(self, ob, prev_a, prev_r, c, h, pred):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h,
                                  self.state_in[2]: pred})[0]

    def update_replay_memory(self, tuple):
        # appends tuple and pops oldest tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def update_exploration(self):
        if self.prob > self.final_prob:
            self.prob -= self.anneal_rate

    def update_target_weights(self):
        sess = tf.get_default_session()
        self.target_weights = sess.run(self.var_list)


class GridQQuestionNet:
    def __init__(self, ob_space, ac_space, grid_size=20):
        self.pixel_changes = tf.placeholder(tf.float32, [None] + list(ob_space))


        # self.acs = tf.placeholder(tf.float32, [None, ac_space])
        # self.rewards = tf.placeholder(tf.float32, [None, 1])
        self.bs = tf.placeholder(tf.int32)
        x = flatten(self.obs)
        x = tf.concat(axis=1, values=[x, self.acs, self.rewards])

        rnn = tf.nn.rnn_cell.BasicRNNCell(num_units = 100)
        self.state_size = rnn.state_size
        step_size = tf.shape(self.obs)[:1]

        self.pred_init = np.zeros((1, rnn.state_size), np.float32)
        pred_in = tf.placeholder(tf.float32, [1, rnn.state_size])

        outputs, state = tf.nn.dynamic_rnn(
            rnn, x, initial_state=pred_in, sequence_length=step_size,
            time_major=False)
        self.pred_targets = tf.reshape(state, shape=[-1, grid_size*grid_size*ac_space])


    def get_initial_features(self):
        return self.pred_init



class GridQTDNet:
    def __init__(self, ob_space, ac_space, grid_size=20):
        self.qnet = GridQQuestionNet(ob_space=ob_space, ac_space=ac_space, grid_size=grid_size)
        self.anet = GridQAnswerNet(ob_space=ob_space, ac_space=ac_space, grid_size=grid_size)


        pass


class BasicQAnswerNet:
    def __init__(self, ob_space, ac_space, replay_size=2000, num_predictions=100):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.bs = tf.placeholder(dtype=tf.int32)
        self.replay_memory = []
        self.replay_size = replay_size

        self.prob = 1.
        self.final_prob = 0.1
        self.anneal_rate = .00000018

        self.num_actions = ac_space

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(axis=1, values=[x, self.action, self.reward])
        x = tf.expand_dims(x, [0])

        size = 256
        lstm = BasicPredictionLSTMCell(size, state_is_tuple=True, num_pred=num_predictions)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        pred_init = np.zeros((1, lstm.state_size.pred), np.float32)
        self.state_init = [c_init, h_init, pred_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        pred_in = tf.placeholder(tf.float32, [1, lstm.state_size.pred])
        self.state_in = [c_in, h_in, pred_in]

        state_in = PredictionLSTMStateTuple(c_in, h_in, pred_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h, lstm_pred = lstm_state

        # Q learning branch
        x = tf.reshape(lstm_outputs, [-1, size])
        self.Q = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reduce_max(self.Q, axis=[1])

        self.state_out = [lstm_c[:1, :], lstm_h[:1, :], lstm_pred[:1, :]]

        # Auxiliary branch
        self.predictions = lstm_pred
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.target_weights = []

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, prev_a, prev_r, c, h, pred):
        sess = tf.get_default_session()
        self.update_exploration()
        fetched = sess.run([self.Q, self.vf] + self.state_out,
                                {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                 self.state_in[0]: c, self.state_in[1]: h,
                                 self.state_in[2]: pred})
        qvals = fetched[0]

        if np.random.uniform > self.prob:
            action = np.argmax(qvals)
        else:
            action = np.random.choice(range(self.num_actions))

        return action, fetched[1], fetched[2:]

    def value(self, ob, prev_a, prev_r, c, h, pred):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h,
                                  self.state_in[2]: pred})[0]

    def update_replay_memory(self, tuple):
        # appends tuple and pops oldest tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def update_exploration(self):
        if self.prob > self.final_prob:
            self.prob -= self.anneal_rate

    def update_target_weights(self):
        sess = tf.get_default_session()
        self.target_weights = sess.run(self.var_list)



class BasicQQuestionNet:
    def __init__(self, ob_space, ac_space, replay_size=2000, num_predictions=100):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        x = tf.reverse(x, axis=[0])
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])

        self.bs = tf.placeholder(dtype=tf.int32)
        self.replay_memory = []
        self.replay_size = replay_size

        self.num_actions = ac_space

        x = flatten(x)
        x = tf.concat(axis=1,
                values=[x, tf.reverse(self.action, axis=[0]), tf.reverse(self.reward, axis=[0])])
        x = tf.expand_dims(x, [0])


        size = num_predictions # number of predictions
        rnn = BasicRNNCell(num_units=size, activation=sigmoid)
        self.pred_init = np.zeros((1, size), np.float32)
        self.pred_in = tf.placeholder(tf.float32, [1, size])

        step_size = tf.shape(self.x)[:1]

        rnn_output, rnn_state = tf.nn.dynamic_rnn(
            rnn, x, initial_state=self.pred_in, sequence_length=step_size,
            time_major=False)

        self.prediction_targs = tf.reshape(rnn_output, shape=[-1, size])
        self.final_prediction_targs = tf.slice(self.prediction_targs, begin=[-1,0], size=[1,-1])
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)



class BasicQTDNet:
    def __init__(self, ob_space, ac_space, replay_size=2000, num_predictions=100):

        self.num_predictions = num_predictions

        with tf.variable_scope("Question"):
            self.qnet = BasicQQuestionNet(ob_space=ob_space, ac_space=ac_space)

        with tf.variable_scope("Answer"):
            self.anet = BasicQAnswerNet(ob_space=ob_space, ac_space=ac_space)

        with tf.variable_scope("Question", reuse=True):

            # Note: this observation sequence should be the flipped version of the original one
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
            self.action = tf.placeholder(tf.float32, [None, ac_space])
            self.reward = tf.placeholder(tf.float32, [None, 1])
            self.bs = tf.placeholder(dtype=tf.int32)

            self.num_actions = ac_space

            x = tf.reverse(x, axis=[0])
            rev_action = tf.reverse(self.action, axis=[0])
            rev_reward = tf.reverse(self.reward, axis=[0])
            x=flatten(x)
            x = tf.concat(axis=1, values=[x, rev_action, rev_reward])
            x = tf.expand_dims(x, [0])

            size = 100 # number of predictions
            rnn = BasicRNNCell(num_units=size)
            self.pred_init = np.zeros((1, size), np.float32)
            self.pred_in = tf.placeholder(tf.float32, [1, size])

            step_size = tf.shape(self.x)[:1]

            rnn_output, rnn_state = tf.nn.dynamic_rnn(
                rnn, x, initial_state=self.pred_in, sequence_length=step_size,
                time_major=False)

            self.prediction_targs = tf.reshape(rnn_output, shape=[-1, size])

            # shape [1, size]. This will be the initial prediction in the answer network
            self.final_prediction_targs = tf.slice(self.prediction_targs, begin=[tf.shape(self.prediction_targs)[0]-1,0], size=[1,-1])

        with tf.variable_scope("Answer", reuse=True):
            x = self.x
            x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
            x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
            x = flatten(x)
            x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
            x = tf.concat(axis=1, values=[x, self.action, self.reward])
            x = tf.expand_dims(x, [0])

            size = 256
            lstm = BasicPredictionLSTMCell(num_units=256, state_is_tuple=True, num_pred=100)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            pred_init = np.zeros((1, lstm.state_size.pred), np.float32)
            self.state_init = [c_init, h_init, pred_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in, self.final_prediction_targs]

            state_in = PredictionLSTMStateTuple(c_in, h_in, self.final_prediction_targs)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h, lstm_pred = lstm_state

            # Q learning branch
            x = tf.reshape(lstm_outputs, [-1, size])
            self.Q = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.vf = tf.reduce_max(self.Q, axis=[1])

            self.state_out = [lstm_c[:1, :], lstm_h[:1, :], lstm_pred[:1, :]]

            # Auxiliary branch
            self.predictions = lstm_pred
            self.target_weights = []

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)





class BasicRNNCell(RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    with vs.variable_scope(scope or "basic_rnn_cell"):
      output = self._activation(
          _linear([inputs, state], self._num_units, True, scope=scope))
    return output, output




# This will be a td net where the question net computes finite sum over 2 layer convolutional features.
#
class ConvTDNet:
    def __init__(self, ob_space, ac_space, future_steps=5, pred_per_step=10, ckpt=None):
        self.ckpt_file = ckpt
        self.replay_memory = []
        self.future_steps = future_steps
        self.bs = tf.placeholder(dtype=tf.int32)
        self.num_actions = ac_space
        self.pred_per_step = pred_per_step
        self.num_pred = pred_per_step*future_steps
        self.use_target = tf.placeholder(dtype=tf.bool)
        self.default_last_prediction = tf.zeros(shape=[self.bs, self.num_pred])
        self.default_target_obs = tf.zeros(shape=[self.bs] + [self.future_steps] + list(ob_space))

        ## Inputs to the question network
        with tf.variable_scope("Question"):
            self.target_obs = tf.placeholder(tf.float32, [None] + [self.future_steps] + list(ob_space))
            feature_list = []
            for i in range(self.future_steps):
                obs = tf.squeeze(tf.slice(self.target_obs, begin=[0,i,0,0,0], size=[-1,1,-1,-1,-1]), axis=[1])
                z = tf.nn.relu(conv2d(obs, 16, "ql1"+str(i), [8, 8], [4, 4]))
                z = tf.nn.relu(conv2d(z, 32, "ql2"+str(i), [4, 4], [2, 2]))
                z = flatten(z)
                z = tf.nn.relu(linear(z, self.pred_per_step, "ql3" + str(i), normalized_columns_initializer(0.1)))
                feature_list.append(z)
            self.target_predictions = tf.concat(feature_list, axis=1)

        with tf.variable_scope("Main"):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
            self.action = tf.placeholder(tf.float32, [None, ac_space])
            self.reward = tf.placeholder(tf.float32, [None, 1])
            self.last_prediction = tf.placeholder(tf.float32, [None, self.num_pred])

            x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
            x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
            x = flatten(x)
            x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))


            p = tf.nn.l2_normalize(self.last_prediction, dim=1)
            p = tf.nn.tanh(linear(p, 256, 'encode_pred', normalized_columns_initializer(0.1)))

            xmain = tf.concat(axis=1, values=[x, self.action, self.reward])
            xaux = tf.concat(axis=1, values=[x, self.action, self.reward, p])

            xmain = tf.nn.relu(linear(xmain, 256, "l4", normalized_columns_initializer(0.1)))

            # Auxiliary branch
            y = tf.nn.relu(linear(xaux, 256, 'auxbranch_l1', normalized_columns_initializer(0.1)))
            self.approx_predictions = linear(y, self.num_pred, 'auxbranch_l2', normalized_columns_initializer(0.1))

            self.predictions = tf.where(self.use_target, self.target_predictions, self.approx_predictions)

            x = tf.concat(axis=1, values=[xmain, self.predictions])

            val = linear(x, 1, "value", normalized_columns_initializer(0.01))
            self.val = tf.reshape(val, shape=[-1])


        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.target_weights = []

    def get_initial_features(self):
        return np.zeros((1, self.num_pred), np.float32)

    def restore(self, sess, saver):
        if self.ckpt_file is not None:
            saver.restore(sess, self.ckpt_file)
            self.target_weights = sess.run(self.var_list)

    def targ_value(self, ob, prev_a, prev_r, target_obs, current_session=None):
        if current_session is not None:
            sess = current_session
        else:
            sess = tf.get_default_session()
        reward_value, prediction_values = sess.run((self.val, self.predictions), {self.x: [ob], self.action: [prev_a],
                                                                                  self.reward: [[prev_r]], self.bs:1,
                                                                                  self.target_obs: target_obs,
                                                                                  self.use_target: True})

        return reward_value, prediction_values

    def approx_value(self, ob, prev_a, prev_r, last_pred, current_session=None):
        if current_session is not None:
            sess = current_session
        else:
            sess = tf.get_default_session()
        reward_value, prediction_values = sess.run((self.val, self.predictions), {self.x: [ob], self.action: [prev_a],
                                                                                  self.reward: [[prev_r]], self.bs:1,
                                                                                  self.last_prediction: last_pred,
                                                                                  self.use_target: False})

        return reward_value, prediction_values

    def update_target_weights(self, current_session=None):
        if current_session is not None:
            sess = current_session
        else:
            sess = tf.get_default_session()

        self.target_weights = sess.run(self.var_list)

    def value_with_target(self, ob, prev_a, prev_r, pred, current_session=None):
        if current_session is not None:
            sess = current_session
        else:
            sess = tf.get_default_session()

        feed_dict = {self.x: ob, self.action: prev_a, self.reward: np.expand_dims(prev_r, axis=1), self.bs:np.shape(ob)[0], self.pred: pred}
        feed_dict.update(zip(self.var_list, self.target_weights))
        reward_value, prediction_values = sess.run((self.val, self.predictions), feed_dict=feed_dict)

        return reward_value, prediction_values





#
# tdnet = BasicQTDNet(ob_space=[84,84,3], ac_space=8)
#
# print([v.name for v in tdnet.var_list])
# #
# #
# #
# # sess = tf.Session()
# # sess.run(tf.initialize_all_variables())
# # x = np.zeros(shape=[20,84,84,3])
# # ac = np.zeros(shape=[20,8])
# # rew = np.zeros(shape=[20,1])
# # pin = np.zeros(shape=[1, 100])
# # bs = 20
# # result = sess.run(tf.shape(qnet.final_prediction_targs), feed_dict={qnet.x:x, qnet.action:ac, qnet.reward:rew, qnet.bs:bs,
# #                                                               qnet.pred_in:pin})
# # print(result)



