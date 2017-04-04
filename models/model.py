import numpy as np
import tensorflow as tf
# import tensorflow.contrib.rnn as rnn
from helpers.layer_helpers import weight_variable, conv2d, flatten, linear, normalized_columns_initializer, categorical_sample
from helpers.custom_rnn_cells import GridPredictionLSTMCell, PredictionLSTMStateTuple


'''
The following class it be used when the tasks are (1) A3C and (2) grid pixel intensity change predictions,
and there is a feedback of predictions to the LSTM cell.
'''
class GridPredictionLSTMPolicy(object):
    def __init__(self, ob_space, ac_space, replay_size=2000, grid_size=20):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.bs = tf.placeholder(dtype=tf.int32)
        self.replay_memory = []
        self.replay_size = replay_size
        self.grid_size = grid_size

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(concat_dim=1, values=[x, self.action, self.reward])
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
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
        x = tf.reshape(lstm_outputs, [-1, size])

        # Actor critic branch
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :], lstm_pred[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]

        # Auxiliary branch
        self.predictions = tf.reshape(lstm_pred, shape=[-1, grid_size, grid_size, ac_space])
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.target_weights = []

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, prev_a, prev_r, c, h, pred):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                         self.state_in[0]: c, self.state_in[1]: h,
                         self.state_in[2]: pred})

    def value(self, ob, prev_a, prev_r, c, h, pred):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h,
                                  self.state_in[2]: pred})[0]

    def update_replay_memory(self, tuple):
        # appends tuple and pops old tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def update_target_weights(self):
        sess = tf.get_default_session()
        self.target_weights = sess.run(self.var_list)


'''
The following class it be used when the tasks are (1) A3C and (2) grid pixel intensity change predictions,
and there is NOT a feedback of predictions to the LSTM cell ie this is a replication from the Jaderberg paper.
'''
class GridLSTMPolicy(object):
    def __init__(self, ob_space, ac_space, mode="Grid", replay_size=2000, grid_size=20):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.replay_memory = []
        self.replay_size = replay_size
        self.grid_size = grid_size
        self.bs = tf.placeholder(dtype=tf.int32)

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(concat_dim=1, values=[x, self.action, self.reward])
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(x, [0])

        size = 256
        lstm = tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # Actor critic branch
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]

        # Auxiliary branch
        if mode == "Grid":
            y = linear(x, 32*(grid_size-10)*(grid_size-10), 'auxbranch', normalized_columns_initializer(0.1))
            y = tf.reshape(y, shape=[-1, grid_size-10, grid_size-10, 32])
            deconv_weights = weight_variable(shape=[4, 4, ac_space, 32], name='deconvweights')
            self.predictions = tf.nn.conv2d_transpose(y, deconv_weights,
                                                      output_shape=[self.bs, grid_size, grid_size, ac_space],
                                                      strides=[1,2,2,1], padding='SAME')

        if mode == "Features":

            pass

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.target_weights = []


    def get_initial_features(self):
        return self.state_init

    def act(self, ob, prev_a, prev_r, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                        self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, prev_a, prev_r, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h})[0]

    def update_replay_memory(self, tuple):
        # appends tuple and pops old tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def update_target_weights(self):
        sess = tf.get_default_session()
        self.target_weights = sess.run(self.var_list)


'''
Baseline A3C agent policy
'''
class A3CLSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(concat_dim=1, values=[x, self.action, self.reward])
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(x, [0])

        size = 256
        lstm = tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state

        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, prev_a, prev_r, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                        self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, prev_a, prev_r, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h})[0]


'''
The following class it be used when the tasks are (1) A3C and (2) convolutional feature predictions,
and there is NOT a feedback of predictions to the LSTM cell ie this is a replication from the Jaderberg paper.
'''
class FeatureLSTMPolicy(object):
    def __init__(self, ob_space, ac_space, mode="Grid", replay_size=2000, grid_size=20):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.replay_memory = []
        self.replay_size = replay_size
        self.grid_size = grid_size
        self.bs = tf.placeholder(dtype=tf.int32)

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
        x = self.conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
        x = flatten(x)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(concat_dim=1, values=[x, self.action, self.reward])
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(x, [0])

        size = 256
        lstm = tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # Actor critic branch
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]

        # Auxiliary branch
        if mode == "Grid":
            y = linear(x, 32*(grid_size-7)*(grid_size-7), 'auxbranch', normalized_columns_initializer(0.1))
            y = tf.reshape(y, shape=[-1, grid_size-7, grid_size-7, 32])
            deconv_weights = weight_variable(shape=[4, 4, ac_space, 32], name='deconvweights')
            self.predictions = tf.nn.conv2d_transpose(y, deconv_weights,
                                                      output_shape=[self.bs, grid_size, grid_size, ac_space],
                                                      strides=[1,2,2,1], padding='SAME')

        if mode == "Features":
            y = linear(x, 16*9*9, 'auxbranch', normalized_columns_initializer(0.1))
            y = tf.reshape(y, shape=[-1, 9, 9, 16])
            deconv_weights = weight_variable(shape=[1, 1, 32*ac_space, 16], name='deconvweights')
            predictions = tf.nn.conv2d_transpose(y, deconv_weights,
                                                      output_shape=[self.bs, 9, 9, 32*ac_space],
                                                      strides=[1,1,1,1], padding='SAME')
            self.predictions = tf.reshape(predictions, shape=[-1, 9, 9, 32, ac_space])


        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.target_weights = []


    def get_initial_features(self):
        return self.state_init

    def act(self, ob, prev_a, prev_r, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf, self.conv_features] + self.state_out,
                        {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                        self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, prev_a, prev_r, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h})[0]

    def update_replay_memory(self, tuple):
        # appends tuple and pops old tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def update_target_weights(self):
        sess = tf.get_default_session()
        self.target_weights = sess.run(self.var_list)




