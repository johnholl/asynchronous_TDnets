from tensorflow.python.platform import tf_logging as logging
#from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
import collections
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn_ops



class GridPredictionLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, ac_space, grid_size=20, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tanh):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 3-tuples of
        the `c_state`, `h_state`, and `pred_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._num_predictions = grid_size*grid_size*ac_space
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self.ac_space = ac_space
    self.grid_size = grid_size

  @property
  def state_size(self):
    return (PredictionLSTMStateTuple(self._num_units, self._num_units, self._num_predictions)
            if self._state_is_tuple else 2 * self._num_units + self._num_predictions)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h, pred = state
      else:
        c, h, pred = array_ops.split(value=state, num_or_size_splits=3, axis=1)

      z = tf.nn.l2_normalize(pred, dim=1)
      z = tf.nn.tanh(linear(z, 256, 'encode_pred', normalized_columns_initializer(0.1)))
      # z = tf.constant(value=0., dtype=tf.float32, shape=[1, 256], name='encode_pred')
      concat = _linear([inputs, h, z], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      ## Now, from the new_c compute a new prediction

      y = linear(new_c, 32*(self.grid_size-10)*(self.grid_size-10), 'auxbranch', normalized_columns_initializer(0.1))
      y = tf.reshape(y, shape=[-1, self.grid_size-10, self.grid_size-10, 32])
      deconv_weights = tf.get_variable("deconv" + "/w", [4, 4, self.ac_space, 32])
      new_pred_unshaped = tf.nn.conv2d_transpose(y, deconv_weights,
                                                output_shape=[1, self.grid_size, self.grid_size, self.ac_space],
                                                strides=[1,2,2,1], padding='SAME')
      new_pred = tf.reshape(new_pred_unshaped, shape=[1, self._num_predictions])


      if self._state_is_tuple:
        new_state = PredictionLSTMStateTuple(new_c, new_h, new_pred)
      else:
        new_state = array_ops.concat(1, [new_c, new_h, new_pred])
      return new_h, new_state


_PredictionLSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h", "pred"))

class PredictionLSTMStateTuple(_PredictionLSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores three elements: `(c, h, pred)`, in that order.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h, pred) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s vs %s" %
                      (str(c.dtype), str(h.dtype), str(pred.dtype)))
    return c.dtype



def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        "Linear/Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          "Linear/Bias", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return nn_ops.bias_add(res, biases)


def weight_variable(shape, initial_weight=None):
    if initial_weight is None:
        initial = tf.random_normal(shape, stddev=0.01)
        return tf.get_variable(initial)
    else:
        return tf.get_variable(initial_weight)



####################################################################

class BasicPredictionLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, num_pred, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tanh):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self.num_pred = num_pred
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

  @property
  def state_size(self):
    return (PredictionLSTMStateTuple(self._num_units, self._num_units, self.num_pred)
            if self._state_is_tuple else 2 * self._num_units + self.num_pred)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h, pred = state
      else:
        c, h, pred = array_ops.split(value=state, num_or_size_splits=3, axis=1)

      z = tf.nn.l2_normalize(pred, dim=1)
      z = tf.nn.tanh(linear(z, 256, 'encode_pred', normalized_columns_initializer(0.1)))
      # z = tf.constant(value=0., dtype=tf.float32, shape=[1, 256], name='encode_pred')
      concat = _linear([inputs, h, z], 4 * self._num_units, True)


      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      new_pred = linear(new_c, self.num_pred, name="predictions")

      if self._state_is_tuple:
        new_state = PredictionLSTMStateTuple(new_c, new_h, new_pred)
      else:
        new_state = array_ops.concat([new_c, new_h, new_pred], 1)
      return new_h, new_state



############################## Question Network cells ##############################

class BasicQuestionRNNCell(RNNCell):
    """The most basic RNN cell. Nothing is scaled, no structure on the observational input.
    Truly the most basic possible question network."""

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
          logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"

            output = self._activation(_linear([inputs, state], self._num_units, True))
        return output, output


class DiscountedQuestionRNNCell(RNNCell):
    """ Gives the predictions the structure of discounted sums of observational features.
     Normalize discount factor to (.7, .99)"""
    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
          logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            discount_factors = tf.get_variable(name="pred_discounts", shape=[self._num_units])
            normalized_discount_factors = .3*tf.nn.sigmoid(discount_factors) + .7
            discount_matrix = tf.matrix_diag(normalized_discount_factors)
            discounted_predictions = tf.matmul(state, discount_matrix)
            # obs_features = self._activation(_linear([inputs], self._num_units, True))
            output = discounted_predictions + inputs

        return output, output


# This will produce bootstrapped targets of a given discount factor
class DiscountRNNCell(RNNCell):
    """ Gives the predictions the structure of discounted sums of observational features.
     Normalize discount factor to (.7, .99)"""
    def __init__(self, num_units, input_size=None, discount_factor=.99, activation=tanh):
        if input_size is not None:
          logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self.discount_factor = discount_factor

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            discount_matrix = tf.matrix_diag(tf.constant(self.discount_factor, shape=[self._num_units]))
            discounted_predictions = tf.matmul(state, discount_matrix)
            # obs_features = self._activation(_linear([inputs], self._num_units, True))
            output = discounted_predictions + inputs

        return output, output



class FiniteQuestionRNNCell(RNNCell):
    """Prediction is only a function of convolutional features of a finite number of steps. This makes the
    targets for the answer network predictions perfect (not bootstrapped)"""

    def __init__(self, depth=5, input_size=None, activation=tanh, obs_size=None):
        if input_size is not None:
          logging.warn("%s: The input_size parameter is deprecated.", self)
        self.obs_size = obs_size
        self._num_units = depth*obs_size
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            state = tf.slice(state, begin=[0, self.obs_size], size=[-1, -1])
            state = tf.concat((state, inputs), axis=1)
            output = state
        return output, output


# ## testing finitequestionRNNCell
#
# cell = FiniteQuestionRNNCell(depth=5, obs_size=2)
#
# a = tf.ones(shape=[1, 3, 2], dtype=tf.float32)
#
# initial_state = tf.zeros(shape=[1, 5*2], dtype=tf.float32)
#
#
# rnn_output, rnn_state = tf.nn.dynamic_rnn(
# cell, a, initial_state=initial_state, sequence_length=[3],
# time_major=False)
#
# sess = tf.Session()
# print(sess.run(rnn_output))





