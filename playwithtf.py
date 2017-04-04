# import tensorflow as tf
# from helpers.layer_helpers import weight_variable
#
# sess = tf.Session()
# with tf.variable_scope("question"):
#     a = tf.placeholder(dtype=tf.float32)
#     w1 = tf.Variable(initial_value=[10.], name='w1')
#     # w1 = weight_variable(shape=[1], name='w1', initial_weight=10.)
#     b = tf.mul(a,w1)
#
#
#     c = tf.placeholder(dtype=tf.float32)
#     w2 = tf.get_variable(name='w2', shape=[1], initializer=tf.constant_initializer(5.))
#
#
#     d = tf.mul(c,w2)
#
# with tf.variable_scope("question", reuse=True):
#     w3 = tf.get_variable(name='w2', shape=[1])
#     e = tf.mul(b,w3)
#
# gradients = tf.gradients(e, w2)
# # tf.train.GradientDescentOptimizer(learning_rate=.1)
#
# sess.run(tf.initialize_all_variables())
# print(sess.run(d, feed_dict={c: 6, w3: [100]}))
#
#
#
# #
# # print([n.name for n in tf.get_default_graph().as_graph_def().node])
# # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# # print([v.name for v in vars])
#
#
#
#
#
# #
# # with tf.variable_scope("foo"):
# #     v = tf.get_variable("v", [1], initializer=tf.constant_initializer(value=5.))
# # with tf.variable_scope("foo", reuse=True):
# #     v1 = tf.get_variable("v", [1])
# #
# # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# # print([v.name for v in vars])


import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.sigmoid(a)
sess = tf.Session()

print(sess.run(b, feed_dict={a: -2}))

c = tf.nn.rnn_cell.BasicLSTMCell(256, activation="sigmoid")
