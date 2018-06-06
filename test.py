import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np

try:
	_tutorial = tf.load_op_library('./build/libtutorial.so')
except Exception as e:
	_tutorial = tf.load_op_library('./libtutorial.so')
custom_add = _tutorial.custom_add

shape = (1,1000,1000,30)

a_data = np.random.random(shape)
b_data = np.random.random(shape)


a = tf.placeholder(tf.float32, shape=shape, name="a")
b = tf.placeholder(tf.float32, shape=shape, name="b")

c_cust = custom_add(a,b)
print(c_cust)
c = a + b

config = tf.ConfigProto(log_device_placement = True)
config.graph_options.optimizer_options.opt_level = -1

with tf.Session(config=config) as sess:
	feed = {a: a_data, b: b_data}
	expected = sess.run(c, feed_dict=feed)
	result = sess.run(c_cust, feed_dict=feed)
	print(np.array_equal(expected,result))