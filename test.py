import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np

try:
	_tutorial = tf.load_op_library('./build/libtutorial.so')
except Exception as e:
	_tutorial = tf.load_op_library('./libtutorial.so')

custom_add = _tutorial.custom_add

shape = (1,10,10)

a_data = np.random.random(shape)
b_data = np.random.random(shape)

a = tf.constant(a_data, shape=shape, name="a")
b = tf.constant(b_data, shape=shape, name="b")

c_cust = custom_add(a,b)
