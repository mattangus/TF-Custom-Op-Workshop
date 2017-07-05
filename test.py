import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np

_tutorial = tf.load_op_library('./build/libtutorial.so')
custom_add = _tutorial.custom_add

shape = (1,10,10,3)

a = tf.constant(np.random.random(shape), shape=shape, name="a")
b = tf.constant(np.random.random(shape), shape=shape, name="b")


c = custom_add(a,b)

with tf.Session() as sess:
	print(sess.run(c))