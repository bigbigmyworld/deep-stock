import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
foo = tf.constant("hello")
sess = tf.compat.v1.Session(foo)
print(sess.run(foo))
