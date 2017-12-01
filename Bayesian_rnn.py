import edward as ed
import tensorflow as tf
from edward.models import Normal

H = 50  # number of hidden units
D = 10  # number of features

def rnn_cell(hprev, xt):
  return tf.tanh(ed.dot(hprev, Wh) + ed.dot(xt, Wx) + bh)

Wh = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
Wx = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
Wy = Normal(loc=tf.zeros([H, 1]), scale=tf.ones([H, 1]))
bh = Normal(loc=tf.zeros(H), scale=tf.ones(H))
by = Normal(loc=tf.zeros(1), scale=tf.ones(1))

x = tf.placeholder(tf.float32, [None, D])
h = tf.scan(rnn_cell, x, initializer=tf.zeros(H))
y = Normal(loc=tf.matmul(h, Wy) + by, scale=1.0)
