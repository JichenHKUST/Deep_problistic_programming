import edward as ed
import tensorflow as tf
from edward.models import Normal
import numpy as np

GLOBAL_SEED = 1234

H = 5  # dimension of hidden state "h"
D = 2  # dimension of latent state "z"
T = 100 # number of time points

## wrapper around tf.get_variable that checks if the variable has already been defined.
## useful for playing around in jupyter. allows you to call cells multiple times
def get_variable_wrap(*args, **kwargs):
    try:
        return tf.get_variable(*args, **kwargs)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        return tf.get_variable(*args, **kwargs)

## do matrix multiplication with weights and add bias (as in fully connected layer)
def fc_act(x, next_layer_size, act=None, name="fc"):
    #nbatches = x.get_shape()[0]
    prev_layer_size = x.get_shape()[1]
    with tf.name_scope("fc"):
        w = get_variable_wrap("weights", [prev_layer_size, next_layer_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
        b = get_variable_wrap("bias", [next_layer_size], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        o = tf.add(tf.matmul(x, w), b)
        if act: return act(o)
        else: return o

## rnn cell for dynamics computation
def rnn_cell(hprev, zprev):
  return tf.tanh(ed.dot(hprev, Wh) + ed.dot(zprev, Wz) + bh)

## encode random variable
def encode_z(ht):
  """
  encode the latent values given hidden state, i.e. z ~ p(z|h)
  """
  phi_size = 2
  z_size = 2

  with tf.variable_scope("phi"):
    phi = fc_act(ht, phi_size, act=tf.nn.relu, name="fc_phi")
  with tf.variable_scope("phi_mu"):
    phi_mu = fc_act(phi, z_size, name="fc_phi_mu")
  with tf.variable_scope("phi_sigma"):
    phi_sigma = fc_act(phi, z_size, act=tf.nn.softplus, name="fc_phi_sigma")
  epsilon = tf.random_normal(shape=[ht.get_shape().as_list()[0], z_size], seed=GLOBAL_SEED)
  # z = mu + epsilon*sigma
  zt = tf.add(phi_mu, tf.multiply(epsilon, phi_sigma))
  return zt

## Generative Model
Wh = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
Wz = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
bh = Normal(loc=tf.zeros(H), scale=tf.ones(H))

Wy = Normal(loc=tf.zeros([H, 1]), scale=tf.ones([H, 1]))
by = Normal(loc=tf.zeros(1), scale=tf.ones(1))

zt = tf.zeros([D,T])
zt[:,0] = Normal(loc=tf.zeros([D ,D]), scale=tf.ones([D ,D]))  # prior on x
ht = tf.zeros([H,T])

for t in range(1, T):
    ht[:,t] = rnn_cell(ht[:,t-1], zt[:,t-1])
    zt[:,t] = encode_z(ht[:,t])
    yt[:,t] = Normal(loc=tf.matmul(zt[:,t], Wy) + by, scale=1.0)

## Inference







