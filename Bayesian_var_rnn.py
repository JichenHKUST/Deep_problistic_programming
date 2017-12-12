import edward as ed
import tensorflow as tf
from edward.models import Normal
import numpy as np
#import matplotlib.pyplot as plt

GLOBAL_SEED = 1234

H = 5  # dimension of hidden state "h"
D = 1  # dimension of latent state "z"
K = 1  # dimension of output
T = 100 # number of time points

## do matrix multiplication with weights and add bias (as in fully connected layer)
def fc_act(x, next_layer_size, act=None, name="fc"):
    #nbatches = x.get_shape()[0]
    prev_layer_size = x.get_shape()[1]

    with tf.name_scope("fc"):
        w = tf.get_variable("weights", [prev_layer_size, next_layer_size], dtype=tf.float32, initializer=tf.random_normal_initializer())
        b = tf.get_variable("bias", [next_layer_size], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
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
  z_size = D
  ht = tf.expand_dims(ht,0)
  with tf.variable_scope("phi"):
    phi = fc_act(ht, phi_size, act=tf.nn.relu, name="fc_phi")
  with tf.variable_scope("phi_mu"):
    phi_mu = fc_act(phi, z_size, name="fc_phi_mu")
  with tf.variable_scope("phi_sigma"):
    phi_sigma = fc_act(phi, z_size, act=tf.nn.softplus, name="fc_phi_sigma")
  # z = mu + epsilon*sigma
  epsilon = 0.01
  zt = tf.add(phi_mu, tf.multiply(epsilon, phi_sigma))
  return zt

## Build toydatase
def build_toy_dataset(T, noise_std=0.5):
    source = 2*np.sin(np.linspace(0, 6, T))
    y = source + np.random.normal(0, noise_std, T)
    return y

## Data

y_data = build_toy_dataset(T)

## Generative Model
Wh = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
Wz = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
bh = Normal(loc=tf.zeros(H), scale=tf.ones(H))

Wy = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
by = Normal(loc=tf.zeros(K), scale=tf.ones(K))

# Initialize data 
zt = Normal(loc=tf.zeros(D), scale=tf.ones(D))  # prior on x
ht = tf.zeros(H)

h = []
z = []
y = []

for t in range(1, T):
    ht = rnn_cell(ht, zt)
    h.append(ht)
    zt0 = encode_z(ht)
    zt = tf.squeeze(zt0, 0)
    z.append(zt)
    yt0 = Normal(loc=tf.matmul(zt0, Wy) + by, scale=1.0)
    yt = tf.squeeze(yt0, 0)
    y.append(yt)

h = tf.stack(h,1)
z = tf.stack(z,1)
y = tf.stack(y,1)

## Inference
qz = [Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D])))) for _ in range(T)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(T))

    inference = ed.KLqp(dict(zip(z, qz)), dict(zip(y, y_data)))
    inference.run(n_iter=2000)

    print(sess.run(T))
    print(sess.run([foo.p for foo in qz]))
    print(sess.run([foo.p for foo in y]))










