from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf
from data import make_pinwheel

import matplotlib.pyplot as plt
from edward.models import MultivariateNormalTriL, Normal
from edward.util import rbf


data = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=5,
                         num_per_class=30, rate=0.4)
N = data.shape[0]
D = 2 # number of features
K = 2 # number of latent dimensions

# Model: deep/shallow GP (generative model)
X = Normal(loc=tf.zeros([N,K]),scale=tf.ones([N,K]))

Kernal = tf.stack([rbf(tf.reshape(xn, [K, 1])) + tf.diag([1e-6, 1e-6])
              for xn in tf.unstack(X)])

Y = MultivariateNormalTriL(loc=tf.zeros([N,D]), scale_tril=tf.cholesky(Kernal))

# Inference (recongnition model)
qX = Normal(loc=tf.Variable(tf.random_normal([N,K])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,K]))))


inference = ed.KLqp({X: qX}, data={Y: data})
inference.run(n_iter=5000)

# Evaluate
sess = ed.get_session()
qX_mean, qX_var = sess.run([qX.mean(), qX.variance()])
plt.scatter(qX_mean[:,0], qX_mean[:,1])

Y_post = ed.copy(Y, {X: qX})
Y_post = Y_post.eval()
plt.scatter(Y_post[:,0], Y_post[:,1])
plt.scatter(data[:,0], data[:,1]) # observation