from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

from edward.models import Bernoulli, MultivariateNormalTriL, Normal
from edward.util import rbf

# Data
def build_toy_dataset(N):
  # step function
  x = 4 * np.random.rand(N)-2
  noise = np.random.normal(0, 0.01, N)
  y = np.sign(x) + noise

  return y

N = 1000
Q = 1 # input space feature
H1 = 1
D = 1

# Model: deep/shallow GP
X = Normal(loc=tf.zeros([N,Q]),scale=tf.ones([N,Q]))
f = MultivariateNormalTriL(loc=tf.zeros(N,D), scale_tril=tf.cholesky(rbf(X)))
#f = MultivariateNormalTriL(loc=tf.zeros(N,), scale_tril=tf.cholesky(rbf(h1)))
y = Bernoulli(logits=f)

# inference
qX = Normal(loc=tf.Variable(tf.random_normal([N,Q])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,Q]))))
qf = Normal(loc=tf.Variable(tf.random_normal([N,D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,D]))))

inference = ed.KLqp({f: qf, X:qX}, data={y: y})
inference.run(n_iter=5000)
