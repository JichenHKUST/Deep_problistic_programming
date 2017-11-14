import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Variational, Normal
from edward.stats import bernoulli, beta

import pandas_datareader as pdr

class AR1(object):
    """p(x, z) = normal(y|x*z, e) * normal(0, e) """

    def __init__(self):
        self.n_vars = 2

        self.lik_variance = 0.1
        self.prior_variance = 0.1

    def log_prob(self, xs, zs):

        log_prior = -tf.reduce_sum(zs*zs, 1) / self.prior_variance

        b = zs[:, 0]
        W = tf.expand_dims(zs[:, 1], 0)

        x_shape = tf.shape(xs['x'])

        x = tf.slice(xs['x'], [0], x_shape - 1)
        x = tf.expand_dims(x, 1)

        y = tf.slice(xs['x'], [1], x_shape - 1)
        y = tf.expand_dims(y, 1)

        mus = tf.matmul(x, W) + b

        log_lik = -tf.reduce_sum(tf.pow(mus - y, 2), 0) / self.lik_variance

        return log_lik + log_prior

ed.set_seed(41)
model = AR1()
variational = Variational()
variational.add(Normal(model.n_vars))

goog = pdr.yahoo.daily.YahooDailyReader(symbols=['goog']).read()
close = goog['Adj Close']

data = {'x': close.values.ravel()}

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=400)