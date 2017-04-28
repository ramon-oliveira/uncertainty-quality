from abc import ABC, abstractmethod

import tensorflow as tf
import edward as ed
from edward.models import Normal, Categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


class Model(ABC):

    @abstractmethod
    def fit(X, y):
        pass

    @abstractmethod
    def predict_proba(X):
        pass

    @abstractmethod
    def predict(X):
        pass


class LogisticRegression(Model):

    def __init__(self, n_samples=None, n_inter=None, inference='mcmc'):
        self.inference = inference
        self.n_samples = n_samples
        self.n_iter = n_iter

    def _inference(self, X_data, y_data):
        N, D = X_data.shape
        X = tf.placeholder(tf.float32, [N, D])
        W = Normal(mu=tf.zeros([D]), sigma=tf.ones([D]))
        b = Normal(mu=tf.zeros([1]), sigma=tf.ones([1]))
        y = Categorical(logits=ed.dot(X, W) + b)

        qW = Normal(mu=tf.Variable(tf.random_normal([D])),
                    sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
        qb = Normal(mu=tf.Variable(tf.random_normal([1])),
                    sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

        inference = ed.KLqp({W: qW, b: qb}, data={X: X_data, y: y_data})
        inference.run(n_samples=self.n_samples, n_iter=self.n_iter)

    def fit(self, X, y):
        self._inference(X, y)

    def predict_proba(X):
        y_post = Normal(mu=ed.dot(X, qw) + qb, sigma=tf.ones([N]))
        tf.eval()
        ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test})
