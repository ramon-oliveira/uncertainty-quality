import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sacred import Experiment
from models import MLP

ex = Experiment('uncertainty-quality-classification')


@ex.config
def cfg():
    seed = 1337
    dataset = 'digits'
    test_size = 0.3
    model = 'linear'
    inference = 'variational'

    n_iter = 10
    n_samples = 5


@ex.capture
def load_dataset(dataset, test_size, _rnd):
    if dataset == 'digits':
        X, y = datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    else:
        raise Exception('Unknown dataset {0}'.format(dataset))

    classes = np.arange(10)
    _rnd.shuffle(classes)
    in_train = classes[:4]
    out_train = classes[4:8]
    unknown = classes[8:]

    X_train = X_train[np.in1d(y_train, in_train)]
    y_train = y_train[np.in1d(y_train, in_train)]

    X_test = X_test[~np.in1d(y_test, unknown)]
    y_test = y_test[~np.in1d(y_test, unknown)]

    return X_train, y_train, X_test, y_test, in_train, out_train, unknown


@ex.capture
def train_variational(X_data, y_data, n_iter, n_samples):
    X = tf.placeholder(tf.float32, [N, D])
    w = Normal(mu=tf.zeros(D), sigma=tf.ones(D))
    b = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

    qw = Normal(mu=tf.Variable(tf.random_normal([D])),
                sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
    qb = Normal(mu=tf.Variable(tf.random_normal([1])),
                sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

    y = Normal(mu=ed.dot(X, w) + b, sigma=tf.ones(N))

    inference = ed.KLqp({w: qw, b: qb}, data={X: X_data, y: y_data})
    inference.run(n_samples=n_samples, n_iter=n_iter)


@ex.capture
def train_dropout(self,)

@ex.capture
def train(X_data, y_data, model, inference):





@ex.automain
def main():
    *data, in_train, out_train, unknown = load_dataset()
    X_train, y_train, X_test, y_test = data
