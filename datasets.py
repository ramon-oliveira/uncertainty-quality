import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.datasets import cifar10


class Dataset(object):

    def __init__(self, validation_size=0.1, test_size=0.2):
        self.train_size = 1 - validation_size - test_size
        self.validation_size = validation_size
        self.test_size = test_size

    @property
    def train_data(self):
        return self.X_train, self.y_train

    @property
    def validation_data(self):
        return self.X_val, self.y_val

    @property
    def test_data(self):
        return self.X_test, self.y_test

    @property
    def input_shape(self):
        return self.X_train.shape[1:]

    @property
    def num_classes(self):
        return len(self.y_train.unique())


class Digits(Dataset):

    def __init__(self, *args, **kwargs):
        super(Digits, self).__init__(*args, **kwargs)

        X, y = datasets.load_digits(return_X_y=True)
        idxs = np.arange(X.shape[0])
        np.random.shuffle(idxs)

        begin = 0
        end = int(self.train_size*len(idxs))
        self.X_train, self.y_train = X[idxs][begin:end], y[idxs][begin:end]

        begin = end
        end = end + int(self.validation_size*len(idxs))
        self.X_val, self.y_val = X[idxs][begin:end], y[idxs][begin:end]

        begin = end
        end = len(idxs)
        self.X_test, self.y_test = X[idxs][begin:end], y[idxs][begin:end]


class MNIST(Dataset):

    def __init__(self, test_size, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train/128
        X_test = X_test/128
        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)

        split = 50000
        self.X_train, self.y_train = X_train[idxs][:split], y_train[idxs][:split]
        self.X_val, self.y_val = X_train[idxs][split:], y_train[idxs][split:]
        self.X_test, self.y_test = X_test, y_test


class CIFAR10(Dataset):

    def __init__(self, test_size, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train/128
        X_test = X_test/128
        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)

        split = 40000
        self.X_train, self.y_train = X_train[idxs][:split], y_train[idxs][:split]
        self.X_val, self.y_val = X_train[idxs][split:], y_train[idxs][split:]
        self.X_test, self.y_test = X_test, y_test


def load(dataset):
    name = dataset.pop('name')

    if name == 'digits':
        dataset = Digits(**dataset)
    elif name == 'mnist':
        dataset = MNIST(**dataset)
    elif name == 'cifar10':
        dataset = CIFAR10(**dataset)
    else:
        raise Exception('Unknown dataset {0}'.format(dataset))

    return dataset
