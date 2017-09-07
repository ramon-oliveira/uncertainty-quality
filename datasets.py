import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.datasets import cifar10


class Dataset(object):
    pass


class Digits(Dataset):

    def __init__(self, test_size, *args, **kwargs):
        super(Digits, self).__init__(*args, **kwargs)

        X, y = datasets.load_digits(return_X_y=True)
        data = train_test_split(X, y, test_size=test_size)
        self.X_train = data[0]
        self.y_train = data[2]
        self.X_test = data[1]
        self.y_test = data[3]

        self.classes = np.array(list(set(y)))
        np.random.shuffle(self.classes)
        self.in_train_classes = self.classes[:10]
        self.out_train_classes = self.classes[10:]
        self.unk_train_classes = self.classes[10:]

    def train(self, with_unk=False):
        X = self.X_train[np.in1d(self.y_train, self.in_train_classes)]
        y = self.y_train[np.in1d(self.y_train, self.in_train_classes)]

        if with_unk:
            X2 = self.X_train[np.in1d(self.y_train, self.unk_train_classes)]
            y2 = self.y_train[np.in1d(self.y_train, self.unk_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        return X, y

    def test(self, with_out=False, with_unk=False):
        X = self.X_test[np.in1d(self.y_test, self.in_train_classes)]
        y = self.y_test[np.in1d(self.y_test, self.in_train_classes)]

        if with_out:
            X2 = self.X_test[np.in1d(self.y_test, self.out_train_classes)]
            y2 = self.y_test[np.in1d(self.y_test, self.out_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        if with_unk:
            X2 = self.X_test[np.in1d(self.y_test, self.unk_train_classes)]
            y2 = self.y_test[np.in1d(self.y_test, self.unk_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        return X, y



class MNIST(Dataset):

    def __init__(self, test_size, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1)/128
        X_test = X_test.reshape(X_test.shape[0], -1)/128

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.classes = np.array(list(set(y_train)))
        np.random.shuffle(self.classes)
        self.in_train_classes = self.classes[:10]
        self.out_train_classes = self.classes[10:]
        self.unk_train_classes = self.classes[10:]

    def train(self, with_unk=False):
        X = self.X_train[np.in1d(self.y_train, self.in_train_classes)]
        y = self.y_train[np.in1d(self.y_train, self.in_train_classes)]

        if with_unk:
            X2 = self.X_train[np.in1d(self.y_train, self.unk_train_classes)]
            y2 = self.y_train[np.in1d(self.y_train, self.unk_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        return X, y

    def test(self, with_out=False, with_unk=False):
        X = self.X_test[np.in1d(self.y_test, self.in_train_classes)]
        y = self.y_test[np.in1d(self.y_test, self.in_train_classes)]

        if with_out:
            X2 = self.X_test[np.in1d(self.y_test, self.out_train_classes)]
            y2 = self.y_test[np.in1d(self.y_test, self.out_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        if with_unk:
            X2 = self.X_test[np.in1d(self.y_test, self.unk_train_classes)]
            y2 = self.y_test[np.in1d(self.y_test, self.unk_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        return X, y


class CIFAR10(Dataset):

    def __init__(self, test_size, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1)/128
        X_test = X_test.reshape(X_test.shape[0], -1)/128
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.classes = np.array(list(set(y_train)))
        np.random.shuffle(self.classes)
        self.in_train_classes = self.classes[:10]
        self.out_train_classes = self.classes[10:]
        self.unk_train_classes = self.classes[10:]

    def train(self, with_unk=False):
        X = self.X_train[np.in1d(self.y_train, self.in_train_classes)]
        y = self.y_train[np.in1d(self.y_train, self.in_train_classes)]

        if with_unk:
            X2 = self.X_train[np.in1d(self.y_train, self.unk_train_classes)]
            y2 = self.y_train[np.in1d(self.y_train, self.unk_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        return X, y

    def test(self, with_out=False, with_unk=False):
        X = self.X_test[np.in1d(self.y_test, self.in_train_classes)]
        y = self.y_test[np.in1d(self.y_test, self.in_train_classes)]

        if with_out:
            X2 = self.X_test[np.in1d(self.y_test, self.out_train_classes)]
            y2 = self.y_test[np.in1d(self.y_test, self.out_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        if with_unk:
            X2 = self.X_test[np.in1d(self.y_test, self.unk_train_classes)]
            y2 = self.y_test[np.in1d(self.y_test, self.unk_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        return X, y


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
