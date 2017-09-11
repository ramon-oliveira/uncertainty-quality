import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.datasets import cifar10
import keras.backend as K


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

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train/128
        X_test = X_test/128

        if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
            X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
            self.input_shape = (1, 28, 28)
        else:
            X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
            X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
            self.input_shape = (28, 28, 1)

        y_train = y_train.ravel()
        y_test = y_test.ravel()
        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)

        split = 50000
        self.X_train, self.y_train = X_train[idxs][:split], y_train[idxs][:split]
        self.X_val, self.y_val = X_train[idxs][split:], y_train[idxs][split:]
        self.X_test, self.y_test = X_test, y_test


class CIFAR10(Dataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train/128
        X_test = X_test/128

        y_train = y_train.ravel()
        y_test = y_test.ravel()
        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)

        split = 40000
        self.X_train, self.y_train = X_train[idxs][:split], y_train[idxs][:split]
        self.X_val, self.y_val = X_train[idxs][split:], y_train[idxs][split:]
        self.X_test, self.y_test = X_test, y_test



class Melanoma(Dataset):

    def __init__(self, batch_size, *args, **kwargs):
        super(Melanoma, self).__init__(*args, **kwargs)
        self.num_classes = 2

        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, 224, 224)
        else:
            self.input_shape = (224, 224, 3)

        self.data_generator = ImageDataGenerator(
            rescale=1/255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )

        self.train_generator = self.data_generator.flow_from_directory(
            directory='data/melanoma/train',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
        )

        self.validation_generator = self.data_generator.flow_from_directory(
            directory='data/melanoma/validation',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
        )

        self.validation_generator = self.data_generator.flow_from_directory(
            directory='data/melanoma/test',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
        )


def load(settings):
    name = settings.pop('name')

    if name == 'digits':
        dataset = Digits(**settings)
    elif name == 'mnist':
        dataset = MNIST(**settings)
    elif name == 'cifar10':
        dataset = CIFAR10(**settings)
    elif name == 'melanoma':
        dataset = Melanoma(**settings)
    else:
        raise Exception('Unknown dataset {0}'.format(name))

    return dataset
