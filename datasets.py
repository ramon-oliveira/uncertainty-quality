import tqdm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import boston_housing
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import re


class Dataset(object):

    def __init__(self, type):
        self.type = type


class MNIST(Dataset):

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__('classification', *args, **kwargs)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(-1, 1, 28, 28)/255
            x_test = x_test.reshape(-1, 1, 28, 28)/255
        else:
            x_train = x_train.reshape(-1, 28, 28, 1)/255
            x_test = x_test.reshape(-1, 28, 28, 1)/255

        y_train = np.eye(10)[y_train.ravel()]
        y_test = np.eye(10)[y_test.ravel()]

        self.input_shape = x_train.shape[1:]
        self.output_size = 10

        split = 50000
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test


class CIFAR10(Dataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__('classification', *args, **kwargs)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train = np.eye(10)[y_train.ravel()]
        y_test = np.eye(10)[y_test.ravel()]

        self.input_shape = x_train.shape[1:]
        self.output_size = 10

        split = 42000
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test


class Melanoma(Dataset):

    def __init__(self, *args, **kwargs):
        super(Melanoma, self).__init__('classification', *args, **kwargs)

        x_train = np.load('data/melanoma/x_train.npy').astype('float32')
        y_train = np.load('data/melanoma/y_train.npy').astype('int32')
        x_test = np.load('data/melanoma/x_test.npy').astype('float32')
        y_test = np.load('data/melanoma/y_test.npy').astype('int32')

        y_train = np.eye(2)[y_train.ravel()]
        y_test = np.eye(2)[y_test.ravel()]

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(-1, 3, 224, 224)/255
            x_test = x_test.reshape(-1, 3, 224, 224)/255
        else:
            x_train = x_train.reshape(-1, 224, 224, 3)/255
            x_test = x_test.reshape(-1, 224, 224, 3)/255

        self.input_shape = x_train.shape[1:]
        self.output_size = 2

        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
        )

        n = 48000
        xs = []
        ys = []
        pbar = tqdm.tqdm(total=n)
        for x, y in datagen.flow(x_train, y_train, batch_size=1000):
            xs.append(x)
            ys.append(y)
            pbar.update(1000)
            if len(xs)*1000 == n:
                pbar.close()
                break
        xs.append(x_train)
        ys.append(y_train)

        x_train = np.vstack(xs)
        y_train = np.vstack(ys)

        split = 42000
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test


class BostonHousing(Dataset):

    def __init__(self, *args, **kwargs):
        super(BostonHousing, self).__init__('regression', *args, **kwargs)
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

        xscaler = StandardScaler().fit(x_train)
        x_train = xscaler.transform(x_train)
        x_test = xscaler.transform(x_test)

        yscaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train = yscaler.transform(y_train).ravel()
        y_test = yscaler.transform(y_test).ravel()

        self.input_shape = x_train.shape[1:]
        self.output_size = 1
        self.split = int(len(x_train)*0.9)

        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test


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
    elif name == 'boston_housing':
        dataset = BostonHousing(**settings)
    else:
        raise Exception('Unknown dataset {0}'.format(name))

    return dataset
