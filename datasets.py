import tqdm
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import boston_housing
# from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import re


class Dataset(object):

    def __init__(self, type):
        self.type = type
        self.sota = {}


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
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        split = 50000
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test


class FashonMNIST(Dataset):

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__('classification', *args, **kwargs)
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

        self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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
        x_train = x_train/255
        x_test = x_test/255

        y_train = np.eye(10)[y_train.ravel()]
        y_test = np.eye(10)[y_test.ravel()]

        self.input_shape = x_train.shape[1:]
        self.output_size = 10
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        split = 42000
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test


class CIFAR100(Dataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR100, self).__init__('classification', *args, **kwargs)
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train/255
        x_test = x_test/255

        y_train = np.eye(100)[y_train.ravel()]
        y_test = np.eye(100)[y_test.ravel()]

        self.input_shape = x_train.shape[1:]
        self.output_size = 100
        self.classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish',
                        'flatfish', 'ray', 'shark', 'trout', 'orchids', 'poppies',
                        'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans',
                        'cups', 'plates', 'apples', 'mushrooms', 'oranges', 'pears',
                        'sweet peppers', 'clock', 'computer keyboard', 'lamp',
                        'telephone', 'television', 'bed', 'chair', 'couch', 'table',
                        'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar',
                        'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf',
                        'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud',
                        'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle',
                        'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine',
                        'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail',
                        'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile',
                        'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse',
                        'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine',
                        'willow', 'bicycle', 'bus', 'motorcycle', 'pickup truck',
                        'train', 'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
        print('classes:', len(self.classes))

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
        self.sota = {
            'rmse_mean': 2.97,
            'rmse_std': 0.84,
            'll_mean': -2.41,
            'll_std': 0.25,
            'reference': 'https://arxiv.org/pdf/1612.01474.pdf',
        }
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.1)
        y_train = np.hstack([y_train.reshape(-1, 1), -np.ones(shape=[len(y_train), 1])])
        y_test = np.hstack([y_test.reshape(-1, 1), -np.ones(shape=[len(y_test), 1])])

        xscaler = StandardScaler().fit(x_train)
        x_train = xscaler.transform(x_train)
        x_test = xscaler.transform(x_test)
        self.xscaler = xscaler

        yscaler = StandardScaler().fit(y_train)
        y_train = yscaler.transform(y_train)
        y_test = yscaler.transform(y_test)
        self.yscaler = yscaler

        self.input_shape = x_train.shape[1:]
        self.output_size = 2

        split = int(len(x_train)*0.8)
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test

        print(self.x_train.shape, self.y_train.shape)
        print(self.x_val.shape, self.y_val.shape)
        print(self.x_test.shape, self.y_test.shape)


class Kin8nm(Dataset):

    def __init__(self, *args, **kwargs):
        super(Kin8nm, self).__init__('regression', *args, **kwargs)
        self.sota = {
            'rmse_mean': 0.09,
            'rmse_std': 0.00,
            'll_mean': 1.20,
            'll_std': 0.02,
            'reference': 'https://arxiv.org/pdf/1612.01474.pdf',
        }
        df = pd.read_csv('data/kin8nm/kin8nm.csv', header=None)
        df[9] = 0.0
        x = df.values[:, :8]
        y = df.values[:, 8:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        xscaler = StandardScaler().fit(x_train)
        x_train = xscaler.transform(x_train)
        x_test = xscaler.transform(x_test)
        self.xscaler = xscaler

        yscaler = StandardScaler().fit(y_train)
        y_train = yscaler.transform(y_train)
        y_test = yscaler.transform(y_test)
        self.yscaler = yscaler

        self.input_shape = x_train.shape[1:]
        self.output_size = 2

        split = int(len(x_train)*0.8)
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test


class ProteinStructure(Dataset):

    def __init__(self, *args, **kwargs):
        super(ProteinStructure, self).__init__('regression', *args, **kwargs)
        self.sota = {
            'rmse_mean': 4.36,
            'rmse_std': 0.04,
            'll_mean': -2.83,
            'll_std': 0.02,
            'reference': 'https://arxiv.org/pdf/1612.01474.pdf',
        }
        df = pd.read_csv('data/protein_structure/protein_structure.csv')
        x = df.values[:, 1:].astype('float32')
        y = df.values[:, 0:1].astype('float32')
        y = np.hstack([y, -np.ones(shape=[len(y), 1])])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        xscaler = StandardScaler().fit(x_train)
        x_train = xscaler.transform(x_train)
        x_test = xscaler.transform(x_test)
        self.xscaler = xscaler

        yscaler = StandardScaler().fit(y_train)
        y_train = yscaler.transform(y_train)
        y_test = yscaler.transform(y_test)
        self.yscaler = yscaler

        self.input_shape = x_train.shape[1:]
        self.output_size = 2

        split = int(len(x_train)*0.8)
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
    elif name == 'fashion_mnist':
        dataset = FashonMNIST(**settings)
    elif name == 'cifar10':
        dataset = CIFAR10(**settings)
    elif name == 'cifar100':
        dataset = CIFAR100(**settings)
    elif name == 'melanoma':
        dataset = Melanoma(**settings)
    elif name == 'boston_housing':
        dataset = BostonHousing(**settings)
    elif name == 'kin8nm':
        dataset = Kin8nm(**settings)
    elif name == 'protein_structure':
        dataset = ProteinStructure(**settings)
    else:
        raise Exception('Unknown dataset {0}'.format(name))

    settings['name'] = name

    return dataset
