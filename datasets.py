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
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from scipy.misc import imresize
import re
import joblib as jl


class Dataset(object):

    def __init__(self, type, train_size=1.0):
        self.type = type
        self.sota = {}
        self.train_size = train_size


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

        self.sota = {
            'accuracy': 99.79,
            'reference': 'http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html',
        }
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
        super(FashonMNIST, self).__init__('classification', *args, **kwargs)
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(-1, 1, 28, 28)/255
            x_test = x_test.reshape(-1, 1, 28, 28)/255
        else:
            x_train = x_train.reshape(-1, 28, 28, 1)/255
            x_test = x_test.reshape(-1, 28, 28, 1)/255

        y_train = np.eye(10)[y_train.ravel()]
        y_test = np.eye(10)[y_test.ravel()]

        self.sota = {
            'accuracy': 0.967,
            'reference': 'http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html',
        }
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

        idxs = np.random.choice(split, replace=False, size=int(split*self.train_size))
        self.x_train = self.x_train[idxs]
        self.y_train = self.y_train[idxs]


class CIFAR10(Dataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__('classification', *args, **kwargs)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train/255
        x_test = x_test/255

        y_train = np.eye(10)[y_train.ravel()]
        y_test = np.eye(10)[y_test.ravel()]

        self.sota = {
            'accuracy': 96.53,
            'reference': 'http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html',
        }
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

        idxs = np.random.choice(split, replace=False, size=int(split*self.train_size))
        self.x_train = self.x_train[idxs]
        self.y_train = self.y_train[idxs]


class CIFAR100(Dataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR100, self).__init__('classification', *args, **kwargs)
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        x_train = x_train/255
        x_test = x_test/255

        y_train = np.eye(100)[y_train.ravel()]
        y_test = np.eye(100)[y_test.ravel()]

        self.sota = {
            'accuracy': 75.72,
            'reference': 'http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html',
        }
        self.input_shape = x_train.shape[1:]
        self.output_size = 100
        self.classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed',
                        'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge',
                        'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                        'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
                        'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                        'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
                        'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree',
                        'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
                        'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
                        'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                        'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
                        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                        'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
                        'sweet_pepper', 'table', 'tank', 'telephone', 'television',
                        'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                        'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        print('classes:', len(self.classes))

        split = 42000
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test

        idxs = np.random.choice(split, replace=False, size=int(split*self.train_size))
        self.x_train = self.x_train[idxs]
        self.y_train = self.y_train[idxs]


class MotorolaTriage(Dataset):

    def __init__(self, *args, **kwargs):
        super(MotorolaTriage, self).__init__('classification', *args, **kwargs)
        data = jl.load(open('data/motorola_triage/motorola_triage.pkl', 'rb'))
        len(data)
        data = data[self.train_size]
        x_train = data['train']['x'].toarray()
        y_train = data['train']['y_true']
        classes = sorted(list(set(y_train.tolist())))

        x_test = data['test']['x'].toarray()
        y_test = data['test']['y_true']
        mask = np.in1d(y_test, classes)
        x_test = x_test[mask]
        y_test = y_test[mask]

        y_train = np.array([np.eye(len(classes))[classes.index(c)] for c in y_train])
        y_test = np.array([np.eye(len(classes))[classes.index(c)] for c in y_test])

        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)

        self.sota = {
            'accuracy': 54,
            'reference': '',
        }
        self.input_shape = x_train.shape[1:]
        self.output_size = len(classes)
        self.classes = classes
        print('classes:', len(self.classes))

        split = int(x_train.shape[0]*0.8)
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test

        # idxs = np.random.choice(split, replace=False, size=int(split*self.train_size))
        # self.x_train = self.x_train[idxs]
        # self.y_train = self.y_train[idxs]


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
        self.classes = ['not-melanoma', 'melanoma']

        # datagen = ImageDataGenerator(
        #     rotation_range=30,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     shear_range=0.1,
        #     zoom_range=0.1,
        #     fill_mode='nearest',
        #     horizontal_flip=True,
        #     vertical_flip=True,
        # )
        #
        # n = 48000
        # xs = []
        # ys = []
        # pbar = tqdm.tqdm(total=n)
        # for x, y in datagen.flow(x_train, y_train, batch_size=1000):
        #     xs.append(x)
        #     ys.append(y)
        #     pbar.update(1000)
        #     if len(xs)*1000 == n:
        #         pbar.close()
        #         break
        # xs.append(x_train)
        # ys.append(y_train)
        # x_train = np.vstack(xs)
        # y_train = np.vstack(ys)

        split = int(len(x_train)*0.8)
        self.x_train = x_train[:split]
        self.y_train = y_train[:split]
        self.x_val = x_train[split:]
        self.y_val = y_train[split:]
        self.x_test = x_test
        self.y_test = y_test

        print('train shape:', self.y_train.shape)
        print('val shape:', self.y_val.shape)
        print('test shape:', self.y_test.shape)
        print('train prop:', self.y_train.argmax(axis=1).mean())
        print('val prop:', self.y_val.argmax(axis=1).mean())
        print('test prop:', self.y_test.argmax(axis=1).mean())


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
    elif name == 'motorola_triage':
        dataset = MotorolaTriage(**settings)
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
