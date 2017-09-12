import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.datasets import cifar10
import keras.backend as K


class Dataset(object):

    def __init__(self, batch_size=32):
        self.batch_size = batch_size


class MNIST(Dataset):

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            self.X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
            self.X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
        else:
            self.X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
            self.X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        self.y_train = y_train
        self.y_test = y_test

        self.input_shape = self.X_train.shape[1:]
        self.num_classes = 10

        self.split = 50000
        self.train_samples = self.split
        self.validation_samples = len(self.X_train) - self.split
        self.test_samples = len(self.X_test)

    def train(self):
        data_generator = ImageDataGenerator(rescale=1/255)
        return data_generator.flow(
            x=self.X_train[:self.split],
            y=np.eye(self.num_classes)[self.y_train[:self.split].ravel()],
            batch_size=self.batch_size,
        )

    def validation(self):
        data_generator = ImageDataGenerator(rescale=1/255)
        return data_generator.flow(
            x=self.X_train[self.split:],
            y=np.eye(self.num_classes)[self.y_train[self.split:].ravel()],
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test(self):
        data_generator = ImageDataGenerator(rescale=1/255)
        return data_generator.flow(
            x=self.X_test,
            y=np.eye(self.num_classes)[self.y_test.ravel()],
            batch_size=self.batch_size,
            shuffle=False,
        )


class CIFAR10(Dataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        self.input_shape = self.X_train.shape[1:]
        self.num_classes = 10

        self.split = 40000
        self.train_samples = self.split
        self.validation_samples = len(self.X_train) - self.split
        self.test_samples = len(self.X_test)

    def train(self):
        data_generator = ImageDataGenerator(
            rescale=1/255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )
        return data_generator.flow(
            x=self.X_train[:self.split],
            y=np.eye(self.num_classes)[self.y_train[:self.split].ravel()],
            batch_size=self.batch_size,
            shuffle=True,
        )

    def validation(self):
        data_generator = ImageDataGenerator(rescale=1/255)
        return data_generator.flow(
            x=self.X_train[self.split:],
            y=np.eye(self.num_classes)[self.y_train[self.split:].ravel()],
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test(self):
        data_generator = ImageDataGenerator(rescale=1/255)
        return data_generator.flow(
            x=self.X_test,
            y=np.eye(self.num_classes)[self.y_test.ravel()],
            batch_size=self.batch_size,
            shuffle=False,
        )


class Melanoma(Dataset):

    def __init__(self, *args, **kwargs):
        super(Melanoma, self).__init__(*args, **kwargs)
        self.num_classes = 2

        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, 224, 224)
        else:
            self.input_shape = (224, 224, 3)

        self.train_samples = 1800
        self.validation_samples = 200
        self.test_samples = 600

    def train(self):
        data_generator = ImageDataGenerator(
            rescale=1/255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )
        return data_generator.flow_from_directory(
            directory='data/melanoma/train',
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=['other', 'melanoma'],
        )

    def validation(self):
        data_generator = ImageDataGenerator(rescale=1/255)
        return data_generator.flow_from_directory(
            directory='data/melanoma/validation',
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            classes=['other', 'melanoma'],
        )

    def test(self):
        data_generator = ImageDataGenerator(rescale=1/255)
        return data_generator.flow_from_directory(
            directory='data/melanoma/test',
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            classes=['other', 'melanoma'],
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
