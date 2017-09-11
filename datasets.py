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
            X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
            X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
        else:
            X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
            X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        self.input_shape = X_train.shape[1:]
        self.num_classes = 10

        split = 50000
        data_generator = ImageDataGenerator(
            rescale=1/255,
            # rotation_range=40,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # shear_range=0.2,
            # zoom_range=0.2,
            # horizontal_flip=True,
            # fill_mode='nearest',
        )

        self.train_samples = split
        self.train_generator = data_generator.flow(
            x=X_train[:split],
            y=np.eye(self.num_classes)[y_train[:split].ravel()],
            batch_size=self.batch_size,
        )

        self.validation_samples = len(X_train) - split
        self.validation_generator = data_generator.flow(
            x=X_train[split:],
            y=np.eye(self.num_classes)[y_train[split:].ravel()],
            batch_size=self.batch_size,
        )

        data_generator = ImageDataGenerator(rescale=1/255)
        self.test_samples = len(X_test)
        self.test_generator = data_generator.flow(
            x=X_test,
            y=np.eye(self.num_classes)[y_test.ravel()],
            batch_size=self.batch_size,
        )


class CIFAR10(Dataset):

    def __init__(self, *args, **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        self.input_shape = X_train.shape[1:]
        self.num_classes = 10
        split = 40000
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

        self.train_samples = split
        self.train_generator = data_generator.flow(
            x=X_train[:split],
            y=np.eye(self.num_classes)[y_train[:split].ravel()],
            batch_size=self.batch_size,
        )

        self.validation_samples = len(X_train) - split
        self.validation_generator = data_generator.flow(
            x=X_train[split:],
            y=np.eye(self.num_classes)[y_train[split:].ravel()],
            batch_size=self.batch_size,
        )

        data_generator = ImageDataGenerator(rescale=1/255)
        self.test_samples = len(X_test)
        self.test_generator = data_generator.flow(
            x=X_test,
            y=np.eye(self.num_classes)[y_test.ravel()],
            batch_size=self.batch_size,
        )


class Melanoma(Dataset):

    def __init__(self, *args, **kwargs):
        super(Melanoma, self).__init__(*args, **kwargs)
        self.num_classes = 2

        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, 224, 224)
        else:
            self.input_shape = (224, 224, 3)

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

        self.train_samples = 1811
        self.train_generator = data_generator.flow_from_directory(
            directory='data/melanoma/train',
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
        )

        self.validation_samples = 189
        self.validation_generator = data_generator.flow_from_directory(
            directory='data/melanoma/validation',
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
        )

        self.test_samples = 600
        self.test_generator = data_generator.flow_from_directory(
            directory='data/melanoma/test',
            target_size=(224, 224),
            batch_size=self.batch_size,
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
