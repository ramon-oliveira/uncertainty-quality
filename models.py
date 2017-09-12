import os
import numpy as np
import tensorflow as tf
import edward as ed
import tqdm
import uuid
from edward.models import Normal, Categorical, Multinomial
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from layers import BayesianDropout
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class BaseModel(object):

    def fit(self, dataset):
        # es = EarlyStopping(monitor='val_loss', patience=10)
        weights_filename = 'runs/' + uuid.uuid4().hex + '.hdf5'
        cp = ModelCheckpoint(weights_filename, save_best_only=True)

        self.probabilistic_model.fit_generator(
            generator=dataset.train(),
            validation_data=dataset.validation(),
            steps_per_epoch=dataset.train_samples//self.batch_size,
            validation_steps=dataset.validation_samples//self.batch_size,
            epochs=self.epochs,
            workers=4,
            callbacks=[cp],
            # callbacks=[cp, es],
        )

        self.probabilistic_model.load_weights(weights_filename)
        self.model.set_weights(self.probabilistic_model.get_weights())

        return self

    def predict(self, gen, samples, probabilistic=False):
        assert (samples % self.batch_size) == 0

        if probabilistic:
            model = self.probabilistic_model
        else:
            model = self.model

        return model.predict_generator(
            generator=gen,
            steps=samples//self.batch_size,
        )


class MLP(object):

    def __init__(self, dataset, p_dropout=0.5, epochs=100, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size

        # deterministic model
        model = Sequential()
        model.add(Dense(1024, input_shape=dataset.input_shape))
        model.add(Dropout(p_dropout))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Dropout(p_dropout))
        model.add(Activation('relu'))
        model.add(Dense(dataset.output_size))

        # probabilistic model
        probabilistic_model = Sequential()
        probabilistic_model.add(Dense(1024, input_shape=dataset.input_shape))
        probabilistic_model.add(BayesianDropout(p_dropout))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(Dense(1024))
        probabilistic_model.add(BayesianDropout(p_dropout))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(Dense(dataset.output_size))

        if dataset.type == 'classification':
            compile_params = {
                'loss': 'categorical_crossentropy',
                'optimizer': 'adam',
                'metrics': ['accuracy']
            }
            model.add(Activation('softmax'))
            probabilistic_model.add(Activation('softmax'))
        else:
            compile_params = {
                'loss': 'mse',
                'optimizer': 'adam',
                'metrics': ['mae']
            }

        model.compile(**compile_params)
        probabilistic_model.compile(**compile_params)
        self.model = model
        self.probabilistic_model = probabilistic_model

    def fit(self, dataset):
        weights_filename = 'runs/' + uuid.uuid4().hex + '.hdf5'
        cp = ModelCheckpoint(weights_filename, save_best_only=True)

        self.model.fit(
            x=dataset.X_train,
            y=dataset.y_train,
            validation_data=(dataset.X_val, dataset.y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[cp],
        )

        self.probabilistic_model.load_weights(weights_filename)
        self.model.set_weights(self.probabilistic_model.get_weights())
        os.remove(weights_filename)
        return self

    def predict(self, X, probabilistic=False):
        if probabilistic:
            model = self.probabilistic_model
        else:
            model = self.model
        return model.predict(X, batch_size=self.batch_size)


class CNN(BaseModel):

    def __init__(self, input_shape, num_classes, epochs=10, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model = model

        probabilistic_model = Sequential()
        probabilistic_model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(Conv2D(32, (3, 3)))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(MaxPooling2D(pool_size=(2, 2)))
        probabilistic_model.add(BayesianDropout(0.25))
        probabilistic_model.add(Conv2D(64, (3, 3), padding='same'))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(Conv2D(64, (3, 3)))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(MaxPooling2D(pool_size=(2, 2)))
        probabilistic_model.add(BayesianDropout(0.25))
        probabilistic_model.add(Flatten())
        probabilistic_model.add(Dense(512))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(BayesianDropout(0.5))
        probabilistic_model.add(Dense(num_classes))
        probabilistic_model.add(Activation('softmax'))
        probabilistic_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.probabilistic_model = probabilistic_model


class VGG(BaseModel):

    def __init__(self, input_shape, num_classes, epochs=10, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        model = VGG16(include_top=False, input_shape=input_shape)
        x = Flatten(name='flatten')(model.output)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x)
        self.model = Model(inputs=model.input, outputs=x)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        probabilistic_model = VGG16(include_top=False, input_shape=input_shape)
        x = Flatten(name='flatten')(probabilistic_model.output)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x)
        self.probabilistic_model = Model(inputs=probabilistic_model.input, outputs=x)
        self.probabilistic_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def load(settings):
    name = settings.pop('name')

    if name == 'logistic_regression':
        model = LogisticRegression(**settings)
    elif name == 'mlp':
        model = MLP(**settings)
    elif name == 'cnn':
        model = CNN(**settings)
    elif name == 'vgg':
        model = VGG(**settings)
    else:
        raise Exception('Unknown model {0}'.format(name))

    return model
