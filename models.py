import os
import tqdm
import uuid
import numpy as np

from keras.models import Sequential
from keras.models import Model
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


class BaseModel(object):

    def __init__(self, epochs=100, batch_size=32, posterior_samples=50):
        self.epochs = epochs
        self.batch_size = batch_size
        self.posterior_samples = posterior_samples

    def fit(self, x_train, y_train, x_val, y_val):
        weights_filename = 'runs/' + uuid.uuid4().hex + '.hdf5'
        cp = ModelCheckpoint(weights_filename, save_best_only=True)

        self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(x_val, y_val),
            callbacks=[cp],
        )

        self.model.load_weights(weights_filename)
        self.probabilistic_model.set_weights(self.model.get_weights())
        os.remove(weights_filename)
        return self

    def predict(self, x, probabilistic=False):
        if probabilistic:
            model = self.probabilistic_model
            y_pred = [model.predict(x, batch_size=self.batch_size, verbose=0)
                      for i in tqdm.trange(self.posterior_samples, desc='sampling')]
            y_pred = np.array(y_pred)
        else:
            model = self.model
            y_pred = model.predict(x, batch_size=self.batch_size, verbose=0)

        return y_pred


class MLP(BaseModel):

    def __init__(self, dataset, dropout=0.5, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.dropout = dropout

        # deterministic model
        model = Sequential()
        model.add(Dense(1024, input_shape=dataset.input_shape))
        model.add(Dropout(dropout))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Dropout(dropout))
        model.add(Activation('relu'))
        model.add(Dense(dataset.output_size))

        # probabilistic model
        probabilistic_model = Sequential()
        probabilistic_model.add(Dense(1024, input_shape=dataset.input_shape))
        probabilistic_model.add(BayesianDropout(dropout))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(Dense(1024))
        probabilistic_model.add(BayesianDropout(dropout))
        probabilistic_model.add(Activation('relu'))
        probabilistic_model.add(Dense(dataset.output_size))

        if dataset.type == 'classification':
            model.add(Activation('softmax'))
            probabilistic_model.add(Activation('softmax'))
            compile_params = {
                'loss': 'categorical_crossentropy',
                'optimizer': 'adam',
                'metrics': ['accuracy']
            }
        else:
            compile_params = {
                'loss': 'mse',
                'optimizer': 'adam'
            }

        model.compile(**compile_params)
        probabilistic_model.compile(**compile_params)
        self.model = model
        self.probabilistic_model = probabilistic_model


class CNN(BaseModel):

    def __init__(self, dataset, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=dataset.input_shape))
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
        model.add(Dense(dataset.output_size))
        model.add(Activation('softmax'))
        opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model = model

        probabilistic_model = Sequential()
        probabilistic_model.add(Conv2D(32, (3, 3), padding='same', input_shape=dataset.input_shape))
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
        probabilistic_model.add(Dense(dataset.output_size))
        probabilistic_model.add(Activation('softmax'))
        opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
        probabilistic_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.probabilistic_model = probabilistic_model


class VGG(BaseModel):

    def __init__(self, dataset, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

        model = VGG16(include_top=False, input_shape=dataset.input_shape)
        x = Flatten(name='flatten')(model.output)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(dataset.output_size, activation='softmax', name='predictions')(x)
        self.model = Model(inputs=model.input, outputs=x)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        probabilistic_model = VGG16(include_top=False, input_shape=dataset.input_shape)
        x = Flatten(name='flatten')(probabilistic_model.output)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(dataset.output_size, activation='softmax', name='predictions')(x)
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
