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
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from layers import BayesianDropout
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class BaseModel(object):

    def fit(self, dataset):
        es = EarlyStopping(monitor='val_loss', patience=10)
        weights_filename = 'runs/' + uuid.uuid4().hex + '.hdf5'
        cp = ModelCheckpoint(weights_filename, save_best_only=True)

        self.model.fit_generator(
            generator=dataset.train_generator,
            validation_data=dataset.validation_generator,
            steps_per_epoch=dataset.train_samples//self.batch_size,
            validation_steps=dataset.validation_samples//self.batch_size,
            epochs=self.epochs,
            workers=4,
            callbacks=[cp, es],
        )

        self.model.load_weights(weights_filename)
        self.probabilistic_model.set_weights(self.model.get_weights())

        return self

    def predict(self, gen, samples, probabilistic=False):
        if probabilistic:
            model = self.probabilistic_model
        else:
            model = self.model
        return model.predict_generator(
            generator=gen,
            steps=samples//self.batch_size,
            workers=4,
        )


class LogisticRegression(object):

    def __init__(self, n_samples=5, n_iter=100, inference='variational'):
        self.inference = inference
        self.n_samples = n_samples
        self.n_iter = n_iter
        self.model = None

    def fit(self, X, y):
        self.label_encoder = LabelEncoder().fit(y)
        y = self.oh_encoder.transform(y)

        DIN = X.shape[1]
        DOUT = len(set(y))

        X_data = tf.placeholder(tf.float32, [None, DIN])
        W = Normal(loc=tf.zeros([DIN, DOUT]), scale=tf.ones([DIN, DOUT]))
        b = Normal(loc=tf.zeros([DOUT]), scale=tf.ones([DOUT]))
        y_data = Categorical(logits=tf.matmul(X_data, W) + b)

        qW = Normal(loc=tf.Variable(tf.random_normal([DIN, DOUT])),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([DIN, DOUT]))))
        qb = Normal(loc=tf.Variable(tf.random_normal([DOUT])),
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([DOUT]))))

        self.model = ed.KLqp({W: qW, b: qb}, data={X_data: X, y_data: y})
        self.model.run(n_samples=self.n_samples, n_iter=self.n_iter)

        self.W = qW
        self.b = qb

    def predict_proba(self, X):
        W = self.W.eval()
        b = self.b.eval()
        return np.matmul(X, W) + b

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class MLP(object):

    def __init__(self, epochs=10, batch_size=32, inference='dropout'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.inference = inference
        self.model = None

    def fit(self, X, y):
        self.label_encoder = LabelEncoder().fit(y)
        n_labels = len(self.label_encoder.classes_)
        y = np.eye(n_labels)[y]

        model = Sequential()

        model.add(Dense(1024, activation='relu', input_shape=(X.shape[1],)))

        if self.inference == 'dropout':
            model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        else:
            model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))

        if self.inference == 'dropout':
            model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        else:
            model.add(Dropout(0.5))
        model.add(Dense(n_labels, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
        self.model = model
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X, verbose=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


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
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
        probabilistic_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
