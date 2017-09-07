from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import edward as ed
import tqdm
from edward.models import Normal, Categorical, Multinomial
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from layers import BayesianDropout
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Model(ABC):

    @abstractmethod
    def fit(X, y):
        pass

    @abstractmethod
    def predict_proba(X):
        pass

    @abstractmethod
    def predict(X):
        pass


class LogisticRegression(Model):

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


class MLP(Model):

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


class CNN(Model):

    def __init__(self, input_shape, num_classes, epochs=10, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.model = model

        proba_model = Sequential()
        proba_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        proba_model.add(Conv2D(64, (3, 3), activation='relu'))
        proba_model.add(MaxPooling2D(pool_size=(2, 2)))
        proba_model.add(BayesianDropout(0.5))
        proba_model.add(Flatten())
        proba_model.add(Dense(128, activation='relu'))
        proba_model.add(BayesianDropout(0.5))
        proba_model.add(Dense(num_classes, activation='softmax'))
        proba_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.proba_model = proba_model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
        self.proba_model.set_weights(self.model.get_weights())

    def predict_proba(self, X, probabilistic=False):
        model = self.model
        if probabilistic:
            model = self.proba_model

        return model.predict(X, batch_size=self.batch_size, verbose=0)

    def predict(self, X, probabilistic=False):
        y = self.predict_proba(X, probabilistic)
        return y.argmax(axis=1)


def load(model):
    name = model.pop('name')

    if name == 'logistic_regression':
        model = LogisticRegression(**model)
    elif name == 'mlp':
        model = MLP(**model)
    elif name == 'cnn':
        model = CNN(**model)
    else:
        raise Exception('Unknown model {0}'.format(model))

    return model
