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
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable



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


    def predict_proba(self, X):
        return self.model.predict_proba(X, verbose=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class TorchCNN(nn.Module):
    def __init__(self):
        super(TorchCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.dropout2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNN(Model):

    def __init__(self, epochs=10, batch_size=32, inference='dropout', cuda=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.inference = inference
        self.cuda = cuda
        self.cnn = TorchCNN()
        if cuda:
            cnn.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.cnn.parameters())

    def fit(self, X, y):
        self.label_encoder = LabelEncoder().fit(y)
        n_labels = len(set(y))

        for epoch in range(self.epochs):
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)
            for end in tqdm.tqdm(range(self.batch_size, len(idxs), self.batch_size), desc='Epoch {}'.format(epoch+1)):
                X_batch = X[end-self.batch_size:end]
                y_batch = y[end-self.batch_size:end]
                images = Variable(torch.from_numpy(X_batch.astype('float32')))
                labels = Variable(torch.from_numpy(y_batch.astype('long')))

                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.cnn(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, X, probabilistic=False):
        if probabilistic:
            self.cnn.train()
        else:
            self.cnn.eval()

        idxs = np.arange(len(X))
        outputs = []
        for end in range(self.batch_size, len(idxs) + (len(idxs)%self.batch_size)+1, self.batch_size):
            X_batch = X[end-self.batch_size:end]
            images = Variable(torch.from_numpy(X_batch.astype('float32')))
            outputs.append(self.cnn(images).data.numpy())

        y = np.concatenate(outputs, axis=0)

        return y

    def predict(self, X, probabilistic=False):
        if probabilistic:
            self.cnn.train()
        else:
            self.cnn.eval()

        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        outputs = []
        for end in range(self.batch_size, len(idxs), self.batch_size):
            X_batch = X[end-self.batch_size:end]
            images = Variable(torch.from_numpy(X_batch.astype('float32')))
            outputs.append(self.cnn(images).data.numpy())
        y = np.concatenate(outputs, axis=0)

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
