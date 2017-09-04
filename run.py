import numpy as np
from sacred import Experiment
import datasets
import models

ex = Experiment('uncertainty-quality')


@ex.config
def cfg():
    seed = 1337

    dataset = {
        'name': 'digits',
        'test_size': 0.2,
    }

    model = {
        'name': 'logistic_regression',
        'inference': 'variational',
        'n_iter': 100,
        'n_samples': 5,
    }

    posterior_samples = 50


@ex.capture
def train(model, dataset):
    X, y = dataset.train()
    model.fit(X, y)


@ex.capture
def evaluate(model, dataset, posterior_samples):
    X_train, y_train = dataset.train()

    y_train_score = np.array([model.predict_proba(X_train)
                        for _ in range(posterior_samples)])

    y_train_pred = y_train_score.mean(axis=0).argmax(axis=1)
    print(y_train_pred[:10])
    y_train_pred = model.label_encoder.inverse_transform(y_train_pred)
    print(y_train_pred[:10])
    acc_train = (y_train == y_train_pred).mean()
    print('acc_train:', acc_train)

@ex.automain
def run(model, dataset):
    model = models.load(model)
    dataset = datasets.load(dataset)

    X_train, y_train = dataset.train()
    print(X_train.shape, y_train.shape)
    X_test, y_test = dataset.test()
    print(X_test.shape, y_test.shape)

    train(model, dataset)
    evaluate(model, dataset)
