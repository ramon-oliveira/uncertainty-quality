import numpy as np
from sacred import Experiment
import datasets
import models

ex = Experiment('uncertainty-quality')


@ex.config
def cfg():
    seed = 1337

    dataset = {
        'name': 'mnist',
        'test_size': 0.2,
    }

    model = {
        'name': 'logistic_regression',
        'inference': 'variational',
        'n_iter': 1000,
        'n_samples': 5,
    }

    posterior_samples = 50


@ex.capture
def train(model, dataset):
    X, y = dataset.train()
    model.fit(X, y)


@ex.capture
def std_uncertainty(y_score):
    y_pred = y_score.mean(axis=0).argmax(axis=1)
    y_std = np.array([y_score[:, i, c].std() for i, c in enumerate(y_pred)])

    return y_pred, y_std


@ex.capture
def evaluate(model, dataset, posterior_samples):
    X_train, y_train = dataset.train()
    X_test, y_test = dataset.test()

    y_train_score = np.array([model.predict_proba(X_train)
                              for _ in range(posterior_samples)])
    y_test_score = np.array([model.predict_proba(X_test)
                             for _ in range(posterior_samples)])

    y_train_pred = y_train_score.mean(axis=0).argmax(axis=1)
    y_train_pred = model.label_encoder.inverse_transform(y_train_pred)
    acc_train = (y_train == y_train_pred).mean()

    y_test_pred = y_test_score.mean(axis=0).argmax(axis=1)
    y_test_pred = model.label_encoder.inverse_transform(y_test_pred)
    acc_test = (y_test == y_test_pred).mean()

    print('acc_train:', acc_train)
    print('acc_test:', acc_test)
    ex.info['acc_train'] = acc_train
    ex.info['acc_test'] = acc_test

    y, y_uncertainty = std_uncertainty(y_test_score)

    l = np.array(sorted(zip(y_uncertainty, model.label_encoder.inverse_transform(y), y_test)))
    print((l[:, 2] == l[:, 1]).mean())
    acc = []
    for end in range(50, len(y), 10):
        acc.append((l[:end, 1] == l[:end, 2]).mean())

    print(acc)

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
