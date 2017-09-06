import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import datasets
import models
from scipy.stats import entropy

ex = Experiment('uncertainty-quality')
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver.create('runs/'))


@ex.config
def cfg():
    seed = 1337

    dataset = {
        # 'name': 'mnist',
        'name': 'cifar10',
        'test_size': 0.2,
    }


    # model = {
    #     'name': 'logistic_regression',
    #     'inference': 'variational',
    #     'n_iter': 1000,
    #     'n_samples': 5,
    # }

    model = {
        'name': 'mlp',
        'epochs': 1,
        'inference': 'maximum_likelihood',
        # 'inference': 'dropout',
    }

    posterior_samples = 50


@ex.capture
def train(model, dataset):
    X, y = dataset.train()
    model.fit(X, y)
    # model.model.save_weights('runs/weights.hdf5')
    # model.model.load_weights('runs/weights.hdf5')


@ex.capture
def std_uncertainty(y_score):
    y_pred = y_score.mean(axis=0).argmax(axis=1)
    y_std = np.array([y_score[:, i, c].std() for i, c in enumerate(y_pred)])

    return y_pred, y_std


@ex.capture
def entropy_uncertainty(y_score):
    y_pred_mean = y_score.mean(axis=0)
    y_entropy = []
    for y in y_pred_mean:
        y_entropy.append(entropy(y))
    y_entropy = np.array(y_entropy)
    y_pred = y_score.mean(axis=0).argmax(axis=1)

    return y_pred, y_entropy


@ex.capture
def evaluate(model, dataset, posterior_samples, _log):
    X_test, y_test = dataset.test()

    y_test_score = np.array([model.predict_proba(X_test)
                             for _ in range(posterior_samples)])

    y_test_pred = y_test_score.mean(axis=0).argmax(axis=1)
    y_test_pred = model.label_encoder.inverse_transform(y_test_pred)
    acc_test = (y_test == y_test_pred).mean()
    _log.info('test accuracy: {0:.2f}'.format(acc_test*100))
    ex.info['accuracy_test'] = acc_test

    y, y_uncertainty = std_uncertainty(y_test_score)
    l = np.array(sorted(zip(y_uncertainty, model.label_encoder.inverse_transform(y), y_test)))
    prop_acc = []
    for end in range(50, len(y), 10):
        prop = end/len(y)
        acc = (l[:end, 1] == l[:end, 2]).mean()
        prop_acc.append([prop, acc])
    ex.info['prop_acc_uncertainty_std'] = prop_acc

    y, y_uncertainty = entropy_uncertainty(y_test_score)
    l = np.array(sorted(zip(y_uncertainty, model.label_encoder.inverse_transform(y), y_test)))
    prop_acc = []
    for end in range(50, len(y), 10):
        prop = end/len(y)
        acc = (l[:end, 1] == l[:end, 2]).mean()
        prop_acc.append([prop, acc])
    ex.info['prop_acc_uncertainty_entropy'] = prop_acc

@ex.automain
def run(model, dataset):
    model = models.load(model)
    dataset = datasets.load(dataset)

    X_train, y_train = dataset.train()
    X_test, y_test = dataset.test()

    train(model, dataset)
    evaluate(model, dataset)
