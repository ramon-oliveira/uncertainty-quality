import xgboost as xgb
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import datasets
import models
import tqdm
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression

ex = Experiment('uncertainty-quality')
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver.create('runs/'))


@ex.config
def cfg():
    seed = 1337

    dataset_settings = {
        'name': 'cifar10',
    }

    model_settings = {
        'name': 'cnn',
        'epochs': 100,
    }

    posterior_samples = 50


@ex.capture
def train(model, dataset):
    # X_train, y_train = dataset.train_data
    # y_train = np.eye(dataset.num_classes)[y_train.ravel()]
    # X_val, y_val = dataset.validation_data
    # y_val = np.eye(dataset.num_classes)[y_val.ravel()]
    # model.fit(X_train, y_train, X_val, y_val)

    model.model.load_weights('runs/5e7954645601424ca829d5ea98f6bc87.hdf5')
    model.probabilistic_model.load_weights('runs/5e7954645601424ca829d5ea98f6bc87.hdf5')

def uncertainty_std_argmax(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)
    y_std = np.array([y_probabilistic[:, i, c].std()
                      for i, c in enumerate(y_pred)])

    return y_pred, y_std


def uncertainty_std_mean(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)
    y_std_mean = y_probabilistic.std(axis=0).mean(axis=1)

    return y_pred, y_std_mean


def uncertainty_entropy(y_deterministic):
    y_entropy = []
    for y in y_deterministic:
        y_entropy.append(entropy(y))
    y_entropy = np.array(y_entropy)
    y_pred = y_deterministic.argmax(axis=1)

    return y_pred, y_entropy


@ex.capture
def uncertainty_classifer(model, dataset, X_pred_uncertainty, posterior_samples):
    X_val, y_val = dataset.validation_data

    y_deterministic = model.predict_proba(X_val, probabilistic=False)
    y_probabilistic = [model.predict_proba(X_val, probabilistic=True)
                       for i in range(posterior_samples)]
    y_probabilistic = np.array(y_probabilistic)

    uncertainties = [
        ('uncertainty_std_argmax', uncertainty_std_argmax, y_probabilistic),
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
    ]

    X = np.zeros([len(X_val), len(uncertainties) + dataset.num_classes])
    X[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    y = (y_val != y_probabilistic.mean(axis=0).argmax(axis=1))
    for i, (name, func, y_score) in enumerate(uncertainties):
        _, uncertainty = func(y_score)
        X[:, i] = uncertainty

    clf = xgb.XGBClassifier().fit(X, y)
    return clf.predict_proba(X_pred_uncertainty)[:, 1]



@ex.capture
def evaluate(model, dataset, posterior_samples, _log):
    X_test, y_test = dataset.test_data

    y_deterministic = model.predict_proba(X_test, probabilistic=False)
    y_probabilistic = [model.predict_proba(X_test, probabilistic=True)
                       for i in tqdm.trange(posterior_samples, desc='sampling')]
    y_probabilistic = np.array(y_probabilistic)
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)

    acc_test = (y_test == y_pred).mean()
    ex.info['accuracy_test'] = acc_test
    _log.info('test accuracy: {0:.2f}'.format(acc_test))

    uncertainties = [
        ('uncertainty_std_argmax', uncertainty_std_argmax, y_probabilistic),
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
    ]

    X_pred_uncertainty = np.zeros([len(X_test), len(uncertainties) + dataset.num_classes])
    X_pred_uncertainty[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    for i, (name, func, y_score) in enumerate(uncertainties):
        y_hat, uncertainty = func(y_score)
        l = np.array(sorted(zip(uncertainty, y_test, y_hat)))
        prop_acc = []
        for end in range(50, len(y_hat), 20):
            prop = end/len(y_hat)
            acc = (l[:end, 1] == l[:end, 2]).mean()
            prop_acc.append([prop, acc])
        ex.info[name] = prop_acc

        X_pred_uncertainty[:, i] = uncertainty

    uncertainty = uncertainty_classifer(model, dataset, X_pred_uncertainty)
    l = np.array(sorted(zip(uncertainty, y_test, y_pred)))
    prop_acc = []
    for end in range(50, len(y_pred), 20):
        prop = end/len(y_pred)
        acc = (l[:end, 1] == l[:end, 2]).mean()
        prop_acc.append([prop, acc])
    ex.info['uncertainty_classifer'] = prop_acc


@ex.automain
def run(model_settings, dataset_settings, _log):
    _log.info('dataset_settings: ' + str(dataset_settings))
    dataset = datasets.load(dataset_settings)

    model_settings.update({
        'input_shape': dataset.input_shape,
        'num_classes': dataset.num_classes,
    })
    _log.info('model_settings: ' + str(model_settings))
    model = models.load(model_settings)

    train(model, dataset)
    evaluate(model, dataset)
