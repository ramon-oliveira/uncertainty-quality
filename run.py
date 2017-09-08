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

ex = Experiment('uncertainty-quality')
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver.create('runs/'))


@ex.config
def cfg():
    seed = 1337

    dataset_settings = {
        'name': 'cifar10',
        'test_size': 0.2,
    }

    model_settings = {
        'name': 'cnn',
        'epochs': 20,
    }

    posterior_samples = 50


@ex.capture
def train(model, dataset):
    X_train, y_train = dataset.train_data
    y_train = np.eye(dataset.num_classes)[y_train.ravel()]
    X_val, y_val = dataset.validation_data
    y_val = np.eye(dataset.num_classes)[y_val.ravel()]

    model.fit(X_train, y_train, X_val, y_val)


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
    X_test = X_test.reshape(-1, *model.input_shape)

    y_test_score = np.array([model.predict_proba(X_test)
                             for _ in tqdm.tqdm(range(posterior_samples), desc='sampling')])

    y_test_score_proba = np.array([model.predict_proba(X_test, probabilistic=True)
                                   for _ in tqdm.tqdm(range(posterior_samples), desc='sampling')])

    y_test_pred = y_test_score.mean(axis=0).argmax(axis=1)

    acc_test = (y_test == y_test_pred).mean()
    _log.info('test accuracy: {0:.2f}'.format(acc_test*100))
    ex.info['accuracy_test'] = acc_test

    y, y_uncertainty = std_uncertainty(y_test_score_proba)
    l = np.array(sorted(zip(y_uncertainty, y, y_test)))
    prop_acc = []
    for end in range(50, len(y), 20):
        prop = end/len(y)
        acc = (l[:end, 1] == l[:end, 2]).mean()
        prop_acc.append([prop, acc])
    ex.info['prop_acc_uncertainty_std'] = prop_acc

    y, y_uncertainty = entropy_uncertainty(y_test_score)
    l = np.array(sorted(zip(y_uncertainty, y, y_test)))
    prop_acc = []
    for end in range(50, len(y), 10):
        prop = end/len(y)
        acc = (l[:end, 1] == l[:end, 2]).mean()
        prop_acc.append([prop, acc])
    ex.info['prop_acc_uncertainty_entropy'] = prop_acc

    y, y_uncertainty = entropy_uncertainty(y_test_score_proba)
    l = np.array(sorted(zip(y_uncertainty, y, y_test)))
    prop_acc = []
    for end in range(50, len(y), 10):
        prop = end/len(y)
        acc = (l[:end, 1] == l[:end, 2]).mean()
        prop_acc.append([prop, acc])
    ex.info['prop_acc_uncertainty_entropy_proba'] = prop_acc


@ex.automain
def run(model_settings, dataset_settings, _log):
    dataset = datasets.load(dataset_settings)

    model_settings.update({
        'input_shape': dataset.input_shape,
        'num_classes': dataset.num_classes,
    })
    model = models.load(model_settings)

    _log.info('dataset_settings: ' + str(dataset_settings))
    _log.info('model_settings: ' + str(model_settings))

    train(model, dataset)
    evaluate(model, dataset)
