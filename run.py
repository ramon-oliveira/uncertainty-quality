
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import datasets
import models

import evaluate_classification
import evaluate_regression

ex = Experiment('uncertainty-quality')
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver.create('runs/', template='template_classification.html'))


@ex.config
def cfg():
    seed = 1337
    num_experiments = 1

    dataset_settings = {
        'name': 'cifar10',
    }

    model_settings = {
        'name': 'vggtop',
        'epochs': 100,
        'batch_size': 100,
        'posterior_samples': 100,
    }


@ex.capture
def train(model, dataset):
    model.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val)
    # weights_filename = 'cifar10_weights.hdf5'
    # weights_filename = 'runs/20d35c0c7d7b47a7bcd664c52ec3063f.hdf5'
    # model.model.load_weights(weights_filename)
    # model.probabilistic_model.set_weights(model.model.get_weights())


@ex.capture
def evaluate(model, dataset):
    if dataset.type == 'classification':
        return evaluate_classification.evaluate(model, dataset)
    else:
        return evaluate_regression.evaluate(model, dataset)


@ex.automain
def run(model_settings, dataset_settings, num_experiments, _log):
    _log.info('dataset_settings: ' + str(dataset_settings))
    _log.info('model_settings: ' + str(model_settings))
    ex.info['evaluations'] = []
    for i in range(1, num_experiments+1):
        print('#'*10, 'Run', i, '#'*10)
        dataset_settings['train_size'] = i/num_experiments
        dataset = datasets.load(dataset_settings)
        model_settings.update({'dataset': dataset})
        model = models.load(model_settings)
        train(model, dataset)
        ex.info['evaluations'].append(evaluate(model, dataset))
    ex.info['sota'] = dataset.sota
