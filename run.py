
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
storage_observer = FileStorageObserver.create('runs/', template='template_classification.html')
ex.observers.append(storage_observer)


@ex.config
def cfg():
    seed = 1337
    num_experiments = 5

    dataset_settings = {
        'name': 'cifar10',
    }

    model_settings = {
        'name': 'vggtop',
        'epochs': 100,
        'batch_size': 100,
        'posterior_samples': 100,
    }


@ex.named_config
def cfg_cifar100():
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
    model.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val, save_dir=storage_observer.dir)


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
