
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
ex.observers.append(FileStorageObserver.create('runs/', template='template_regression.html'))


@ex.config
def cfg():
    seed = 1337
    num_experiments = 10

    dataset_settings = {
        'name': 'protein_structure',
    }

    model_settings = {
        'name': 'mlp',
        'dropout': 0.05,
        'layers': [100, 100],
        'epochs': 300,
        'batch_size': 100,
        'posterior_samples': 1000,
    }


@ex.capture
def train(model, dataset):
    model.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val)


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
    for i in range(num_experiments):
        print('#'*10, 'Run', i+1, '#'*10)
        dataset = datasets.load(dataset_settings)
        model_settings.update({'dataset': dataset})
        model = models.load(model_settings)
        train(model, dataset)
        ex.info['evaluations'].append(evaluate(model, dataset))
