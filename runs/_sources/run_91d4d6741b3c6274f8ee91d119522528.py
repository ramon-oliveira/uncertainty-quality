
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
ex.observers.append(FileStorageObserver.create('runs/'))


@ex.config
def cfg():
    seed = 1337

    dataset_settings = {
        'name': 'protein_structure',
    }

    model_settings = {
        'name': 'mlp',
        'dropout': 0.05,
        'layers': [50],
        'epochs': 300,
        'batch_size': 100,
        'posterior_samples': 100,
    }


@ex.capture
def train(model, dataset):
    model.fit(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val)


@ex.capture
def evaluate(model, dataset):
    if dataset.type == 'classification':
        evaluate_classification.evaluate(ex, model, dataset)
    else:
        evaluate_regression.evaluate(ex, model, dataset)


@ex.automain
def run(model_settings, dataset_settings, _log):
    _log.info('dataset_settings: ' + str(dataset_settings))
    _log.info('model_settings: ' + str(model_settings))
    dataset = datasets.load(dataset_settings)
    model_settings.update({'dataset': dataset})
    model = models.load(model_settings)

    train(model, dataset)
    evaluate(model, dataset)
