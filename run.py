import xgboost as xgb
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
    epochs = 100
    batch_size = 100
    posterior_samples = 50

    dataset_settings = {
        'name': 'mnist',
        'batch_size': batch_size,
    }

    model_settings = {
        'name': 'cnn',
        'batch_size': batch_size,
        'epochs': epochs,
    }


@ex.capture
def train(model, dataset):
    model.fit(dataset)


def uncertainty_std_argmax(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)
    y_std = np.array([y_probabilistic[:, i, c].std()
                      for i, c in enumerate(y_pred)])

    return y_pred, y_std


def uncertainty_std_mean(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)
    y_std_mean = y_probabilistic.std(axis=0).mean(axis=1)

    return y_pred, y_std_mean


def uncertainty_entropy(y_score):
    y_entropy = []
    for y in y_score:
        y_entropy.append(entropy(y))
    y_entropy = np.array(y_entropy)
    y_pred = y_score.argmax(axis=1)

    return y_pred, y_entropy


def uncertainty_mean_entropy(y_probabilistic):
    y_entropy = []
    for y_score_sample in y_probabilistic:
        _, uncertainty = uncertainty_entropy(y_score_sample)
        y_entropy.append(uncertainty)
    y_mean_entropy = np.array(y_entropy).mean(axis=0)
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)

    return y_pred, y_mean_entropy


@ex.capture
def uncertainty_classifer(model, dataset, X_pred_uncertainty, posterior_samples):
    y_val = []
    for _, y in dataset.validation_generator:
        y_val.extend(y.argmax(axis=1))
        if len(y_val) >= dataset.validation_samples: break
    y_val = np.array(y_val)

    y_deterministic = model.predict(gen=dataset.validation_generator,
                                    samples=dataset.validation_samples,
                                    probabilistic=False)
    y_probabilistic = [model.predict(gen=dataset.validation_generator,
                                     samples=dataset.validation_samples,
                                     probabilistic=True)
                       for i in tqdm.trange(posterior_samples, desc='sampling validation')]
    y_probabilistic = np.array(y_probabilistic)

    uncertainties = [
        ('uncertainty_std_argmax', uncertainty_std_argmax, y_probabilistic),
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
        ('uncertainty_entropy_mean', uncertainty_entropy, y_probabilistic.mean(axis=0)),
        ('uncertainty_mean_entropy', uncertainty_mean_entropy, y_probabilistic),
    ]

    X = np.zeros([len(y_val), len(uncertainties) + dataset.num_classes])
    X[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    y = (y_val != y_probabilistic.mean(axis=0).argmax(axis=1))
    for i, (name, func, y_score) in enumerate(uncertainties):
        _, uncertainty = func(y_score)
        X[:, i] = uncertainty

    clf = xgb.XGBClassifier().fit(X, y)
    return clf.predict_proba(X_pred_uncertainty)[:, 1]


@ex.capture
def evaluate(model, dataset, posterior_samples, _log):
    y_test = []
    for _, y in dataset.test_generator:
        y_test.extend(y.argmax(axis=1))
        if len(y_test) >= dataset.test_samples: break
    y_test = np.array(y_test)

    y_deterministic = model.predict(gen=dataset.test_generator,
                                    samples=dataset.test_samples,
                                    probabilistic=False)
    y_probabilistic = [model.predict(gen=dataset.test_generator,
                                     samples=dataset.test_samples,
                                     probabilistic=True)
                       for i in tqdm.trange(posterior_samples, desc='sampling')]
    y_probabilistic = np.array(y_probabilistic)
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)

    acc_test = (y_test.ravel() == y_pred.ravel()).mean()
    ex.info['accuracy_test'] = acc_test
    _log.info('test accuracy: {0:.2f}'.format(acc_test))

    uncertainties = [
        ('uncertainty_std_argmax', uncertainty_std_argmax, y_probabilistic),
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
        ('uncertainty_entropy_mean', uncertainty_entropy, y_probabilistic.mean(axis=0)),
        ('uncertainty_mean_entropy', uncertainty_mean_entropy, y_probabilistic),
    ]

    X_pred_uncertainty = np.zeros([len(y_test), len(uncertainties) + dataset.num_classes])
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
