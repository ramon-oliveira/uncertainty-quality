import numpy as np
import xgboost as xgb
from scipy.stats import entropy
from sklearn import linear_model


def uncertainty_std_mean(y_probabilistic):
    y_std_mean = y_probabilistic.std(axis=0).mean(axis=1)

    return y_std_mean


def uncertainty_entropy(y_score):
    y_entropy = []
    for y in y_score:
        y_entropy.append(entropy(y))
    y_entropy = np.array(y_entropy)

    return y_entropy


def uncertainty_mean_entropy(y_probabilistic):
    y_entropy = []
    for y_score_sample in y_probabilistic:
        uncertainty = uncertainty_entropy(y_score_sample)
        y_entropy.append(uncertainty)
    y_mean_entropy = np.array(y_entropy).mean(axis=0)

    return y_mean_entropy


def uncertainty_classifer(model, dataset, test_uncertainty):
    # x_val = np.vstack([dataset.x_train, dataset.x_val])
    # y_val = np.vstack([dataset.y_train, dataset.y_val])
    x_val, y_val = dataset.x_val, dataset.y_val

    y_deterministic = model.predict(x_val, probabilistic=False)
    y_probabilistic = model.predict(x_val, probabilistic=True)

    uncertainties = [
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
    ]

    x = np.zeros([len(y_val), len(uncertainties) + dataset.output_size])
    x[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    y = (y_val - y_probabilistic.mean(axis=0))**2

    for i, (name, func, y_score) in enumerate(uncertainties):
        uncertainty = func(y_score)
        x[:, i] = uncertainty

    print(test_uncertainty.max(), test_uncertainty.min())
    print(x.max(), x.min())

    # rg = xgb.XGBRegressor().fit(x, y)
    rg = linear_model.RidgeCV(alphas=np.linspace(0.1, 10, 20)).fit(x, y)
    uncertainty = rg.predict(test_uncertainty)
    return uncertainty


def evaluate(ex, model, dataset):
    x_test, y_test = dataset.x_test, dataset.y_test

    y_deterministic = model.predict(x_test, probabilistic=False)
    y_probabilistic = model.predict(x_test, probabilistic=True)

    y_true = dataset.yscaler.inverse_transform(y_test)
    y_pred = dataset.yscaler.inverse_transform(y_probabilistic.mean(axis=0))

    rmse_test = np.sqrt(np.mean((y_true.squeeze() - y_pred.squeeze())**2))
    ex.info['rmse_test'] = rmse_test
    print('rmse test: {0:.4f}'.format(rmse_test))

    uncertainties = [
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
    ]

    test_uncertainty = np.zeros([len(y_test), len(uncertainties) + dataset.output_size])
    test_uncertainty[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    for i, (name, func, y_score) in enumerate(uncertainties):
        uncertainty = func(y_score)
        l = np.array(sorted(zip(uncertainty, y_true, y_pred)))
        prop_acc = []
        for end in range(2, len(y_pred)):
            prop = end/len(y_pred)
            rmse = np.sqrt(np.mean((l[:end, 1].squeeze() - l[:end, 2].squeeze())**2))
            prop_acc.append([prop, rmse])
        print(name, np.trapz(y=np.array(prop_acc)[:, 1], x=np.array(prop_acc)[:, 0]))
        ex.info[name] = prop_acc
        test_uncertainty[:, i] = uncertainty

    uncertainty = uncertainty_classifer(model, dataset, test_uncertainty)
    l = np.array(sorted(zip(uncertainty, y_true, y_pred)))
    prop_acc = []
    for end in range(2, len(y_pred)):
        prop = end/len(y_pred)
        rmse = np.sqrt(np.mean((l[:end, 1].squeeze() - l[:end, 2].squeeze())**2))
        prop_acc.append([prop, rmse])
    print('uncertainty_classifer', np.trapz(y=np.array(prop_acc)[:, 1], x=np.array(prop_acc)[:, 0]))
    ex.info['uncertainty_classifer'] = prop_acc
