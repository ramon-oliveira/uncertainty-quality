import numpy as np
import xgboost as xgb
from scipy.stats import entropy
from sklearn import linear_model


def uncertainty_std_predicted_mean(y_probabilistic):
    return y_probabilistic.std(axis=0)[:, 0]


def uncertainty_predicted_std(y_probabilistic):
    return y_probabilistic.mean(axis=0)[:, 1]


def uncertainty_predicted_std_det(y_deterministic):
    return y_deterministic[:, 1]


def uncertainty_std_predicted_std(y_probabilistic):
    return y_probabilistic.std(axis=0)[:, 1]


def uncertainty_nll_gaussian(y_det_prob):
    y_deterministic, y_probabilistic = y_det_prob
    def log_gaussian2(x, mean, log_std):
        log_var = 2*log_std
        return -np.log(2*np.pi)/2.0 - log_var/2.0 - (x-mean)**2/(2*np.exp(log_var))

    uncertainty = []
    for i, (mean, log_std) in enumerate(y_deterministic):
        u = np.mean(log_gaussian2(y_probabilistic[:, 0], mean, log_std))
        uncertainty.append(u)
    return np.array(uncertainty)


def uncertainty_nll_gaussian_std(y_det_prob):
    y_deterministic, y_probabilistic = y_det_prob
    def log_gaussian2(x, mean, log_std):
        log_var = 2*log_std
        return -np.log(2*np.pi)/2.0 - log_var/2.0 - (x-mean)**2/(2*np.exp(log_var))

    uncertainty = []
    for i, (mean, log_std) in enumerate(y_deterministic):
        u = -np.std(log_gaussian2(y_probabilistic[:, 0], mean, log_std))
        uncertainty.append(u)
    return np.array(uncertainty)


def uncertainty_classifer(model, dataset, test_uncertainty):
    # x_val = np.vstack([dataset.x_train, dataset.x_val])
    # y_val = np.vstack([dataset.y_train, dataset.y_val])
    x_val, y_val = dataset.x_val, dataset.y_val

    y_deterministic = model.predict(x_val, probabilistic=False)
    y_probabilistic = model.predict(x_val, probabilistic=True)

    uncertainties = [
        ('uncertainty_std_predicted_mean', uncertainty_std_predicted_mean, (y_probabilistic)),
        ('uncertainty_predicted_std', uncertainty_predicted_std, (y_probabilistic)),
        ('uncertainty_predicted_std_det', uncertainty_predicted_std_det, (y_deterministic)),
        ('uncertainty_std_predicted_std', uncertainty_std_predicted_std, (y_probabilistic)),
        ('uncertainty_nll_gaussian', uncertainty_nll_gaussian, (y_deterministic, y_probabilistic)),
        ('uncertainty_nll_gaussian_std', uncertainty_nll_gaussian_std, (y_deterministic, y_probabilistic)),
    ]

    x = np.zeros([len(y_val), len(uncertainties) + dataset.output_size])
    x[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    # use as target
    y = (y_val[:, 0] - y_probabilistic.mean(axis=0)[:, 0])**2
    # y = []
    # for mean in y_val[:, 0]:
    #     u = np.sqrt(np.mean((y_probabilistic.mean(axis=0)[:, 0] - mean)**2))
    #     y.append(u)
    # y = np.array(y)

    for i, (name, func, y_score) in enumerate(uncertainties):
        uncertainty = func(y_score)
        x[:, i] = uncertainty

    print(x.shape, y.shape)

    rg = xgb.XGBRegressor(max_depth=2).fit(x, y)
    # rg = linear_model.RidgeCV(alphas=np.linspace(0.1, 10, 20)).fit(x, y)
    uncertainty = rg.predict(test_uncertainty)
    return uncertainty


def evaluate(ex, model, dataset):
    x_test, y_test = dataset.x_test, dataset.y_test

    y_deterministic = model.predict(x_test, probabilistic=False)
    y_probabilistic = model.predict(x_test, probabilistic=True)

    y_true = dataset.yscaler.inverse_transform(y_test)
    y_pred = dataset.yscaler.inverse_transform(y_probabilistic.mean(axis=0))

    rmse_test = np.sqrt(np.mean((y_true[:, 1] - y_pred[:, 1])**2))
    ex.info['rmse_test'] = rmse_test
    print('rmse test: {0:.4f}'.format(rmse_test))

    uncertainties = [
        ('uncertainty_std_predicted_mean', uncertainty_std_predicted_mean, (y_probabilistic)),
        ('uncertainty_predicted_std', uncertainty_predicted_std, (y_probabilistic)),
        ('uncertainty_predicted_std_det', uncertainty_predicted_std_det, (y_deterministic)),
        ('uncertainty_std_predicted_std', uncertainty_std_predicted_std, (y_probabilistic)),
        ('uncertainty_nll_gaussian', uncertainty_nll_gaussian, (y_deterministic, y_probabilistic)),
        ('uncertainty_nll_gaussian_std', uncertainty_nll_gaussian_std, (y_deterministic, y_probabilistic)),
    ]

    test_uncertainty = np.zeros([len(y_test), len(uncertainties) + dataset.output_size])
    test_uncertainty[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    for i, (name, func, y_score) in enumerate(uncertainties):
        uncertainty = func(y_score)
        print(uncertainty.shape)
        l = np.array(sorted(zip(uncertainty, y_true[:, 0], y_pred[:, 0])))
        prop_acc = []
        for end in range(100, len(y_pred), 20):
            prop = end/len(y_pred)
            rmse = np.sqrt(np.mean((l[:end, 1].squeeze() - l[:end, 2].squeeze())**2))
            prop_acc.append([prop, rmse])
        print(name, np.trapz(y=np.array(prop_acc)[:, 1], x=np.array(prop_acc)[:, 0]))
        ex.info[name] = prop_acc
        test_uncertainty[:, i] = uncertainty

    uncertainty = uncertainty_classifer(model, dataset, test_uncertainty)
    l = np.array(sorted(zip(uncertainty, y_true[:, 0], y_pred[:, 0])))
    prop_acc = []
    for end in range(100, len(y_pred), 20):
        prop = end/len(y_pred)
        rmse = np.sqrt(np.mean((l[:end, 1].squeeze() - l[:end, 2].squeeze())**2))
        prop_acc.append([prop, rmse])
    print('uncertainty_classifer', np.trapz(y=np.array(prop_acc)[:, 1], x=np.array(prop_acc)[:, 0]))
    ex.info['uncertainty_classifer'] = prop_acc
