import numpy as np
import xgboost as xgb
from scipy.stats import entropy
from sklearn import linear_model
from scipy.stats import norm
from scipy.misc import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


def uncertainty_std_predicted_mean(y_probabilistic):
    return y_probabilistic.std(axis=0)[:, 0]


def uncertainty_predicted_std(y_probabilistic):
    return y_probabilistic.mean(axis=0)[:, 1]


def uncertainty_predicted_std_det(y_deterministic):
    return y_deterministic[:, 1]


def uncertainty_std_predicted_std(y_probabilistic):
    return -y_probabilistic.std(axis=0)[:, 1]


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

    y_true = dataset.yscaler.inverse_transform(y_val)
    y_pred = dataset.yscaler.inverse_transform(y_probabilistic.mean(axis=0))
    y_pred_det = dataset.yscaler.inverse_transform(y_deterministic)

    uncertainties = [
        ('uncertainty_std_predicted_mean', uncertainty_std_predicted_mean, (y_probabilistic)),
        ('uncertainty_predicted_std', uncertainty_predicted_std, (y_probabilistic)),
        ('uncertainty_predicted_std_det', uncertainty_predicted_std_det, (y_deterministic)),
        ('uncertainty_std_predicted_std', uncertainty_std_predicted_std, (y_probabilistic)),
        ('uncertainty_nll_gaussian', uncertainty_nll_gaussian, (y_deterministic, y_probabilistic)),
        ('uncertainty_nll_gaussian_std', uncertainty_nll_gaussian_std, (y_deterministic, y_probabilistic)),
    ]

    # use as input
    x = np.zeros([len(y_val), len(uncertainties) + dataset.output_size])
    x[:, len(uncertainties):] = y_probabilistic.mean(axis=0)

    # use as target
    # y = (y_true[:, 0] - y_pred[:, 0])**2
    samples = []
    for i in range(model.posterior_samples):
        samples.append(dataset.yscaler.inverse_transform(y_probabilistic[i]))
    samples = np.array(samples)
    y = []
    for i in range(len(y_true)):
        y.append(np.std(samples[:, i, 0])**2)
    y = np.array(y)

    for i, (name, func, y_score) in enumerate(uncertainties):
        uncertainty = func(y_score)
        x[:, i] = uncertainty

    # poly = PolynomialFeatures(2).fit(x)
    # x = poly.transform(x)
    xscaler = StandardScaler().fit(x)
    yscaler = StandardScaler().fit(y.reshape(-1, 1))
    x = xscaler.transform(x)
    y = yscaler.transform(y.reshape(-1, 1))

    # params = {
    #     'n_estimators': [100, 200],
    #     'learning_rate': [0.1, 0.01],
    #     'max_depth': [2, 3, 4],
    # }
    # rg = GridSearchCV(RandomForestRegressor(), params, scoring='neg_mean_squared_error', n_jobs=3).fit(x, y)
    # rg = RandomizedSearchCV(xgb.XGBRegressor(), params, scoring='neg_mean_squared_error', n_jobs=3).fit(x, y)
    rg = xgb.XGBRegressor().fit(x, y)
    # rg = linear_model.RidgeCV(alphas=np.linspace(0.01, 10, 100), cv=10, scoring='neg_mean_squared_error').fit(x, y)
    # test_uncertainty = poly.transform(test_uncertainty)
    test_uncertainty = xscaler.transform(test_uncertainty)
    uncertainty = yscaler.inverse_transform(rg.predict(test_uncertainty)).ravel()
    return uncertainty


def evaluate(model, dataset):
    info = {}
    x_test, y_test = dataset.x_test, dataset.y_test

    y_deterministic = model.predict(x_test, probabilistic=False)
    y_probabilistic = model.predict(x_test, probabilistic=True)

    y_true = dataset.yscaler.inverse_transform(y_test)
    y_pred = dataset.yscaler.inverse_transform(y_probabilistic.mean(axis=0))
    y_pred_det = dataset.yscaler.inverse_transform(y_deterministic)

    rmse_test = np.sqrt(np.mean((y_true[:, 0] - y_pred[:, 0])**2))
    info['rmse_test'] = rmse_test
    print('rmse test: {0:.4f}'.format(rmse_test))

    rmse_test_det = np.sqrt(np.mean((y_true[:, 0] - y_pred_det[:, 0])**2))
    info['rmse_test_det'] = rmse_test_det
    print('rmse test det: {0:.4f}'.format(rmse_test_det))

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
        l = np.array(sorted(zip(uncertainty, y_true[:, 0], y_pred[:, 0])))
        prop_acc = []
        for end in range(5, len(y_pred), 2):
            prop = end/len(y_pred)
            rmse = np.sqrt(np.mean((l[:end, 1].squeeze() - l[:end, 2].squeeze())**2))
            prop_acc.append([prop, rmse])

        auc = np.trapz(y=np.array(prop_acc)[:, 1], x=np.array(prop_acc)[:, 0])
        print(name, auc)
        info[name] = prop_acc
        info[name+'_auc'] = auc
        test_uncertainty[:, i] = uncertainty

    uncertainty = uncertainty_classifer(model, dataset, test_uncertainty)
    l = np.array(sorted(zip(uncertainty, y_true[:, 0], y_pred[:, 0])))
    prop_acc = []
    for end in range(5, len(y_pred), 2):
        prop = end/len(y_pred)
        rmse = np.sqrt(np.mean((l[:end, 1].squeeze() - l[:end, 2].squeeze())**2))
        prop_acc.append([prop, rmse])
    auc = np.trapz(y=np.array(prop_acc)[:, 1], x=np.array(prop_acc)[:, 0])
    print('uncertainty_classifer', auc)
    info['uncertainty_classifer'] = prop_acc
    info['uncertainty_classifer_auc'] = auc

    ll = []
    for x, mean, sigma2 in zip(y_pred[:, 0], y_true[:, 0], uncertainty):
        sigma2 = max(1e-6, sigma2)
        sigma = np.sqrt(sigma2)
        ll.append(norm.logpdf(x, loc=mean, scale=sigma))
    ll = np.array(ll)
    ll_test = np.mean(ll)
    print('log-likelihood normal uncertainty:', ll_test)
    info['ll_normal_uncertainty'] = ll_test

    ll = []
    for x, mean, log_std in zip(y_pred[:, 0], y_true[:, 0], y_pred[:, 1]):
        sigma = np.exp(log_std)
        ll.append(norm.logpdf(x, loc=mean, scale=sigma))
    ll = np.array(ll)
    ll_test = np.mean(ll)
    print('log-likelihood normal:', ll_test)
    info['ll_normal'] = ll_test

    y_hat = np.array([dataset.yscaler.inverse_transform(y_probabilistic[i]) for i in range(model.posterior_samples)])
    tau = 0.159707652696 # obtained from BO
    ll = [(logsumexp(-0.5 * tau * (y_true[i, 0] - y_hat[:, i, 0])**2, axis=0) - np.log(model.posterior_samples)
          - 0.5*np.log(2*np.pi) - 0.5*np.log(tau)) for i in range(len(y_true))]
    ll_test = np.mean(ll)
    print('log-likelihood tau:', ll_test)
    info['ll_tau'] = ll_test

    tau = uncertainty
    tau[tau <= 0] = 1e-6
    ll = [(logsumexp(-0.5 * tau[i] * (y_true[i, 0] - y_hat[:, i, 0])**2, axis=0) - np.log(model.posterior_samples)
          - 0.5*np.log(2*np.pi) - 0.5*np.log(tau[i])) for i in range(len(y_true))]
    ll_test = np.mean(ll)
    print('log-likelihood uncertainty:', ll_test)
    info['ll_uncertainty'] = ll_test

    return info
