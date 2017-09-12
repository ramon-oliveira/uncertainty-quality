import tqdm
from sklearn import metrics

def uncertainty_std_argmax(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0)
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


def uncertainty_classifer(model, dataset, X_pred_uncertainty, posterior_samples):
    y_val = []
    for _, y in dataset.validation():
        y_val.extend(y.argmax(axis=1))
        if len(y_val) >= dataset.validation_samples: break
    y_val = np.array(y_val)

    y_deterministic = model.predict(gen=dataset.validation(),
                                    samples=dataset.validation_samples,
                                    probabilistic=False)
    y_probabilistic = [model.predict(gen=dataset.validation(),
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


def evaluate(model, dataset, posterior_samples, _log):
    y_deterministic = model.predict(dataset.X_test, probabilistic=False)
    y_probabilistic = [model.predict(dataset.X_test, probabilistic=True)
                       for i in tqdm.trange(posterior_samples, desc='sampling')]
    y_probabilistic = np.array(y_probabilistic)
    y_pred = y_probabilistic.mean(axis=0)

    print(len(dataset.y_test), len(y_pred))
    print(y_test[:10])
    print(y_pred[:10])
    print(y_deterministic.argmax(axis=1)[:10])
    mse_test = metrics.mean_squared_error(dataset.y_test, y_pred)
    mse_test_det = metrics.mean_squared_error(dataset.y_test, y_deterministic)
    ex.info['mse_test'] = mse_test
    _log.info('test mse: {0:.4f}'.format(mse_test))
    _log.info('test mse deterministic: {0:.4f}'.format(mse_test_det))

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
