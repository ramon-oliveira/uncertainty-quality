import numpy as np
import xgboost as xgb
from scipy.stats import entropy
import base64
from PIL import Image
import io
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection


class StackedModel(object):

    def __init__(self, base_clf, clf_params, n_clf):
        self.base_clf = base_clf
        self.clf_params = clf_params
        self.n_clf = n_clf

    def fit(self, X, y):
        skf = model_selection.StratifiedKFold(n_splits=self.n_clf)
        self.clfs = []
        for idxs, _ in skf.split(X, y):
            self.clfs.append(self.base_clf(**self.clf_params).fit(X[idxs], y[idxs]))

        X_proba = np.array([clf.predict_proba(X)[:, 1] for clf in self.clfs]).T
        self.top_clf = self.base_clf(**self.clf_params).fit(X_proba, y)

        return self

    def predict(self, X):
        X_proba = np.array([clf.predict_proba(X)[:, 1] for clf in self.clfs]).T
        return self.top_clf.predict(X_proba)

    def predict_proba(self, X):
        X_proba = np.array([clf.predict_proba(X)[:, 1] for clf in self.clfs]).T
        return self.top_clf.predict_proba(X_proba)


def top_n(y_true, y_score, n=5):
    y_pred = np.argsort(y_score, axis=1)[:, -n:]
    acc = []
    for t, p in zip(y_true, y_pred):
        if t in p:
            acc.append(1)
        else:
            acc.append(0)
    return np.array(acc)


def uncertainty_std_argmax(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)
    y_std = np.array([y_probabilistic[:, i, c].std()
                      for i, c in enumerate(y_pred)])

    return y_pred, y_std


def uncertainty_mean_argmax(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)
    y_mean = np.array([y_probabilistic[:, i, c].mean()
                      for i, c in enumerate(y_pred)])

    return y_pred, 1 - y_mean


def uncertainty_argmax(y_deterministic):
    y_pred = y_deterministic.argmax(axis=1)
    y_mean = y_deterministic.max(axis=1)

    return y_pred, 1 - y_mean


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
    y_entropy = (y_entropy - y_entropy.min())/(y_entropy.max() - y_entropy.min())
    return y_pred, y_entropy


def uncertainty_mean_entropy(y_probabilistic):
    y_entropy = []
    for y_score_sample in y_probabilistic:
        _, uncertainty = uncertainty_entropy(y_score_sample)
        y_entropy.append(uncertainty)
    y_mean_entropy = np.array(y_entropy).mean(axis=0)
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)

    return y_pred, y_mean_entropy


def uncertainty_classifer(model, dataset, test_uncertainty):
    x_val, y_val = dataset.x_val, dataset.y_val.argmax(axis=1)
    # x_train, y_train = dataset.x_train, dataset.y_train.argmax(axis=1)
    # print(x_train.shape, y_train.shape)
    # print(x_val.shape, y_val.shape)
    # x_val = np.concatenate([x_train, x_val], axis=0)
    # y_val = np.concatenate([y_train, y_val], axis=0)

    y_deterministic = model.predict(x_val, probabilistic=False)
    y_probabilistic = model.predict(x_val, probabilistic=True)

    uncertainties = [
        ('uncertainty_std_argmax', uncertainty_std_argmax, y_probabilistic),
        ('uncertainty_mean_argmax', uncertainty_mean_argmax, y_probabilistic),
        ('uncertainty_argmax', uncertainty_argmax, y_deterministic),
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
        ('uncertainty_entropy_mean', uncertainty_entropy, y_probabilistic.mean(axis=0)),
        ('uncertainty_mean_entropy', uncertainty_mean_entropy, y_probabilistic),
    ]

    x = np.zeros([len(y_val), len(uncertainties) + dataset.output_size])
    x[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    y = (y_val != y_probabilistic.mean(axis=0).argmax(axis=1))
    # y = 1 - top_n(y_val, y_probabilistic.mean(axis=0))
    for i, (name, func, y_score) in enumerate(uncertainties):
        _, uncertainty = func(y_score)
        x[:, i] = uncertainty

    clf = StackedModel(xgb.XGBClassifier, {}, 10).fit(x, y)
    # clf = xgb.XGBClassifier().fit(x, y)
    # clf = linear_model.LogisticRegression().fit(x, y)
    return clf.predict_proba(test_uncertainty)[:, 1]


def uncertainty_metrics(info, suffix, success, uncertainty):
    brier = metrics.brier_score_loss(y_true=success, y_prob=1-uncertainty)
    info['brier_'+suffix] = brier
    print('brier_'+suffix, brier)

    auc_hendricks = metrics.roc_auc_score(y_true=success, y_score=1-uncertainty)
    info['auc_hendricks_'+suffix] = auc_hendricks
    print('auc_hendricks_'+suffix, auc_hendricks)

    precision, recall, _ = metrics.precision_recall_curve(y_true=success, probas_pred=1-uncertainty)
    aupr = metrics.auc(recall, precision)
    info['aupr_hendricks_success_'+suffix] = aupr
    print('aupr_hendricks_success_'+suffix, aupr)

    precision, recall, _ = metrics.precision_recall_curve(y_true=1-success, probas_pred=(1-uncertainty)*-1)
    aupr = metrics.auc(recall, precision)
    info['aupr_hendricks_fail_'+suffix] = aupr
    print('aupr_hendricks_fail_'+suffix, aupr)



def evaluate(model, dataset):
    info = {}
    x_test, y_test = dataset.x_test, dataset.y_test.argmax(axis=1)

    y_deterministic = model.predict(x_test, probabilistic=False)
    y_probabilistic = model.predict(x_test, probabilistic=True)
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)

    acc_test = (y_test == y_pred).mean()
    info['accuracy_test'] = acc_test
    print('test accuracy: {0:.4f}'.format(acc_test))

    acc_top5_test = np.mean(top_n(y_test, y_probabilistic.mean(axis=0)))
    info['accuracy_top5_test'] = acc_top5_test
    print('test accuracy top 5: {0:.4f}'.format(acc_top5_test))

    uncertainties = [
        ('uncertainty_std_argmax', uncertainty_std_argmax, y_probabilistic),
        ('uncertainty_mean_argmax', uncertainty_mean_argmax, y_probabilistic),
        ('uncertainty_argmax', uncertainty_argmax, y_deterministic),
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
        ('uncertainty_entropy_mean', uncertainty_entropy, y_probabilistic.mean(axis=0)),
        ('uncertainty_mean_entropy', uncertainty_mean_entropy, y_probabilistic),
    ]

    test_uncertainty = np.zeros([len(y_test), len(uncertainties) + dataset.output_size])
    test_uncertainty[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    for i, (name, func, y_score) in enumerate(uncertainties):
        y_hat, uncertainty = func(y_score)
        l = np.array(sorted(zip(uncertainty, y_test, y_hat)))
        prop_acc = []
        for end in range(50, len(y_hat), 20):
            prop = end/len(y_hat)
            acc = (l[:end, 1] == l[:end, 2]).mean()
            prop_acc.append([prop, acc])
        info[name] = prop_acc
        auc = np.trapz(y=np.array(prop_acc)[:, 1], x=np.array(prop_acc)[:, 0])
        info[name+'_auc'] = auc
        print(name+'_auc:', auc)

        test_uncertainty[:, i] = uncertainty

    uncertainty = uncertainty_classifer(model, dataset, test_uncertainty)
    l = np.array(sorted(zip(uncertainty, y_test, y_pred, np.arange(len(y_test), dtype='int'))))
    prop_acc = []
    for end in range(50, len(y_pred), 20):
        prop = end/len(y_pred)
        acc = (l[:end, 1] == l[:end, 2]).mean()
        prop_acc.append([prop, acc])
    info['uncertainty_classifer'] = prop_acc
    auc = np.trapz(y=np.array(prop_acc)[:, 1], x=np.array(prop_acc)[:, 0])
    info['uncertainty_classifer_auc'] = auc
    print('uncertainty_classifer_auc:', auc)

    success = (y_test == y_pred)
    success_det = (y_test == y_deterministic.argmax(axis=1))
    # success = top_n(y_test, y_probabilistic.mean(axis=0))
    # success_det = top_n(y_test, y_deterministic.argmax(axis=1))

    max_proba = y_probabilistic.mean(axis=0).max(axis=1)
    _, std_max_proba = uncertainty_std_argmax(y_probabilistic)
    _, entropy_uncertainty = uncertainty_entropy(y_deterministic)

    print('--'*10, 'MAX PROBA', '--'*10)
    uncertainty_metrics(info, 'maxproba', success, 1 - max_proba)
    print('--'*10, 'MAX PROBA DET', '--'*10)
    uncertainty_metrics(info, 'maxprobadet', success_det, 1 - y_deterministic.max(axis=1))
    print('--'*10, 'STD MAX PROBA', '--'*10)
    uncertainty_metrics(info, 'stdmaxproba', success, std_max_proba)
    print('--'*10, 'ENTROPY', '--'*10)
    uncertainty_metrics(info, 'entropy', success_det, entropy_uncertainty)
    print('--'*10, 'STACKING', '--'*10)
    uncertainty_metrics(info, 'stacking', success, uncertainty)

    if len(x_test.shape) < 4:
        return info

    classes = dataset.classes
    examples = []
    for i, (u, t, p, idx) in enumerate(l[:12]):
        img = x_test[int(idx)]*255
        img = img.astype('uint8')
        if img.shape[-1] == 1:
            img = Image.fromarray(img.squeeze())
        else:
            img = Image.fromarray(img, 'RGB')
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        examples.append({
            'uncertainty': u,
            'true': classes[int(t)],
            'predicted': classes[int(p)],
            'base64': img_str
        })

    for i, (u, t, p, idx) in enumerate(l[-12:], start=3):
        img = x_test[int(idx)]*255
        img = img.astype('uint8')
        if img.shape[-1] == 1:
            img = Image.fromarray(img.squeeze())
        else:
            img = Image.fromarray(img, 'RGB')
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        examples.append({
            'uncertainty': u,
            'true': classes[int(t)],
            'predicted': classes[int(p)],
            'base64': img_str
        })

    info['examples'] = examples

    return info
