import numpy as np
import xgboost as xgb
from scipy.stats import entropy
import base64
from PIL import Image
import io
from sklearn import metrics


def uncertainty_std_argmax(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)
    y_std = np.array([y_probabilistic[:, i, c].std()
                      for i, c in enumerate(y_pred)])

    return y_pred, y_std


def uncertainty_mean_argmax(y_probabilistic):
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)
    y_mean = np.array([y_probabilistic[:, i, c].mean()
                      for i, c in enumerate(y_pred)])

    return y_pred, y_mean


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


def uncertainty_classifer(model, dataset, test_uncertainty):
    x_val, y_val = dataset.x_val, dataset.y_val.argmax(axis=1)

    y_deterministic = model.predict(x_val, probabilistic=False)
    y_probabilistic = model.predict(x_val, probabilistic=True)

    uncertainties = [
        ('uncertainty_std_argmax', uncertainty_std_argmax, y_probabilistic),
        ('uncertainty_mean_argmax', uncertainty_mean_argmax, y_probabilistic),
        ('uncertainty_std_mean', uncertainty_std_mean, y_probabilistic),
        ('uncertainty_entropy', uncertainty_entropy, y_deterministic),
        ('uncertainty_entropy_mean', uncertainty_entropy, y_probabilistic.mean(axis=0)),
        ('uncertainty_mean_entropy', uncertainty_mean_entropy, y_probabilistic),
    ]

    x = np.zeros([len(y_val), len(uncertainties) + dataset.output_size])
    x[:, len(uncertainties):] = y_probabilistic.mean(axis=0)
    y = (y_val != y_probabilistic.mean(axis=0).argmax(axis=1))
    for i, (name, func, y_score) in enumerate(uncertainties):
        _, uncertainty = func(y_score)
        x[:, i] = uncertainty

    clf = xgb.XGBClassifier().fit(x, y)
    return clf.predict_proba(test_uncertainty)[:, 1]


def evaluate(model, dataset):
    info = {}
    x_test, y_test = dataset.x_test, dataset.y_test.argmax(axis=1)

    y_deterministic = model.predict(x_test, probabilistic=False)
    y_probabilistic = model.predict(x_test, probabilistic=True)
    y_pred = y_probabilistic.mean(axis=0).argmax(axis=1)

    acc_test = (y_test == y_pred).mean()
    info['accuracy_test'] = acc_test
    print('test accuracy: {0:.4f}'.format(acc_test))

    uncertainties = [
        ('uncertainty_std_argmax', uncertainty_std_argmax, y_probabilistic),
        ('uncertainty_mean_argmax', uncertainty_mean_argmax, y_probabilistic),
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

    brier_pred = metrics.brier_score_loss(y_true=(y_test == y_pred), y_prob=y_probabilistic.mean(axis=0).max(axis=1))
    brier_unc = metrics.brier_score_loss(y_true=(y_test != y_pred), y_prob=uncertainty)
    info['brier_prediction'] = brier_pred
    info['brier_uncertainty'] = brier_unc
    print('brier_prediction', brier_pred)
    print('brier_uncertainty', brier_unc)

    success = (y_test == y_pred)
    proba = y_probabilistic.mean(axis=0).max(axis=1)
    auc_hendricks_softmax = metrics.roc_auc_score(y_true=success, y_score=proba)
    auc_hendricks_uncertainty = metrics.roc_auc_score(y_true=success, y_score=1-uncertainty)
    info['auc_hendricks_softmax'] = auc_hendricks_softmax
    info['auc_hendricks_uncertainty'] = auc_hendricks_uncertainty
    print('auc_hendricks_softmax', auc_hendricks_softmax)
    print('auc_hendricks_uncertainty', auc_hendricks_uncertainty)

    precision, recall, _ = metrics.precision_recall_curve(y_true=success, probas_pred=proba)
    aupr = metrics.auc(recall, precision)
    info['aupr_hendricks_softmax_success'] = aupr
    print('aupr_hendricks_softmax_success', aupr)

    precision, recall, _ = metrics.precision_recall_curve(y_true=success, probas_pred=1-uncertainty)
    aupr = metrics.auc(recall, precision)
    info['aupr_hendricks_uncertainty_success'] = aupr
    print('aupr_hendricks_uncertainty_success', aupr)

    precision, recall, _ = metrics.precision_recall_curve(y_true=1-success, probas_pred=proba*-1)
    aupr = metrics.auc(recall, precision)
    info['aupr_hendricks_softmax_fail'] = aupr
    print('aupr_hendricks_softmax_fail', aupr)

    precision, recall, _ = metrics.precision_recall_curve(y_true=1-success, probas_pred=(1-uncertainty)*-1)
    aupr = metrics.auc(recall, precision)
    info['aupr_hendricks_uncertainty_fail'] = aupr
    print('aupr_hendricks_uncertainty_fail', aupr)


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
