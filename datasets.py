from sklearn import datasets


def load_digits(test_size=0.2):
    X, y = datasets.load_digits(return_X_y=True)
    data = train_test_split(X, y, test_size=test_size)
    X_train, X_test, y_train, y_test = data
    return X_train, y_train, X_test, y_test


def load(dataset):
    if dataset == 'digits':
        X_train, y_train, X_test, y_test = load_digits()
    else:
        raise Exception('Unknown dataset {0}'.format(dataset))

    return X_train, y_train, X_test, y_test
