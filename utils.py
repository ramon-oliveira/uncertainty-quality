import numpy as np

def split(X_train, y_train, X_test, y_test):
    classes = np.unique(y_train)
    np.random.shuffle(classes)

    # min 2 classes in train
    # min 1 class for unknown
    # min 2 classes out train and different of unknown
    assert len(classes) >= 5

    begin = 0
    end = (len(classes) - 1)//2
    in_train = classes[begin:end]
    begin = end
    end = end*2
    out_train = classes[begin:end]
    unknown = classes[end:]

    X_train = X_train[np.in1d(y_train, in_train)]
    y_train = y_train[np.in1d(y_train, in_train)]

    y_train = y_train[np.in1d(y_train, in_train)]

    X_test = X_test[~np.in1d(y_test, unknown)]
    y_test = y_test[~np.in1d(y_test, unknown)]
