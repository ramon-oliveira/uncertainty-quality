import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split



class Dataset(object):
    pass


class Digits(Dataset):

    def __init__(self, test_size, *args, **kwargs):
        super(Digits, self).__init__(*args, **kwargs)

        X, y = datasets.load_digits(return_X_y=True)
        data = train_test_split(X, y, test_size=test_size)
        self.X_train = data[0]
        self.y_train = data[2]
        self.X_test = data[1]
        self.y_test = data[3]

        self.classes = np.array(list(set(y)))
        np.random.shuffle(self.classes)
        self.in_train_classes = self.classes[:4]
        self.out_train_classes = self.classes[4:8]
        self.unk_train_classes = self.classes[8:]

    def train(self, with_unknown=False):
        X = self.X_train[np.in1d(self.y_train, self.in_train_classes)]
        y = self.y_train[np.in1d(self.y_train, self.in_train_classes)]

        if with_unknown:
            X2 = self.X_train[np.in1d(self.y_train, self.unk_train_classes)]
            y2 = self.y_train[np.in1d(self.y_train, self.unk_train_classes)]

            X = np.concatenate((X, X2), axis=0)
            y = np.concatenate((y, X2), axis=0)

        return X, y

    def test(self):
        X = self.X_test[~np.in1d(self.y_test, self.unk_train_classes)]
        y = self.y_test[~np.in1d(self.y_test, self.unk_train_classes)]

        return X, y


def load(dataset):
    name = dataset.pop('name')

    if name == 'digits':
        dataset = Digits(**dataset)
    else:
        raise Exception('Unknown dataset {0}'.format(dataset))

    return dataset
