import numpy as np


def create_big_dataset():
    train_X, test_X = np.load(
        'big_dataset/train_airlinescodrnaadult_fold0.npy'), np.load(
        'big_dataset/test_airlinescodrnaadult_fold0.npy')
    train_y, test_y = np.load(
        'big_dataset/trainy_airlinescodrnaadult_fold0.npy'), np.load(
        'big_dataset/testy_airlinescodrnaadult_fold0.npy')
    dataset_dict = dict(train_data=(train_X, train_y),
                        test_data=(test_X, test_y))
    return dataset_dict
