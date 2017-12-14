import nn
import load_data
import config
import tf_set
from collections import Counter
import numpy as np


def test_load_1hot():
    numbers = config.train_numbers
    data, y1 = load_data.load_data_labels(numbers)
    data, y2 = load_data.load_data_1hot(numbers)
    print(Counter(y1))
    c2 = np.sum(y2, axis=0)
    print(config.considered_classes)
    print(c2)


def test_mfcc_data():
    data_raw = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    labels_raw = np.array([3, 7, 11, 15, 19, 23])
    mfcc_train = tf_set.TFSet(data_raw, labels_raw)
    print_dl(mfcc_train.shuffle_data(), mfcc_train.shuffle_labels())
    data, labels, _ = mfcc_train.next_batch(1)
    print_dl(mfcc_train.shuffle_data(), mfcc_train.shuffle_labels())
    print_dl(data, labels)

    data, labels, _ = mfcc_train.next_batch(2)
    print_dl(mfcc_train.shuffle_data(), mfcc_train.shuffle_labels())
    print_dl(data, labels)
    data, labels, _ = mfcc_train.next_batch(3)
    print_dl(mfcc_train.shuffle_data(), mfcc_train.shuffle_labels())
    print_dl(data, labels)
    data, labels, _ = mfcc_train.next_batch(4)
    print_dl(mfcc_train.shuffle_data(), mfcc_train.shuffle_labels())
    print_dl(data, labels)
    data, labels, _ = mfcc_train.next_batch(5)
    print_dl(mfcc_train.shuffle_data(), mfcc_train.shuffle_labels())
    print_dl(data, labels)
    data, labels, _ = mfcc_train.next_batch(7)
    print_dl(data, labels)


def print_dl(data, labels):
    for row, l in zip(data, labels):
        print(row, l)
    print()
    # print()


if __name__ == '__main__':
    test_mfcc_data()
    # test_load_1hot()
