import pickle
import numpy as np
import config
from collections import Counter


def dump_list(l, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(l, pickle_f)


def load_pickle(pickle_name):
    r = None
    with open(pickle_name, 'rb') as pickle_f:
        r = pickle.load(pickle_f)
    return r


def process_results(gt, pr, classes):
    LEN = len(classes)
    matrix = np.zeros((LEN, LEN))
    # print(Counter(gt))
    # print(Counter(pr))
    # print(Counter(zip(gt, pr)))
    for i, j in zip(gt, pr):
        matrix[i, j] += 1
    # print(matrix)
    return matrix, classes


def print_csv():
    gt = load_pickle('./pickles/gt.pickle')
    pr = load_pickle('./pickles/pr.pickle')
    matrix, classes = process_results(gt, pr, config.classes)
    print('\\', end='\t')
    for c in classes:
        print(c, end='\t')
    print()
    for i in range(len(classes)):
        print(classes[i], end='\t')
        for ele in matrix[i]:
            print(ele, end='\t')
        print()
    print()


if __name__ == '__main__':
    print_csv()
