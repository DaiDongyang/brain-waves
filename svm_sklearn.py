import load_data
import config
import evaluate
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time
import numpy as np
from collections import defaultdict
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as pyplot


# def lda_reduce()

def extract_trains(train_set, labels, step, start):
    """
    to reduce the training set size by extract a sample every step samples
    :param train_set: training set
    :param labels: labels of training set
    :param step: every step samples extract one
    :param start: the index of the first element to extract
    :return: new training set after extracting
    """
    new_train_data = train_set[start::step]
    new_labels = labels[start::step]
    return new_train_data, new_labels


def grid_run_svm(train_set, train_ls, test_set, test_ls, kernel, Cs, gammas):
    rs = []
    for C in Cs:
        for gamma in gammas:
            clf = SVC(C=C, kernel=kernel, gamma=gamma)
            clf.fit(train_set, train_ls)
            results = clf.predict(test_set)
            acc = evaluate.evaluate_acc(test_ls, results)
            print(acc, C, gamma)
            rs.append((acc, C, gamma))
    return sorted(rs, key=lambda item: item[0], reverse=True)


if __name__ == '__main__':
    C_range = np.logspace(-2, 2, 10)
    gamma_range = np.logspace(-10, 10, 20)
    kernels = ['rbf', 'poly', 'sigmoid']
    extract_start = 0
    extract_step = 200
    t1 = time.time()
    pca = PCA(n_components=10)
    train_r_data, train_labels = load_data.load_data_labels(config.train_numbers)
    test_r_data, test_labels = load_data.load_data_labels(config.test_numbers)
    pca.fit(train_r_data, train_labels)
    train_data, train_labels = extract_trains(train_r_data, train_labels, extract_step,
                                              extract_start)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_r_data)
    results_dict = defaultdict(lambda _: None)
    for kernel_ in kernels:
        results_ = grid_run_svm(train_data, train_labels, test_data, test_labels, kernel_, C_range, gamma_range)
        results_dict[kernel_] = results_

    with open('results_dict.pickle', 'wb') as pickle_file:
        pickle.dump(list(results_dict.items()), pickle_file)

    with open('result.txt', 'w') as outf:
        for kernel_, results_ in results_dict.items():
            print(kernel_)
            for result_ in results_:
                print(result_)
            print()

