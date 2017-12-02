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


def grid_run_and_save():
    C_range = np.logspace(-2, 2, 10)
    gamma_range = np.logspace(-10, 5, 15)
    kernels = ['rbf', 'poly', 'sigmoid']
    extract_start = 1  # random set, not special require
    extract_step = 200
    t1 = time.time()
    pca = PCA(n_components=config.d)
    train_r_data, train_labels = load_data.load_data_labels(config.train_numbers)
    test_r_data, test_labels = load_data.load_data_labels(config.test_numbers)
    pca.fit(train_r_data, train_labels)
    train_data, train_labels = extract_trains(train_r_data, train_labels, extract_step,
                                              extract_start)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_r_data)
    results_dict = defaultdict(lambda _: None)
    best_combine = (0, None, None, None)
    for kernel_ in kernels:
        results_ = grid_run_svm(train_data, train_labels, test_data, test_labels, kernel_, C_range, gamma_range)
        results_dict[kernel_] = results_
        if results_[0][0] > best_combine[0]:
            best_combine = (results_[0][0], kernel_, results_[0][1], results_[0][2])

    pickle_name = 'acc_params_dict_step' + str(extract_step) + '.pickle'
    with open(pickle_name, 'wb') as pickle_file:
        pickle.dump(list(results_dict.items()), pickle_file)

    acc_params_f = 'acc_params_step' + str(extract_step) + '.txt'
    with open(acc_params_f, 'w') as outf:
        for kernel_, results_ in results_dict.items():
            print(kernel_)
            print(kernel_, file=outf)
            for result_ in results_:
                print(result_)
                print(result_, file=outf)
            print()
            print(file=outf)
        t2 = time.time()
        print('run time(grid):', t2 - t1)
        print('run time(grid):', t2 - t1, file=outf)
    return best_combine, results_dict


def svm_run_evaluate(kernel, C, gamma):
    if (kernel is None) or (C is None) or (gamma is None):
        return
    t1 = time.time()
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    pca = PCA(n_components=config.d)
    train_r_data, train_labels = load_data.load_data_labels(config.train_numbers)
    test_r_data, test_labels = load_data.load_data_labels(config.test_numbers)

    pca.fit(train_r_data, train_labels)
    train_data = pca.transform(train_r_data)
    test_data = pca.transform(test_r_data)

    clf.fit(train_data, train_labels)
    results = clf.predict(test_data)
    t2 = time.time()
    print('single svm(all instances) total time:', t2 - t1)
    gr_predicts = list()
    gr_predicts.append(test_labels)
    gr_predicts.append(train_labels)
    with open('gr_predicts.pickle', 'rb') as gr_predicts_f:
        pickle.dump(gr_predicts, gr_predicts_f)

    predict_filename = 'predict_results_' + str(kernel) + '_' + str(C) + '_' + str(gamma) + '.txt'
    with open(predict_filename, 'w') as predict_f:
        print('predict result:', file=predict_f)
        for result in results:
            print(result, file=predict_f)

    ground_truth_filename = 'ground_truth.txt'
    with open(ground_truth_filename, 'w') as gr_f:
        print('ground truth:', file=gr_f)
        for label in test_labels:
            print(label, file=gr_f)

    acc, pre, rec, F1, macro_pre, macro_rec, macro_F1 = evaluate.result_evaluate(test_labels, results,
                                                                                 config.considered_classes)

    print('Accuracy is', acc)
    print('precision of', config.considered_classes, 'is:', pre)
    print('recall of', config.considered_classes, 'is:', rec)
    print('F1 of', config.considered_classes, 'is:', F1)
    print('Macro-precision is:', macro_pre)
    print('Macro-recall is:', macro_rec)
    print('Macro-F1 is:', macro_F1)


if __name__ == '__main__':
    best_combine_, _ = grid_run_and_save()
    # acc, kernel, C, gamma = best_combine
    # best_combine_ = (0.77497900923593621, 'rbf', 1.6681005372000592, 2.2758459260747865e-05)
    svm_run_evaluate(best_combine_[1], best_combine_[2], best_combine_[3])

