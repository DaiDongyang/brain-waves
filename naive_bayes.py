"""
Bayes Decision
"""
import numpy as np


def train_single(subset):
    """
    get parameters of class-conditional probability
    :param subset: subset of training set, all the instances in the subset have the same label(in the same class)
    :return:
        miu: an array, the ith element means the expectation of the ith feature (by Maximum likelihood estimation)
        sigma: an array, the ith element means the standard deviation of the ith feature (Maximum likelihood estimation)
    """
    n, _ = subset.shape
    miu = np.average(subset, axis=0)
    diff_sq = (subset - miu) ** 2
    sigma = np.sqrt(np.sum(diff_sq, axis=0) / n)
    return miu, sigma


def trains(samples, labels, c_list):
    """
    train training set
    :param samples: training set
    :param labels: labels corresponding to training set
    :param c_list: all kinds of classes, if c_list is [1, 2, 3], we have three classes, 1、2 and 3
    :return:
        np.array(miu_m): a matrix, the ith row is the expectations of class c_list[i]'s class-conditional probability
        np.array(sigma_m): a matrix, the ith row is the standard deviation of
            class c_list[i]'s class-conditional probability
        c_probs: Priori probability, have the same size with c_list
    """
    c_probs = list()
    miu_m = list()
    sigma_m = list()
    rows, _ = samples.shape
    for c in c_list:
        idx = labels == c
        c_prob = np.sum(idx) / rows
        subset = samples[idx, :]
        miu, sigma = train_single(subset)
        c_probs.append(c_prob)
        miu_m.append(miu)
        sigma_m.append(sigma)
    return np.array(miu_m), np.array(sigma_m), c_probs


def naive_bayes_predict(samples, c_list, miu_m, sigma_m, c_probs):
    """
    predict the labels of test set
    :param samples: test set, m * n matrix which means m instances and every instance has n features
    :param c_list: all the classes which is arranged in a list
    :param miu_m: k * n matrix, k = len(c_list).
        miu_m[i][j] means the jth feature's expectation of c_list[i]'s class-conditional probability
    :param sigma_m: k * n matrix, k = len(c_list).
        sigma_m[i][j] means the jth feature's standard deviation of c_list[i]'s class-conditional probability
    :param c_probs: Priori probability, have the same size with c_list
    :return: an array with m elements(results), the predict results of test set. results[i] is the predict value
                of samples[i,:]
    """
    p_lists = list()
    rows, _ = samples.shape
    for i in range(len(c_list)):
        exp = np.sum((samples - miu_m[i]) ** 2 / (2 * sigma_m[i] ** 2), axis=1)
        sum_log_sigma = np.sum(np.log(sigma_m[i]))
        p = np.log(c_probs[i]) - sum_log_sigma - exp
        p_lists.append(list(p))
    p_matrix = np.array(p_lists)
    idx_r = np.argmax(p_matrix, axis=0)
    c_matrix = np.array(c_list).repeat(rows).reshape(len(c_list), -1).transpose()
    results = c_matrix[range(rows), idx_r]
    return results
