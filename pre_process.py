"""
Preprocessing data before classification
"""
import scipy.io as sio
import time
import numpy as np


def get_k_freqs(set, fs, k):
    """
    get k main frequencies with the top kth maximum amplitude by fft transform
    :param set: data to process, a m*n matrix meaning m instances and every instance has n features
    :param fs: Sampling Rate
    :param k: the number of main frequencies to return
    :return:
        k_freqs: the k main frequencies with the top kth maximum amplitude
        the last two element to return is the entire information of Spectrum
    """
    rows, N = set.shape
    f = fs * np.arange(1, N+1)/N
    S_double = np.fft.fft(set, n=N, axis=1)
    S_single_abs = np.abs(S_double[:, 0: int(N/2)])
    S_single_argpart = np.argsort(S_single_abs, axis=1)
    k_max_argpart = S_single_argpart[:, : -1*k-1: -1]
    idx = (np.arange(rows)*N).reshape(-1, 1) + k_max_argpart
    idx_flat = idx.flat
    f_matrix_flat = np.array(list(f)*rows)
    k_freqs_flat = f_matrix_flat[idx_flat]
    k_freqs = k_freqs_flat.reshape(-1, k)
    return k_freqs, S_single_abs, f[0:int(N/2)]


def pca_limit_d(train_set, test_set, d):
    """
    reduce dimension with specifying target dimension by PCA
    :param train_set: training set, m*n matrix which means m training instances and every instance has n old features
    :param test_set: test set, k*n matrix which means k test instances and every instance has n old features
    :param d: the target dimension to reduce
    :return:
        trains: new training set, m*d matrix
        tests: new test set, k*d matrix
        loss: the loss rate of the sum of eigenvalues
    """
    _, old_d = train_set.shape
    if d >= old_d:
        return train_set, test_set, 0
    mu = np.average(train_set, axis=0)
    avg_trains = train_set - mu
    cov = np.cov(avg_trains.T)
    c_roots, Q = np.linalg.eig(cov)
    c_roots = c_roots.real
    Q = Q.real
    idxs = np.argpartition(c_roots, d * (-1))[d*(-1):]
    M = Q[:, idxs]
    loss = 1 - np.sum(c_roots[idxs])/np.sum(c_roots)
    trains = avg_trains.dot(M)
    tests = np.dot((test_set - mu), M)
    return trains, tests, loss


def pca_limit_loss(train_set, test_set, loss):
    """
    reduce dimension with specifying the loss rate of the sum of eigenvalues by PCA
    :param train_set: training set, m*n matrix which means m training instances and every instance has n old features
    :param test_set: test set, k*n matrix which means k test instances and every instance has n old features
    :param loss: the loss rate of the sum of eigenvalues
    :return:
        trains: new training set, m*d matrix
        tests: new test set, k*d matrix
        loss: the target dimension to reduce
    """
    mu = np.average(train_set, axis=0)
    avg_trains = train_set - mu
    cov = np.cov(avg_trains.T)
    c_roots, Q = np.linalg.eig(cov)
    c_roots = c_roots.real
    Q = Q.real
    idxs = np.argsort(c_roots)[::-1]
    threshold = (1 - loss)*(np.sum(c_roots))
    sum = 0
    i = 0
    for c_root in c_roots:
        i += 1
        sum += c_root
        if sum >= threshold:
            break
    M = Q[:, idxs[:i+1]]
    d = i
    trains = avg_trains.dot(M)
    tests = np.dot((test_set - mu), M)
    return trains, tests, d
