import pre_process
import numpy as np
import load_data
import scipy.io as sio
import matplotlib.pyplot as plt


def test_pca_limit_d():
    r_trains, r_train_ls = load_data.load_data_labels(load_data.train_numbers)
    r_tests, r_test_ls = load_data.load_data_labels(load_data.test_numbers)
    d = 50
    trains, tests, loss = pre_process.pca_limit_d(r_trains, r_tests, d)
    print(trains.shape)
    print(tests.shape)
    print(loss)


def test_pca_limit_loss():
    loss = 0.1
    r_trains, r_train_ls = load_data.load_data_labels(load_data.train_numbers)
    r_tests, r_test_ls = load_data.load_data_labels(load_data.test_numbers)
    # d = 5
    trains, tests, d = pre_process.pca_limit_loss(r_trains, r_tests, loss)
    print(trains.shape)
    print(tests.shape)


def test_get_k_freqs():
    a = sio.loadmat('./mats/sleep_data_row3_1.mat')
    data_raw = a['data']
    data = data_raw.reshape((-1, 1000))
    fs = 200
    k = 3
    i = 4000
    k_freqs, S_single, f = pre_process.get_k_freqs(data, fs, k)
    print(k_freqs[i])
    plt.plot(f[0:20], S_single[i, 0:20])
    plt.show()


if __name__ == '__main__':
    # test_pca_limit_loss()
    # test_pca_limit_d()
    test_get_k_freqs()