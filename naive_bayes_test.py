import naive_bayes
import load_data
import pre_process
import numpy as np
import time


def test_train_single():
    A = np.array([[3, 4], [3, 4], [3, 4]])
    miu, sigma = naive_bayes.train_single(A)
    print(miu)
    print(sigma)


def test_trains():
    train_r_data, train_labels = load_data.load_data_labels(load_data.train_numbers)
    test_r_data, test_labels = load_data.load_data_labels(load_data.test_numbers)
    d = 5
    train_data, test_data, loss = pre_process.pca_limit_d(train_r_data, test_r_data, d)
    miu_m, sigma_m = naive_bayes.trains(train_data, train_labels, load_data.filter_l_number)
    print(miu_m)
    print(sigma_m)


def test_naiyes():
    k_frq = 5
    d = 10
    fs = 200
    c_list = load_data.filter_l_number
    train_r_data, train_labels = load_data.load_data_labels(load_data.train_numbers)
    test_r_data, test_labels = load_data.load_data_labels(load_data.test_numbers)
    train_frqs, _, _ = pre_process.get_k_freqs(train_r_data, fs, k_frq)
    test_frqs, _, _, = pre_process.get_k_freqs(test_r_data, fs, k_frq)

    train_data1, test_data1, loss = pre_process.pca_limit_d(train_r_data, test_r_data, d)
    train_data = np.hstack((train_data1, train_frqs))
    test_data = np.hstack((test_data1, test_frqs))
    miu_m, sigma_m, c_probs = naive_bayes.trains(train_data, train_labels, c_list)
    results = naive_bayes.naive_bayes_predict(test_data, c_list, miu_m, sigma_m, c_probs)
    # for result in results:
    #     print(result)
    check = results == test_labels
    acc = np.sum(check)/len(check)
    print(acc)


if __name__ == '__main__':
    test_naiyes()
    # test_trains()
    # test_train_single()
