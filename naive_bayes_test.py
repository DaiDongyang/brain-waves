import naive_bayes
import load_data
import pre_process
import numpy as np


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

if __name__ == '__main__':
    test_trains()
    # test_train_single()
