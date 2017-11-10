import brain_waves
import numpy as np
import load_data

def test_pca_limit_d():
    r_trains, r_train_ls = load_data.load_data_labels(load_data.train_numbers)
    r_tests, r_test_ls = load_data.load_data_labels(load_data.test_numbers)
    d = 50
    trains, tests, loss = brain_waves.pca_limit_d(r_trains, r_tests, d)
    print(trains.shape)
    print(tests.shape)
    print(loss)


def test_pca_limit_loss():
    loss = 0.1
    r_trains, r_train_ls = load_data.load_data_labels(load_data.train_numbers)
    r_tests, r_test_ls = load_data.load_data_labels(load_data.test_numbers)
    # d = 5
    trains, tests, d = brain_waves.pca_limit_loss(r_trains, r_tests, loss)
    print(trains.shape)
    print(tests.shape)

if __name__ == '__main__':
    # test_pca_limit_loss()
    test_pca_limit_d()