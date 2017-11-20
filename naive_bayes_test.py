import naive_bayes
import numpy as np


def test_train_single():
    A = np.array([[3, 4], [3, 4], [3, 4]])
    miu, sigma = naive_bayes.train_single(A)
    print(miu)
    print(sigma)


if __name__ == '__main__':
    test_train_single()
