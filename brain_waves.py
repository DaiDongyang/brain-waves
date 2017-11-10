import scipy.io as sio
import numpy as np

# f = 200HZ 傅立叶变换？
# 1000 维的数据经过傅立叶变幻得到一个新的纬度，然后可以把1000维降到5维，这样我们只考虑6（5+1，1是傅立叶变幻得到的纬度）
# todo: fft


def pca_limit_d(train_set, test_set, d):
    _, old_d = train_set.shape
    if d >= old_d:
        return train_set, test_set, 0
    miu = np.average(train_set, axis=0)
    avg_trains = train_set - miu
    cov = np.cov(avg_trains.T)
    c_roots, Q = np.linalg.eig(cov)
    c_roots = c_roots.real
    Q = Q.real
    idxs = np.argpartition(c_roots, d * (-1))[d*(-1):]
    M = Q[:, idxs]
    loss = 1 - np.sum(c_roots[idxs])/np.sum(c_roots)
    trains = avg_trains.dot(M)
    tests = np.dot((test_set - miu), M)
    return trains, tests, loss


def pca_limit_loss(train_set, test_set, loss):
    miu = np.average(train_set, axis=0)
    avg_trains = train_set - miu
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
    tests = np.dot((test_set - miu), M)
    return trains, tests, d
