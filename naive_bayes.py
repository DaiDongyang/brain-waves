from collections import defaultdict
import load_data
import pre_process
import numpy as np

# [c1, c2, c3] for example
class_list = load_data.filter_l_number

# an m * n matrix, n is the dimension of the data
# m is the kinds of labels, if class_list is [c1, c2, c3],
# first row is the miu on the condition of c1
miu_matrix = None

# an m * n matrix, n is the dimension of the data
# m is the kinds of labels, if class_list is [c1, c2, c3],
# first row is the sigma on the condition of c1
sigma_matrix = None


# every row of the subset is an instance
def train_single(subset):
    n, _ = subset.shape
    miu = np.average(subset, axis=0)
    diff_sq = (subset - miu)**2
    sigma = np.sqrt(np.sum(diff_sq, axis=0)/n)
    return miu, sigma


# todo: complete test
def trains(samples, labels, c_list):
    miu_m = list()
    sigma_m = list()
    for c in c_list:
        idx = labels == c
        subset = samples[idx, :]
        miu, sigma = train_single(subset)
        miu_m.append(miu)
        sigma_m.append(sigma)
    return np.array(miu_m), np.array(sigma_m)


# todo: test
def naive_bayes_predict(samples, c_list, miu_m, sigma_m, c_probs):
    p_lists = list()
    rows, _ = samples.shape
    for i in range(len(c_list)):
        # ln(p)
        # todo: forget sigma in the front of exp
        p = np.sum((samples - miu_m[i])**2/(2 * sigma_m[i]**2), axis=1) * np.log(1/c_probs[i])
        p_lists.append(p)
    p_matrix = np.array(p_lists)
    idx_r = np.argmax(p_matrix, axis=1)
    idx = zip(range(rows), idx_r)
    c_matrix = np.array(c_list).repeat(rows).reshape(len(c_list), -1).transpose()
    results = c_matrix(idx)
    return results


