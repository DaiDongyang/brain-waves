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
    c_probs = list()
    miu_m = list()
    sigma_m = list()
    rows, _ = samples.shape
    for c in c_list:
        idx = labels == c
        c_prob = np.sum(idx)/rows
        subset = samples[idx, :]
        miu, sigma = train_single(subset)
        c_probs.append(c_prob)
        miu_m.append(miu)
        sigma_m.append(sigma)
    return np.array(miu_m), np.array(sigma_m), c_probs


# todo: test
def naive_bayes_predict(samples, c_list, miu_m, sigma_m, c_probs):
    p_lists = list()
    rows, _ = samples.shape
    for i in range(len(c_list)):
        exp = np.sum((samples - miu_m[i])**2/(2 * sigma_m[i]**2), axis=1)
        sum_log_sigma = np.sum(np.log(sigma_m[i]))
        p = np.log(c_probs[i]) - sum_log_sigma - exp
        # print(p.shape)
        p_lists.append(list(p))
    p_matrix = np.array(p_lists)
    idx_r = np.argmax(p_matrix, axis=0)
    c_matrix = np.array(c_list).repeat(rows).reshape(len(c_list), -1).transpose()
    # print(c_matrix.shape)
    # print(c_matrix[range(rows), idx_r])
    results = c_matrix[range(rows), idx_r]
    return results


