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
