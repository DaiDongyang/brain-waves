"""
configuration
"""

# the path of mat fold
mats_path = './mats/'

# the path of label fold
labels_path = './labels/'

# the original dimension of data before dimension reduction by PCA
origin_dim = 1000

# the list named 'train_numbers' is used to identify the training set
# if the train_numbers is [1, 2, 3], the 'sleep_data_row3_1.mat', 'sleep_data_row3_2.mat', 'sleep_data_row3_3.mat' and
# 'HypnogramAASM_subject1.txt', 'HypnogramAASM_subject2.txt', 'HypnogramAASM_subject3.txt' will be treat as train set
# and train set
train_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # list(range(1,20))
# train_numbers = [1, 2]

# the list named 'test_numbers' is used to identify the test set
test_numbers = [20]

# the class be considered, if considered_classes = [1, 2, 3], we will only consider data corresponds to classes 1, 2, 3
considered_classes = [1, 2, 3]

is_filter_test = True

# the k main frequency to be considered, if freq_k = 3, we consider the frequency with the top 3 highest energy,
# if freq_k = 1, we only consider the primary frequency with the highest energy
freq_k = 1

# Sampling rate
fs = 200

# The target dimension to reduce for PCA process
d = 5

# The name of file to save predict result
result_file = 'result.txt'

# The name of file to save ground truth
gt_file = 'ground_truth.txt'
