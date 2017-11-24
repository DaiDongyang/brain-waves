"""
load data from mat files and txt files
by default, the mat file is in the fold 'mats/', the txt file for the label is in the fold 'labels/'
"""
import scipy.io as sio
import numpy as np
import os
import config


def load_pair(mat_file_path, label_file_path):
    """
    load data from a mat file and its corresponding file, for example,
    this function can be called like "load_pair('./mats/sleep_data_row3_1.mat','./labels/HypnogramAASM_subject1.txt')"
    :param mat_file_path: the path of mat file
    :param label_file_path: the path of label(txt) file
    :return:
        instances: a list of features (list of list, can be converted into numpy matrix)
        labels: a list of labels, corresponding to list of features
    """
    mat = sio.loadmat(mat_file_path)
    instances = list(mat['data'].reshape((-1, config.origin_dim)))
    labels = list()
    with open(label_file_path, 'r') as label_f:
        next(label_f)
        for line in label_f:
            label = int(line.strip())
            labels.append(label)
    return instances, labels


def get_file_pairs(numbers, mats_fold, labels_fold):
    """
    get related mat file and label file
    :param numbers: a list to identified which file to be considered,
    if numbers = [1, 2, 3], then the 'sleep_data_row3_1.mat', 'sleep_data_row3_2.mat', 'sleep_data_row3_3.mat' and
    'HypnogramAASM_subject1.txt', 'HypnogramAASM_subject2.txt', 'HypnogramAASM_subject3.txt' will be considered,
    and other file will be ignored
    :param mats_fold: the path of mat fold, the mat file (such as sleep_data_row3_2.mat) is in the fold
    :param labels_fold: the path of labels fold, the label file (such as HypnogramAASM_subject2.txt) is in the fold
    :return: return type is [(mat_file_path, labels_file_path)], a list of tuples
    """
    mat_files = os.listdir(mats_fold)
    label_files = os.listdir(labels_fold)
    results = []
    if len(mat_files) == 0:
        raise Exception("Please put mat files under " + config.mats_path)
    if len(label_files) == 0:
        raise Exception("Please put label files under " + config.labels_path)
    for i in numbers:
        pair = []
        for mat_file in mat_files:
            mat_key = 'row3_' + str(i)
            if mat_key in mat_file:
                pair.append(os.path.join(mats_fold, mat_file))
        for label_file in label_files:
            label_key = 'subject' + str(i)
            if label_key in label_file:
                pair.append(os.path.join(labels_fold, label_file))
        if len(pair) == 2:
            results.append(pair)
    return results


def load_numbered_data(numbers, mats_fold, labels_fold):
    """
    load data from mat fold and labels fold
    :param numbers: a list to identified which file to be considered,
    if numbers = [1, 2, 3], then the 'sleep_data_row3_1.mat', 'sleep_data_row3_2.mat', 'sleep_data_row3_3.mat' and
    'HypnogramAASM_subject1.txt', 'HypnogramAASM_subject2.txt', 'HypnogramAASM_subject3.txt' will be considered,
    and other file will be ignored
    :param mats_fold: fold path of mat files
    :param labels_fold: fold path of labels files
    :return:
        the first element to return is a m*n matrix (numpy 2-d array), which means the training data (or test data)
        has m instances, and every instance has n features
        the second element to return is a array (numpy array) with m elements, each element is corresponding to a row of
        the first element to return.
    """
    data = []
    labels = []
    pairs = get_file_pairs(numbers, mats_fold, labels_fold)
    for item in pairs:
        instances, ls = load_pair(item[0], item[1])
        data = data + instances
        labels = labels + ls
    return np.array(data), np.array(labels)


def filter_classes(data, labels, filter_cs):
    """
    filter the data and labels according to filter_cs.
    :param data: a m*n matrix, which means data has m instances and every instance has n features
    :param labels: array with m elements, corresponding to data, data[i, :]'s label is labels[i]
    :param filter_cs: a list, identify the classes to be considered, if filter_cs is [1, 2, 3], we only consider data
        with label '1', '2' or '3'. and ignore data with other kinds of labels
    :return: data and labels after filtering
    """
    if filter_cs is None:
        return data, labels
    filtered_data = []
    filtered_labels = []
    for l in filter_cs:
        check = (labels == l)
        filtered_data = filtered_data + list(data[check, :])
        filtered_labels = filtered_labels + list(labels[check])
    return np.array(filtered_data), np.array(filtered_labels)


def load_data_labels(numbers, mats_fold=config.mats_path, labels_fold=config.labels_path,
                     filter_cs=config.considered_classes):
    """
    load data from files
    :param numbers: a list to identified which file to be considered,
    if numbers = [1, 2, 3], then the 'sleep_data_row3_1.mat', 'sleep_data_row3_2.mat', 'sleep_data_row3_3.mat' and
    'HypnogramAASM_subject1.txt', 'HypnogramAASM_subject2.txt', 'HypnogramAASM_subject3.txt' will be considered,
    and other file will be ignored
    :param mats_fold: fold path of mat files
    :param labels_fold: fold path of labels(txt) files
    :param filter_cs: a list, identify the classes to be considered, if filter_cs is [1, 2, 3], we only consider data
        with label '1', '2' or '3'. and ignore data with other kinds of labels
    :return:
        data: a m*n matrix, which means data has m instances and every instance has n features
        labels: array with m elements, corresponding to data, data[i, :]'s label is labels[i]
    """
    r_data, r_labels = load_numbered_data(numbers, mats_fold, labels_fold)
    data, labels = filter_classes(r_data, r_labels, filter_cs)
    return data, labels
