import scipy.io as sio
import numpy as np
import os
import config


def load_pair(mat_file_path, label_file_path):
    mat = sio.loadmat(mat_file_path)
    # data = mat['data'].flat
    instances = list(mat['data'].reshape((-1, config.origin_dim)))
    labels = list()
    with open(label_file_path, 'r') as label_f:
        next(label_f)
        # i = 0
        for line in label_f:
            # instance = list(data[raw_dim*i:raw_dim * (i + 1)])
            label = int(line.strip())
            # instances.append(instance)
            labels.append(label)
            # i += 1
    # print(data[i*1000-1])
    return instances, labels


def get_file_pairs(numbers, mats_fold, labels_fold):
    mat_files = os.listdir(mats_fold)
    label_files = os.listdir(labels_fold)
    results = []
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
    data = []
    labels = []
    pairs = get_file_pairs(numbers, mats_fold, labels_fold)
    for item in pairs:
        instances, ls = load_pair(item[0], item[1])
        data = data + instances
        labels = labels + ls
    return np.array(data), np.array(labels)


def considered_classes(data, labels, filter_ls):
    if filter_ls is None:
        return data, labels
    filtered_data = []
    filtered_labels = []
    for l in filter_ls:
        check = (labels == l)
        filtered_data = filtered_data + list(data[check, :])
        filtered_labels = filtered_labels + list(labels[check])
    return np.array(filtered_data), np.array(filtered_labels)


def load_data_labels(numbers, mats_fold=config.mats_path, labels_fold=config.labels_path, filter_ls=config.considered_classes):
    r_data, r_labels = load_numbered_data(numbers, mats_fold, labels_fold)
    data, labels = considered_classes(r_data, r_labels, filter_ls)
    return data, labels
