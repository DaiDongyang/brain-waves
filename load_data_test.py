import load_data
import matplotlib.pyplot as plt
import numpy as np


def test_load_pair():
    mat_file = './mats/sleep_data_row3_1.mat'
    label_file = './labels/HypnogramAASM_subject1.txt'
    instances, labels = load_data.load_pair(mat_file, label_file)
    instance = instances[0]
    plt.plot(list(range(len(instance))), instance)
    plt.show()
    # plt.plot(list(range(instances.size())), instances.flat)
    # print(len(instances))
    # print(len(labels))
    # for i in range(1):
    #     print(instances[i], end='\t')
    #     print(labels[i])
        # print(type(instances[i]))
        # print(type(labels[i]))
        # print(tmp)


def test_get_file_pairs():
    results = load_data.get_file_pairs(load_data.train_numbers, load_data.mats_path, load_data.labels_path)
    for result in results:
        print(result)


def test_load_numbered_data():
    data, label = load_data.load_numbered_data(load_data.test_numbers, load_data.mats_path, load_data.labels_path)
    print(data.shape)
    print(label.shape)


def test_filter_data_labels():
    r_datas, r_labels = load_data.load_numbered_data(load_data.test_numbers, load_data.mats_path, load_data.labels_path)
    datas, rs = load_data.considered_classes(r_datas, r_labels, load_data.filter_l_number)
    print(datas.shape)
    print(rs.shape)
    print(np.max(rs))
    print(np.min(rs))


def test_load_data_labels():
    r_data, r_labels = load_data.load_data_labels(load_data.train_numbers)
    print(r_data.shape)
    print(r_labels.shape)


if __name__ == '__main__':
    # test_load_data_labels()
    # test_filter_data_labels()
    # test_load_numbered_data()
    # test_get_file_pairs()
    # test_load_pair()
    pass