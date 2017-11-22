import config
import naive_bayes
import load_data
import pre_process
import numpy as np
import time
from collections import defaultdict


def result_evaluate(g_ls, r_ls, c_list):
    g_ls = g_ls.reshape(-1, 1)
    r_ls = r_ls.reshape(-1, 1)
    check = np.array(g_ls == r_ls)
    total_num = g_ls.size
    acc = np.sum(check) / total_num

    g_ls = list(g_ls.flat)
    r_ls = list(r_ls.flat)
    tps_dict = defaultdict(lambda: 0)
    positives_dict = defaultdict(lambda: 0)
    trues_dict = defaultdict(lambda: 0)
    for i in range(total_num):
        positives_dict[r_ls[i]] += 1
        trues_dict[g_ls[i]] += 1
        if r_ls[i] == g_ls[i]:
            tps_dict[r_ls[i]] += 1
    positives = np.zeros(len(c_list))
    trues = np.zeros(len(c_list))
    tps = np.zeros(len(c_list))
    for i in range(len(c_list)):
        positives[i] = positives_dict[c_list[i]]
        trues[i] = trues_dict[c_list[i]]
        tps[i] = tps_dict[c_list[i]]
    pre = tps/positives
    rec = tps/trues
    F1 = (2 * pre * rec)/(pre + rec)
    macro_pre = np.average(pre)
    macro_rec = np.average(rec)
    macro_F1 = (2 * macro_pre * macro_rec)/(macro_pre + macro_rec)
    return acc, pre, rec, F1, macro_pre, macro_rec, macro_F1


if __name__ == '__main__':
    tests_filter_ls = None
    if config.is_filter_test:
        tests_filter_ls = config.considered_classes
    t1 = time.time()
    c_list = config.considered_classes
    train_r_data, train_labels = load_data.load_data_labels(config.train_numbers)
    test_r_data, test_labels = load_data.load_data_labels(config.test_numbers, filter_ls=tests_filter_ls)
    t2 = time.time()
    print('Load data take time: ', t2 - t1, 's')

    t3 = time.time()
    train_freqs, _, _ = pre_process.get_k_freqs(train_r_data, config.fs, config.freq_k)
    test_freqs, _, _ = pre_process.get_k_freqs(test_r_data, config.fs, config.freq_k)
    t4 = time.time()
    print('FFT take time: ', t4 - t3, 's')

    t5 = time.time()
    train_data1, test_data1, loss = pre_process.pca_limit_d(train_r_data, test_r_data, config.d)
    t6 = time.time()
    print('PCA reduce dim take time: ', t6 - t5, 's')

    train_data = np.hstack((train_data1, train_freqs))
    test_data = np.hstack((test_data1, test_freqs))

    t7 = time.time()
    miu_m, sigma_m, c_probs = naive_bayes.trains(train_data, train_labels, c_list)
    t8 = time.time()
    print('Train time: ', t8 - t7, 's')

    t9 = time.time()
    results = naive_bayes.naive_bayes_predict(test_data, c_list, miu_m, sigma_m, c_probs)
    t10 = time.time()
    print('Test time: ', t10 - t9, 's')

    with open(config.result_file, 'w') as outf:
        print('predict result:', file=outf)
        for result in results:
            print(result, file=outf)

    with open(config.gt_file, 'w') as outf:
        print('ground truth:', file=outf)
        for test_label in test_labels:
            print(test_label, file=outf)

    acc, pre, rec, F1, macro_pre, macro_rec, macro_F1 = result_evaluate(test_labels, results, config.considered_classes)
    t11 = time.time()
    print('Total time: ', t11 - t1, 's')
    print('Accuracy is', acc)
    print('precision of', config.considered_classes, 'is:', pre)
    print('recall of', config.considered_classes, 'is:', rec)
    print('F1 of', config.considered_classes, 'is:', F1)
    print('Macro-precision is:', macro_pre)
    print('Macro-recall is:', macro_rec)
    print('Macro-F1 is:', macro_F1)
