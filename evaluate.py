import numpy as np
from collections import defaultdict


def result_evaluate(g_ls, r_ls, classes):
    """
    evaluate the predict result
    :param g_ls: the ground truth
    :param r_ls: the predict result
    :param classes: a list contains all the related classes
    :return: accuracy, precision(list), recall(list), F1(list), macro_precision, macro_recall, macro_F1
    """
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
    positives = np.zeros(len(classes))
    trues = np.zeros(len(classes))
    tps = np.zeros(len(classes))
    for i in range(len(classes)):
        # add 1 avoid divide by zero
        positives[i] = positives_dict[classes[i]] + 1
        trues[i] = trues_dict[classes[i]] + 1
        tps[i] = tps_dict[classes[i]] + 1
    pre = tps/positives
    rec = tps/trues
    F1 = (2 * pre * rec)/(pre + rec)
    macro_pre = np.average(pre)
    macro_rec = np.average(rec)
    macro_F1 = (2 * macro_pre * macro_rec)/(macro_pre + macro_rec)
    return acc, pre, rec, F1, macro_pre, macro_rec, macro_F1


def evaluate_acc(g_ls, r_ls):
    g_ls = g_ls.reshape(-1, 1)
    r_ls = r_ls.reshape(-1, 1)
    check = np.array(g_ls == r_ls)
    total_num = g_ls.size
    acc = np.sum(check) / total_num
    return acc