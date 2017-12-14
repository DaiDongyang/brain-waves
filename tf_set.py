import numpy as np


class TFSet:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        num = len(data)
        if num != len(labels):
            raise NameError('size not equal for data and labels')
        self.data_size = num
        idx = np.array(range(num))
        np.random.shuffle(idx)
        self.idx = idx
        self.cur = 0

    def next_batch(self, batch_size):
        is_epoch_end = False
        if self.cur == 0:
            np.random.shuffle(self.idx)
        end = self.cur + batch_size
        if end >= self.data_size:
            end = self.data_size
            is_epoch_end = True
            iidx = np.array([i for i in range(self.cur, end)])
            self.cur = 0
        else:
            iidx = np.array([i for i in range(self.cur, end)])
            self.cur += batch_size
        batch_idx = self.idx[iidx]
        return self.data[batch_idx], self.labels[batch_idx], is_epoch_end

    def shuffle_data(self):
        return self.data[self.idx]

    def shuffle_labels(self):
        return self.labels[self.idx]



