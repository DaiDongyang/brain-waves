import tensorflow as tf
import numpy as np
import tf_set
import load_data
import config
import tempfile
import post_process
import sys


def deep_nn(x):
    with tf.name_scope('reshape'):
        h0 = tf.reshape(x, [-1, config.ds[0]])

    with tf.name_scope('fc1'):
        w1 = weight_variable([config.ds[0], config.ds[1]])
        b1 = bias_variable([config.ds[1]])
        h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    with tf.name_scope('dropout1'):
        p1 = tf.placeholder(tf.float32)
        h1_drop = tf.nn.dropout(h1, p1)

    with tf.name_scope('fc2'):
        w2 = weight_variable([config.ds[1], config.ds[2]])
        b2 = bias_variable([config.ds[2]])
        h2 = tf.nn.relu(tf.matmul(h1_drop, w2) + b2)

    with tf.name_scope('dropout2'):
        p2 = tf.placeholder(tf.float32)
        h2_drop = tf.nn.dropout(h2, p2)

    with tf.name_scope('fc3'):
        w3 = weight_variable([config.ds[2], config.ds[3]])
        b3 = bias_variable([config.ds[3]])
        h3 = tf.nn.relu(tf.matmul(h2_drop, w3) + b3)

    with tf.name_scope('dropout3'):
        p3 = tf.placeholder(tf.float32)
        h3_drop = tf.nn.dropout(h3, p3)

    with tf.name_scope('fc4'):
        w4 = weight_variable([config.ds[3], config.ds[4]])
        b4 = bias_variable([config.ds[4]])
        h4 = tf.nn.sigmoid(tf.matmul(h3_drop, w4) + b4)

    return h4, p1, p2, p3


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def calc_loss(x, y_):
    y, p1, p2, p3 = deep_nn(x)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy, p1, p2, p3


def calc_entire_loss(tfset, batch_size, sess):
    x = tf.placeholder(tf.float32, [None, config.ds[0]])
    y_ = tf.placeholder(tf.float32, [None, len(config.considered_classes)])
    cross_entropy, p1, p2, p3 = calc_loss(x, y_)
    is_end = False
    loop_count = 0
    entire_loss = 0
    while not is_end:
        batch_data, batch_y, is_end = tfset.next_batch(batch_size)
        # tolerate some error
        entire_loss += cross_entropy.eval(feed_dict={
            x: batch_data,
            y_: batch_y,
            p1: 1,
            p2: 1,
            p3: 1
        }, session=sess)
        loop_count += 1
    if loop_count == 0:
        loop_count = 1
    return entire_loss/loop_count


def calc_acc(x, y_):
    y, p1, p2, p3 = deep_nn(x)
    with tf.name_scope('accuracy'):
        c_p = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        c_p = tf.cast(c_p, tf.float32)
    acc = tf.reduce_mean(c_p)
    return acc, p1, p2, p3


def calc_entire_acc(tfset, batch_size, sess):
    x = tf.placeholder(tf.float32, [None, config.ds[0]])
    y_ = tf.placeholder(tf.float32, [None, len(config.considered_classes)])
    acc, p1, p2, p3 = calc_acc(x, y_)
    is_end = False
    loop_count = 0
    entire_acc = 0
    while not is_end:
        batch_data, batch_y, is_end = tfset.next_batch(batch_size)
        entire_acc += acc.eval(feed_dict={
            x: batch_data,
            y_: batch_y,
            p1: 1,
            p2: 1,
            p3: 1
        }, session=sess)
        loop_count += 1
    if loop_count == 0:
        loop_count = 1
    return entire_acc/loop_count


def predict_result(tfset, batch_size, sess):
    x = tf.placeholder(tf.float32, [None, config.ds[0]])
    y_ = tf.placeholder(tf.float32, [None, len(config.considered_classes)])
    y, p1, p2, p3 = deep_nn(x)
    with tf.name_scope('result'):
        ground_truth = tf.argmax(y_, 1)
        p_result = tf.argmax(y, 1)
    gts = list()
    prs = list()
    is_end = False
    loop_count = 0
    while not is_end:
        batch_data, batch_y, is_end = tfset.next_batch(batch_size)
        gt = ground_truth.eval(feed_dict={
            x: batch_data,
            y_: batch_y,
            p1: 1,
            p2: 1,
            p3: 1
        }, session=sess)
        pr = p_result.eval(feed_dict={
            x: batch_data,
            y_: batch_y,
            p1: 1,
            p2: 1,
            p3: 1
        }, session=sess)
        gts += gt
        prs += pr
        loop_count += 1
    return np.array(gts), np.array(prs)


def main(_):
    # batch_size = config.batch_size
    train_data, train_l = load_data.load_data_1hot(config.train_numbers)
    test_data, test_l = load_data.load_data_1hot(config.test_numbers)
    train_set = tf_set.TFSet(train_data, train_l)
    test_set = tf_set.TFSet(test_data, test_l)

    x = tf.placeholder(tf.float32, [None, config.ds[0]])
    y_ = tf.placeholder(tf.float32, [None, len(config.considered_classes)])
    loss, p1, p2, p3 = calc_loss(x, y_)

    with tf.name_scope('GradientDescent'):
        train_step = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(loss)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        loop_num = 0
        for i in range(config.epoch_num):
            is_end = False
            while not is_end:
                batch_data, batch_y, is_end = train_set.next_batch(config.batch_size)
                train_step.run(feed_dict={
                    x: batch_data,
                    y_: batch_y,
                    p1: config.keep_probs[0],
                    p2: config.keep_probs[1],
                    p3: config.keep_probs[2]
                })
                loop_num += 1
                if loop_num % config.print_interval == 0:
                    print("loop_num: ", loop_num)
            train_loss = calc_entire_loss(train_set, config.batch_size, sess)
            train_acc = calc_entire_acc(train_set, config.batch_size, sess)
            print('epoch %d, loss: %g, acc: %g' % (i, train_loss, train_acc))
        test_acc = calc_entire_acc(test_set, config.batch_size, sess)
        print('test set, acc: %g' % test_acc)
        gts, prs = predict_result(test_set, config.batch_size, sess)
        print('Saving graph to: %s' % graph_location)
        post_process.dump_list(gts, config.gt_pickle)
        post_process.dump_list(prs, config.pr_pickle)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

