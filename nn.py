import tensorflow as tf
import numpy as np
import tf_set
import load_data
import config
import tempfile
import post_process
import sys
import os


os.environ['CUDA_VISIBLE_DEVICE'] = config.visible_device

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


def main(_):
    # batch_size = config.batch_size
    train_data, train_l = load_data.load_data_1hot(config.train_numbers)
    test_data, test_l = load_data.load_data_1hot(config.test_numbers)
    train_set = tf_set.TFSet(train_data, train_l)
    test_set = tf_set.TFSet(test_data, test_l)

    x = tf.placeholder(tf.float32, [None, config.ds[0]])
    y_ = tf.placeholder(tf.float32, [None, len(config.considered_classes)])

    y, p1, p2, p3 = deep_nn(x)

    # loss, p1, p2, p3 = calc_loss(x, y_)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('accuracy'):
        c_p = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        c_p = tf.cast(c_p, tf.float32)
    acc = tf.reduce_mean(c_p)

    with tf.name_scope('result'):
        g_t = tf.argmax(y_, 1)
        p_r = tf.argmax(y, 1)

    with tf.name_scope('GradientDescent'):
        train_step = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(cross_entropy)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
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

            if i % config.eval_train_interval == 0:
                is_end = False
                loop_count = 0
                train_loss = 0
                train_acc = 0
                while not is_end:
                    batch_data, batch_y, is_end = train_set.next_batch(config.batch_size)
                    train_loss += cross_entropy.eval(feed_dict={
                        x: batch_data,
                        y_: batch_y,
                        p1: 1,
                        p2: 1,
                        p3: 1
                    }, session=sess)
                    train_acc += acc.eval(feed_dict={
                        x: batch_data,
                        y_: batch_y,
                        p1: 1,
                        p2: 1,
                        p3: 1
                    }, session=sess)
                    loop_count += 1
                train_loss = train_loss / loop_count
                train_acc = train_acc / loop_count
                print('epoch %d, loss: %g, acc: %g' % (i, train_loss, train_acc))

            if i % config.save_checkpoint_interval == 0:
                saver.save(sess, config.checkpoint_file, global_step=loop_num)

        is_end = False
        gts = list()
        prs = list()
        loop_count = 0
        test_acc = 0
        while not is_end:
            batch_data, batch_y, is_end = test_set.next_batch(config.batch_size)
            test_acc += acc.eval(feed_dict={
                x: batch_data,
                y_: batch_y,
                p1: 1,
                p2: 1,
                p3: 1
            }, session=sess)
            gt = g_t.eval(feed_dict={
                x: batch_data,
                y_: batch_y,
                p1: 1,
                p2: 1,
                p3: 1
            }, session=sess)
            pr = p_r.eval(feed_dict={
                x: batch_data,
                y_: batch_y,
                p1: 1,
                p2: 1,
                p3: 1
            }, session=sess)
            gts += list(gt)
            prs += list(pr)
            loop_count += 1
        test_acc = test_acc/loop_count

        print('test set, acc: %g' % test_acc)
        print('Saving graph to: %s' % graph_location)
        post_process.dump_list(gts, config.gt_pickle)
        post_process.dump_list(prs, config.pr_pickle)
        print('save ground truth in', config.gt_pickle)
        print('save predict result in', config.pr_pickle)
        print('save checkpoint in', config.checkpoint_file)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

