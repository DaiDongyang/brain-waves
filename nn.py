import tensorflow as tf
import tf_data
import load_data
import config


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

    return h4





def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    pass