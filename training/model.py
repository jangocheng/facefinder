# coding=utf-8
"""Facefinder Model file """
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
from tensorflow import nn
import load


# Make weight and bias variables
def weight(shape):
    """
    Tensorflow placeholder
    Args:
        shape: 

    Returns:

    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="weight")


def bias(shape):
    """
    Tensorflow bias
    Args:
        shape: 

    Returns:

    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")


# Finds the product of a dimension tuple to find the total legth
def dim_prod(dim_arr):
    return np.prod([d for d in dim_arr if d != None])


# Start a TensorFlow session
def start_sess():
    """
    Start the tensorflow session
    Returns:

    """
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    return sess


# Train the model
def train(sess, y, x_hold, y_hold, keep_prob, X, Y, valX, valY, lrate=0.5, epsilon=1e-8, n_epoch=100, batch_size=10,
          print_epoch=100, save_path=None):
    """
    Train the network
    Args:
        sess: 
        y: 
        x_hold: 
        y_hold: 
        keep_prob: 
        X: 
        Y: 
        valX: 
        valY: 
        lrate: 
        epsilon: 
        n_epoch: 
        batch_size: 
        print_epoch: 
        save_path: 

    Returns:

    """
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_hold * tf.log(y + 1e-10), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(learning_rate=lrate, epsilon=epsilon).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hold, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Flatten the input images for the placeholder
    flat_len = dim_prod(x_hold._shape_as_list())
    X = X.reshape((X.shape[0], flat_len))

    print('Starting training session...')

    sess.run(tf.initialize_all_variables())

    batch_num = 0

    batches = batchify(X, Y, batch_size)

    print('Number of batches:', len(batches))

    for i in range(n_epoch):
        avg_acc = 0
        random.shuffle(batches)
        for batchX, batchY in batches:
            avg_acc = avg_acc + accuracy.eval(session=sess, feed_dict={x_hold: batchX, y_hold: batchY, keep_prob: 1})
            train_step.run(session=sess, feed_dict={x_hold: batchX, y_hold: batchY, keep_prob: 0.5})
        print('Epoch ' + str(i) + ': ' + str(avg_acc / len(batches)))
    if (not valX is None) & (not valY is None):
        # Validation
        valX = valX.reshape((valX.shape[0], flat_len))
        val_accuracy = accuracy.eval(session=sess, feed_dict={x_hold: valX, y_hold: valY, keep_prob: 1})
        print('Val acc:', val_accuracy)

    if not save_path is None:
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess, save_path)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(save_path + '_graph', sess.graph)
        writer.flush()
        writer.close()
        print('Model saved')
    return val_accuracy


# Test a model
def test(sess, X, Y, model_path):
    """
    test function
    Args:
        sess: 
        X: 
        Y: 
        model_path: 

    Returns:

    """
    correct_prediction = tf.equal(tf.argmax(self.net, 1), tf.argmax(self.y_hold, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, model_path)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    test_accuracy = accuracy.eval(session=sess, feed_dict={x_hold: X, y_hold: Y, keep_prob: 1})
    return test_accuracy


# Split to mini batches
def batchify(X, Y, batch_size):
    batches = [(X[i:i + batch_size], Y[i:i + batch_size]) for i in xrange(0, X.shape[0], batch_size)]
    random.shuffle(batches)
    return batches


# Build the net in the session
def build_net(sess):
    """
    
    Args:
        sess: 

    Returns:

    """
    in_len = 32
    in_dep = 1

    x_hold = tf.placeholder(tf.float32, shape=[None, in_dep * in_len * in_len])
    y_hold = tf.placeholder(tf.float32, shape=[None, 2])
    keep_prob = tf.placeholder(tf.float32)

    xt = tf.reshape(x_hold, [-1, in_len, in_len, in_dep])

    # Layer 1 - 5x5 convolution
    w1 = weight([5, 5, in_dep, 4])
    b1 = bias([4])
    c1 = nn.relu(nn.conv2d(xt, w1, strides=[1, 2, 2, 1], padding='VALID') + b1)
    o1 = c1

    # Layer 2 - 3x3 convolution
    w2 = weight([3, 3, 4, 16])
    b2 = bias([16])
    c2 = nn.relu(nn.conv2d(o1, w2, strides=[1, 2, 2, 1], padding='VALID') + b2)
    o2 = c2

    # Layer 3 - 3x3 convolution
    w3 = weight([3, 3, 16, 32])
    b3 = bias([32])
    c3 = nn.relu(nn.conv2d(o2, w3, strides=[1, 1, 1, 1], padding='VALID') + b3)
    o3 = c3

    dim = 32 * 4 * 4

    # Fully connected layer - 600 units
    of = tf.reshape(o3, [-1, dim])
    w4 = weight([dim, 600])
    b4 = bias([600])
    o4 = nn.relu(tf.matmul(of, w4) + b4)

    o4 = nn.dropout(o4, keep_prob)

    # Output softmax layer - 2 units
    w5 = weight([600, 2])
    b5 = bias([2])
    y = nn.softmax(tf.matmul(o4, w5) + b5)

    sess.run(tf.global_variables_initializer())

    return y, x_hold, y_hold, keep_prob


# Method to run the training
def train_net():
    """
    
    Returns:

    """
    train, val, test = load.load_dataset()
    sess = start_sess()
    y, x_hold, y_hold, keep_prob = build_net(sess)
    acc = train(sess,
                y,
                x_hold,
                y_hold,
                keep_prob,
                train[0], train[1],
                test[0], test[1],
                lrate=1e-4,
                epsilon=1e-16,
                n_epoch=8,
                batch_size=100,
                print_epoch=1,
                save_path=model_path)
    print("Accuracy:", acc)
    sess.close()
