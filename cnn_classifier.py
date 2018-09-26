# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf

myself_folder_path = 'data/myself'
others_folder_path = 'data/others'


def read_data_sets(data_path, ismyself=True):
    images = []
    for filename in os.listdir(data_path):
        if filename.endswith('.jpg'):
            file_path = os.path.join(data_path, filename)
            image = cv2.imread(file_path)
            images.append(image)
    labels = [0] * len(images) if ismyself else [1] * len(images)
    return images, labels


images1, labels1 = read_data_sets(myself_folder_path, ismyself=True)
images2, labels2 = read_data_sets(others_folder_path, ismyself=False)
images = np.array(images1 + images2, dtype=np.float32)
labels = np.array(labels1 + labels2, dtype=np.int64)

num_images = images.shape[0]
perm = np.arange(num_images)
np.random.shuffle(perm)
images = images[perm]
labels = labels[perm]
train_size = num_images * 7 // 10
test_size = num_images - train_size
train_images = images[:train_size]
train_labels = labels[:train_size]
test_images = images[train_size:]
test_labels = labels[train_size:]


class DataSet(object):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), \
                   np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


train = DataSet(train_images, train_labels)
test = DataSet(test_images, test_labels)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def cnn_model_fn(x):
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 64])
    W_fc1 = weight_variable([8 * 8 * 64, 256])
    b_fc1 = bias_variable([256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([256, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y_ = tf.placeholder(tf.int64, [None])

y_conv, keep_prob = cnn_model_fn(x)

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

max_steps = train_size // 100 * 5
batch_size = 100
models_dir = 'models'
print('max_steps = ', max_steps)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(max_steps):
        batch = train.next_batch(batch_size=batch_size)
        if step % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (step, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        if (step + 1) % 100 == 0 or (step + 1) == max_steps:
            checkpoint_file = os.path.join(models_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file)
            accuracy_l = []
            for _ in range(test_size // 500):
                batch = test.next_batch(500, shuffle=False)
                accuracy_l.append(accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
            print('test accuracy %g' % np.mean(accuracy_l))
