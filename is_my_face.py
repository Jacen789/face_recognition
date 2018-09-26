# -*- coding: utf-8 -*-

import cv2
import dlib
import tensorflow as tf


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
y_conv, keep_prob = cnn_model_fn(x)
predict = tf.argmax(y_conv, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('models'))

predictor_file_path = 'data/models/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_file_path)

cam = cv2.VideoCapture(0)

while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found.")
        continue
    img_copy = img.copy()
    for detection in dets:
        image = dlib.get_face_chip(img, sp(img, detection), size=64)
        cv2.imshow('image', image)
        pred = sess.run(predict, feed_dict={x: [image], keep_prob: 1.0})

        left = detection.left()
        top = detection.top()
        right = detection.right()
        bottom = detection.bottom()
        cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 0, 255), thickness=1)

        if pred[0] == 0:
            cv2.putText(img_copy, 'True', (left, top), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255))
        else:
            cv2.putText(img_copy, 'False', (left, top), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255))

    cv2.imshow('camera', img_copy)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
