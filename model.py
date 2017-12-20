from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model:

    def __init__(self,
                 is_train=True):

        self.input_image = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.images = tf.reshape(self.input_image, [-1, 32, 32, 3])
        self.input_label = tf.placeholder(tf.float32, [None, 10])
        self.labels = tf.cast(self.input_label, tf.int32)

        self.batch_size = 32
        self.learning_rate = 1e-3

        with tf.variable_scope("Lenet") as scope:
            self.train_digits = self.build(True)
            scope.reuse_variables()
            self.pred_digits = self.build(False)

        self.prediction = tf.argmax(self.pred_digits, 1)
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits, 1), tf.argmax(self.labels, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        self.loss = slim.losses.softmax_cross_entropy(self.train_digits, self.labels)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def build(self, is_train=True):

        # with tf.variable_scope('net'):
        #     print('[name {}] shape: {}'.format('init', self.images.shape))
        #     net = conv2d(self.images, 6, name='conv1')
        #     print('[name {}] shape: {}'.format('conv1', net.shape))
        #     net = max_pool(net, name='max_pool1')
        #     print('[name {}] shape: {}'.format('p1', net.shape))
        #     net = conv2d(net, 16, name='conv2')
        #     print('[name {}] shape: {}'.format('conv2', net.shape))
        #     net = max_pool(net, name='max_pool2')
        #     print('[name {}] shape: {}'.format('p2', net.shape))
        #     net = conv2d(net, 120, name = 'conv3')
        #     print('[name {}] shape: {}'.format('conv3', net.shape))
        #     net = tf.layers.flatten(net)
        #     print('[name {}] shape: {}'.format('fff', net.shape))
        #     net = fc(net, 84, name='fc1')
        #     print('[name {}] shape: {}'.format('fc1', net.shape))
        #     net = fc(net, 10, name='fc2')
        #     print('[name {}] shape: {}'.format('fc2', net.shape))
        #     net = tf.nn.softmax(net)
        #     print('[name {}] shape: {}'.format('softmax', net.shape))
        
        with slim.arg_scope([slim.conv2d], padding='VALID', weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            print('[name {}] shape: {}'.format('init', self.images.shape))
            net = slim.conv2d(self.images,6,[5,5],1,scope='conv1')
            print('[name {}] shape: {}'.format('conv1', net.shape))
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            print('[name {}] shape: {}'.format('p2', net.shape))
            net = slim.conv2d(net,16,[5,5],1,scope='conv3')
            print('[name {}] shape: {}'.format('conv3', net.shape))
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            print('[name {}] shape: {}'.format('p4', net.shape))
            net = slim.conv2d(net,120,[5,5],1,scope='conv5')
            print('[name {}] shape: {}'.format('conv5', net.shape))
            net = slim.flatten(net, scope='flat6')
            print('[name {}] shape: {}'.format('flat6', net.shape))
            net = slim.fully_connected(net, 84, scope='fc7')
            print('[name {}] shape: {}'.format('fc7', net.shape))
            digits = slim.fully_connected(net, 10, scope='fc8')
            print('[name {}] shape: {}'.format('fc8', digits.shape))
        return digits

