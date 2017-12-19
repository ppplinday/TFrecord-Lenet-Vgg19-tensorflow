from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import conv2d, max_pool, fc


class Model:

    def __init__(self,
                 is_train=True):

        self.batch_size = 32
        self.learning_rate = 1e-3

        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer()) 

    def build(self, is_train=True):

        self.input_image = tf.placeholder(tf.float32, [None, 3072])
        self.images = tf.reshape(self.input_image, [-1, 32, 32, 3])
        self.input_label = tf.placeholder(tf.float32, [None, 10])
        self.labels = tf.cast(self.input_label, tf.int32)

        with tf.variable_scope('net')
            print('[name {}] shape: {}'.format('init', self.images.shape))
            net = conv2d(self.images, 6, name='conv1')
            print('[name {}] shape: {}'.format('conv1', net.shape))
            net = max_pool(net, name='max_pool1')
            print('[name {}] shape: {}'.format('p1', net.shape))
            net = conv2d(net, 16, name='conv2')
            print('[name {}] shape: {}'.format('conv2', net.shape))
            net = max_pool(net, name='max_pool2')
            print('[name {}] shape: {}'.format('p2', net.shape))
            net = conv2d(net, 120, name = 'conv3')
            print('[name {}] shape: {}'.format('conv3', net.shape))
            net = tf.layers.flatten(net)
            print('[name {}] shape: {}'.format('fff', net.shape))
            net = fc(net, 84, name='fc1')
            print('[name {}] shape: {}'.format('fc1', net.shape))
            net = fc(net, 10, name='fc2')
            print('[name {}] shape: {}'.format('fc2', net.shape))
            net = tf.nn.softmax(net)
            print('[name {}] shape: {}'.format('softmax', net.shape))

        self.prediction = tf.argmax(net, 1)
        self.correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(self.labels, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        self.loss = slim.losses.softmax_cross_entropy(net, self.labels)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def run_single_step(self, images, labels):
        _, accuracy, loss = self.sess.run(
            [self.train_op, self.train_accuracy, self.loss], feed_dict={self.input_image : images, self.input_label : labels})
        return loss, accuracy

