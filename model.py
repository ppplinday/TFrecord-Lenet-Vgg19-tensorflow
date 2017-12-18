from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import conv2d, max_pool 


class Model(object):

    def __init__(self,
                 is_train=True):

        self.batch_size = 32
        self.learning_rate = 1e-3

        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer()) 

    def build(self, is_train=True):

        self.input_image = tf.placeholder(tf.float32, [None, 3072])
        self.images = tf.reshape(self.input_image, [-1, 32, 32, 1])
        self.input_label = tf.placeholder(tf.float32, [None, 10])
        self.labels = tf.cast(self.input_label, tf.int32)

    	net = conv2d(self.image, 6, name='conv1')
    	net = max_pool(net, name='max_pool1')
    	net = conv2d(net, 16, name='conv2')
    	net = max_pool1(net, name='max_pool2')
    	net = conv2d(net, 120, name = 'conv3')
    	net = tf.layers.flatten(net, dtype='float32')
    	net = fc(net, 84, name='fc1')
    	net = fc(net, 10, name='fc2')
    	net = tf.nn.softmax(net)

        self.prediction = tf.argmax(net, 1)
        self.correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(self.input_labels, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        self.loss = slim.losses.softmax_cross_entropy(self.train_digits, self.input_labels)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def run_single_step(self, images, labels):
        _, accuracy, loss = self.sess.run(
            [self.train_op, self.train_accuracy, self.loss], feed_dict={self.input_image : images, self.input_label : labels})
        return loss, accuracy

