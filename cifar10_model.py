import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model_cifar10:

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

        with slim.arg_scope([slim.conv2d], padding='VALID', weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            net1 = slim.conv2d(self.imnages, 4, [3, 3], 1, scope='conv11')
            print(net1.shape)
            net = slim.conv2d(self.images,32,[5,5],1,scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net,16,[5,5],1,scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net,120,[5,5],1,scope='conv5')
            net = slim.flatten(net, scope='flat6')
            net = slim.fully_connected(net, 84, scope='fc7')
            digits = slim.fully_connected(net, 10, scope='fc8')
        return digits

