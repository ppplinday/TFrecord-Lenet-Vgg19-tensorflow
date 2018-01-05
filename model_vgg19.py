import tensorflow as tf
import tensorflow.contrib.slim as slim
import config


class Model_Vgg19:

    def __init__(self, is_train=True):

        self.input_image = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.images = tf.reshape(self.input_image, [-1, 32, 32, 3])
        self.input_label = tf.placeholder(tf.float32, [None, 10])
        self.labels = tf.cast(self.input_label, tf.int32)
        self.global_step = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.num_sample = config.num_sample
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.weight_decay = 0.0005
        print(config.learning_rate)
        
        with tf.variable_scope("Vgg19") as scope:
            self.train_digits = self.build(True)
            scope.reuse_variables()
            self.pred_digits = self.build(False)

        self.prediction = tf.argmax(self.pred_digits, 1)
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits, 1), tf.argmax(self.labels, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        self.loss = slim.losses.softmax_cross_entropy(self.train_digits, self.labels)
        self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step,
         (self.num_sample // self.batch_size) * config.epoch_decay, config.learning_decay, staircase=True)
        self.train_op = tf.train.MomentumOptimizer(self.lr, config.momentum).minimize(self.loss, global_step=self.global_step)


    def build(self, is_train=True):

        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
            activation_fn=tf.nn.relu, 
            weights_regularizer=slim.l2_regularizer(self.weight_decay),
            biases_initializer=tf.zeros_initializer):
            print(self.images.shape)
            net = slim.repeat(self.images, 2, slim.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            print(net.shape)

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], padding='SAME', scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            print(net.shape)

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], padding='SAME', scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            print(net.shape)
            
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], padding='SAME', scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            print(net.shape)

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], padding='SAME', scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            print(net.shape)

            #net = slim.flatten(net, scope='flat')
            #net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc1')
            #net = slim.dropout(net, 0.5, is_training=is_train, scope='drop4')
            net = slim.conv2d(net, 1024, [1, 1], padding='SAME', scope='fc6')
            net = slim.dropout(net, 0.5, is_training=is_train, scope='dropout6')
            print(net.shape)

            net = slim.conv2d(net, 1024, [1, 1], padding='SAME', scope='fc7')
            net = slim.dropout(net, 0.5, is_training=is_train, scope='dropout7')
            print(net.shape)

            digits = slim.conv2d(net, 10, [1, 1], padding='SAME', scope='fc8')
            print(digits.shape)
        return digits

