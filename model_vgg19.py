import tensorflow as tf
import tensorflow.contrib.slim as slim
import config


class Model_Vgg19:

    def __init__(self, is_train=True):

        self.input_image = tf.placeholder(tf.float32, [None, 28, 28, 3])
        self.images = tf.reshape(self.input_image, [-1, 28, 28, 3])
        self.input_label = tf.placeholder(tf.float32, [None, 10])
        self.labels = tf.cast(self.input_label, tf.int32)
        self.global_step = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.num_sample = config.num_sample
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        
        with tf.variable_scope("Lenet") as scope:
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

        with slim.arg_scope([slim.conv2d], padding='VALID', weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
            net = slim.conv2d(self.images, 64, [3, 3], 1, padding='SAME', scope='conv1_1')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn1_1')
            net = slim.conv2d(net, 64, [3, 3], 1, padding='SAME', scope='conv1_2')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn1_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.dropout(net, 0.25, is_training=is_train, scope='drop1')

            net = slim.conv2d(net, 128, [3, 3], 1, padding='SAME', scope='conv2_1')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn2_1')
            net = slim.conv2d(net, 128, [3, 3], 1, padding='SAME', scope='conv2_2')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.dropout(net, 0.25, is_training=is_train, scope='drop2')

            net = slim.conv2d(net, 256, [3, 3], 1, padding='SAME', scope='conv3_1')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn3_1')
            net = slim.conv2d(net, 256, [3, 3], 1, padding='SAME', scope='conv3_2')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn3_2')
            net = slim.conv2d(net, 256, [3, 3], 1, padding='SAME', scope='conv3_3')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn3_3')
            net = slim.conv2d(net, 256, [3, 3], 1, padding='SAME', scope='conv3_4')
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='bn3_4')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.dropout(net, 0.25, is_training=is_train, scope='drop3')

            net = slim.flatten(net, scope='flat')
            net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc1')
            net = slim.dropout(net, 0.5, is_training=is_train, scope='drop4')
            net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc2')
            net = slim.dropout(net, 0.5, is_training=is_train, scope='drop5')
            digits = slim.fully_connected(net, 10, scope='fc3')
        return digits

