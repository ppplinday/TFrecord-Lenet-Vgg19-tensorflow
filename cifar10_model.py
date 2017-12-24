import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model_cifar10:

    def __init__(self,
                 is_train=True):

        self.input_image = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.images = tf.reshape(self.input_image, [-1, 224, 224, 3])
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
            print(self.images.shape)
            net = slim.conv2d(self.images, 64, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv1')
            print(net.shape)
            net = slim.conv2d(net, 64, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv2')
            print(net.shape)
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            print(net.shape)

            net = slim.conv2d(net, 128, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv3')
            print(net.shape)
            net = slim.conv2d(net, 128, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv4')
            print(net.shape)
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            print(net.shape)

            net = slim.conv2d(net, 256, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv5')
            print(net.shape)
            net = slim.conv2d(net, 256, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv6')
            print(net.shape)
            net = slim.conv2d(net, 256, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv7')
            print(net.shape)
            net = slim.conv2d(net, 256, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv8')
            print(net.shape)
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            print(net.shape)

            net = slim.conv2d(net, 512, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv9')
            print(net.shape)
            net = slim.conv2d(net, 512, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv10')
            print(net.shape)
            net = slim.conv2d(net, 512, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv11')
            print(net.shape)
            net = slim.conv2d(net, 512, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv12')
            print(net.shape)
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            print(net.shape)

            net = slim.conv2d(net, 512, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv13')
            print(net.shape)
            net = slim.conv2d(net, 512, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv14')
            print(net.shape)
            net = slim.conv2d(net, 512, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv15')
            print(net.shape)
            net = slim.conv2d(net, 512, [3, 3], 1, padding='SAME', activation_fn=tf.nn.relu, scope='conv16')
            print(net.shape)
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            print(net.shape)

            net = slim.flatten(net, scope='flat')
            print(net.shape)
            net = slim.fully_connected(net, 4096, scope='fc1')
            print(net.shape)
            net = slim.fully_connected(net, 4096, scope='fc2')
            print(net.shape)
            net = slim.fully_connected(net, 10, scope='fc3')
            print(net.shape)

            digits = net
        return digits

