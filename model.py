from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import conv2d, max_pool 


class Model(object):

    def __init__(self,
                 config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.c_dim = self.config.data_info[2]
        self.conv_info = self.config.conv_info

        self.image = tf.placeholder(
            name = 'image', dtype = tf.float32,
            shape = [self.batch_size, self.input_height, self.input_width, self.c_dim],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
    	fd = {
    		self.image : batch_chunk['image'],
    	}

    	if is_training is not None:
    		fd[self.is_training] = is_training

    	return fd

    def build(self, is_train=True):

    	net = conv2d(self.image, 6, name='conv1')
    	net = max_pool(net, name='max_pool1')
    	net = conv2d(net, 16, name='conv2')
    	net = max_pool1(net, name='max_pool2')
    	net = conv2d(net, 120, name = 'conv3')
    	net = tf.layers.flatten(net, dtype='float32')
    	net = fc(net, 84, name='fc1')
    	net = fc(net, 10, name='fc2')
    	net = tf.nn.softmax(net)




