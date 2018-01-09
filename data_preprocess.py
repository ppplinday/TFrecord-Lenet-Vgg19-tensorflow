import six
import random
import numpy as np
from scipy.misc import imresize
import tensorflow as tf
from skimage import transform as skimage_transform

def rotate_reshape(images, output_shape):
    """ Rotate and reshape n images"""
    # def r_r(img):
    #    """ Rotate and reshape one image """
    #    img = np.reshape(img, output_shape, order="F")
    #    img = np.rot90(img, k=3)
    # new_images = list(map(r_r, images))
    new_images = []
    for img in images:
        img = np.reshape(img, output_shape, order="F")
        # img = np.rot90(img, k=3)
        new_images.append(img)
    return new_images

def rescale(images, new_size):
    """ Rescale image to new size"""
    return list(map(lambda img: imresize(img, new_size), images))


def subtract_mean_rgb(images):
    """ Normalize by subtracting from the mean RGB value of all images"""
    return images - np.round(np.mean(images))

def _preprocess(images_1d, n_labels=10, dshape=(32, 32, 3),
                reshape=[224, 224, 3]):
    """ Preprocesses CIFAR10 images
    images_1d: np.ndarray
        Unprocessed images
    labels_1d: np.ndarray
        1d vector of labels
    n_labels: int, 10
        Images are split into 10 classes
    dshape: array, [32, 32, 3]
        Images are 32 by 32 RGB
    """
    # Reshape and rotate 1d vector into image
    images_raw = rotate_reshape(images_1d, dshape)
    # Rescale images to 224,244
    images = rescale(images_raw, reshape)
    # Subtract mean RGB value from every pixel
    #images = subtract_mean_rgb(images_rescaled)
    return images

USE_OPENCV = 0

def cv_rotate(img, angle):
    if USE_OPENCV == 1:
        # img = img.transpose(1, 2, 0) / 255.
        # center = (img.shape[0] // 2, img.shape[1] // 2)
        # r = cv.getRotationMatrix2D(center, angle, 1.0)
        # img = cv.warpAffine(img, r, img.shape[:2])
        # img = img.transpose(2, 0, 1) * 255.
        # img = img.astype(np.float32)
        a = 1
    else:
        # scikit-image's rotate function is almost 7x slower than OpenCV
        img = img / 255.
        img = skimage_transform.rotate(img, angle, mode='edge')
        img = img * 255.
        img = img.astype(np.float32)
    return img

def pca_lighting(img, sigma, eigen_value=None, eigen_vector=None):
    """AlexNet style color augmentation
    This method adds a noise vector drawn from a Gaussian. The direction of
    the Gaussian is same as that of the principal components of the dataset.
    This method is used in training of AlexNet [#]_.
    .. [#] Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. \
    ImageNet Classification with Deep Convolutional Neural Networks. \
    NIPS 2012.
    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW and RGB format.
        sigma (float): Standard deviation of the Gaussian. In the original
            paper, this value is 10% of the range of intensity
            (25.5 if the range is :math:`[0, 255]`).
        eigen_value (~numpy.ndarray): An array of eigen values. The shape
            has to be :math:`(3,)`. If it is not specified, the values computed
            from ImageNet are used.
        eigen_vector (~numpy.ndarray): An array of eigen vectors. The shape
            has to be :math:`(3, 3)`. If it is not specified, the vectors
            computed from ImageNet are used.
    Returns:
        An image in CHW format.
    """

    if sigma <= 0:
        return img

    # these values are copied from facebook/fb.resnet.torch
    if eigen_value is None:
        eigen_value = np.array((0.2175, 0.0188, 0.0045))
    if eigen_vector is None:
        eigen_vector = np.array((
            (-0.5675, -0.5808, -0.5836),
            (0.7192, -0.0045, -0.6948),
            (0.4009, -0.814,  0.4203)))

    alpha = np.random.normal(0, sigma, size=3)

    img = img.copy()
    img += eigen_vector.dot(eigen_value * alpha).reshape((1, 1, 3))

    return img

def random_flip(img):
	ccc = random.choice([True, False])
	if ccc == True:
		img = img[:, ::-1, :]
	return img

def random_expand(img, max_ratio=4):
	H,W,C = img.shape
	ratio = random.uniform(1, max_ratio)
	out_H, out_W = int(H * ratio), int(W * ratio)

	x_offset = random.randint(0, out_H - H)
	y_offset = random.randint(0, out_W - W)

	out_img = np.empty((out_H, out_W, C), dtype=img.dtype)
	out_img[:] = np.array(0).reshape((1, 1, -1))
	out_img[y_offset:y_offset + H, x_offset:x_offset + W, :] = img

	return out_img

def random_crop(img, size):
	H, W = size
	if img.shape[0] == H:
		x_offset = 0
	else:
		x_offset = random.choice(range(img.shape[0] - H))
	x_slice = slice(x_offset, x_offset + H)

	if img.shape[1] == W:
		y_offset = 0
	else:
		y_offset = random.choice(range(img.shape[1] - W))
	y_slice = slice(y_offset, y_offset + W)

	img = img[x_slice, y_slice, :]
	return img


def transform(inputs, mean, std, random_angle=15., pca_sigma=255., expand_ratio=1.0, crop_size=(32, 32), train=True):
	img = inputs

	# Random rotate
	if random_angle != 0:
		angle = np.random.uniform(-random_angle, random_angle)
		img = cv_rotate(img, angle)

    # Color augmentation
	if train and pca_sigma != 0:
		img = pca_lighting(img, pca_sigma)

	# Standardization
	img -= mean[None, None, :]
	img /= std[None, None, :]

	if train == True:
		img = random_flip(img)
		if expand_ratio > 1:
			img = random_expand(img, expand_ratio)
	
		img = random_crop(img, crop_size)

	return img

def transform_test(inputs, mean, std, random_angle=15., pca_sigma=255., expand_ratio=1.0, crop_size=(32, 32), train=True):
	img = inputs
	img = random_crop(img, crop_size)
	return img

def data_preprocess(X_train, train=True, model='lenet'):
	x_mean1 = tf.reduce_mean(X_train, axis=(1,2,3))
	x_mean2 = tf.reduce_mean(X_train, axis=(0,1,2))
	x_mean = np.mean([x for x in X_train], axis=(0,1,2))
	print(x_mean1.eval())
	print(x_mean2.eval())
	print(x_mean)
	x_std = np.std([x for x in X_train], axis=(0,1,2))
	x_res = []
	for x in X_train:
		if model == 'lenet':
			img = transform(x, x_mean, x_std, expand_ratio=1.2, crop_size=(28,28), train=train)
		else:
			img = transform(x, x_mean, x_std, expand_ratio=1.2, crop_size=(32,32), train=train)
		x_res.append(img)
	x_res = np.array(x_res)
	return x_res

def label_one_hot(input, num_class):
	output = np.zeros((input.shape[0], num_class))
	for i in range(input.shape[0]):
		output[i, input[i]] = 1.0
	return output