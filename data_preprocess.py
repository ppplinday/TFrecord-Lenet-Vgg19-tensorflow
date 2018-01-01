import numpy as np
from scipy.misc import imresize
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

def transform(inputs, mean, std, random_angle=15., pca_sigma=255., expand_ratio=1.0, crop_size=(32, 32), train=True):
	img = inputs

	# Random rotate
	if random_angle != 0:
		angle = np.random.uniform(-random_angle, random_angle)
		img = cv_rotate(img, angle)

    # Color augmentation
	if train and pca_sigma != 0:
		img = pca_lighting(img, pca_sigma)

	return img