# Lenet-tensorflow

This is the tenet model and vgg19 model for cifar10, acc for lenet is about 84% and acc for vgg19 is about 86%. There are some wrong about validation dataset in my code, instead, I use test dataset as my validation(this is not the right way, but other in my code is right).

My code shows how to write tfrecord, data argument, train, eval and model

For building tfrecord

    python3 tfrecord.py

For training

    python3 train.py tenet

    python3 train.py vgg19


For eval
    python3 eval.py tenet

    python3 eavl.py vgg19

You also can change the config in config.py