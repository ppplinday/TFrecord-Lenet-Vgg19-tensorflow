# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle as p
from data_preprocess import label_one_hot

def load_CIFAR_batch(filename):
    
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT, one_hot=True):

    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)         
        ys.append(Y)
    Xtr = np.concatenate(xs) 
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    if one_hot == True:
        Ytr = label_one_hot(Ytr, 10)
        Yte = label_one_hot(Yte, 10)
    return Xtr, Ytr, Xte, Yte