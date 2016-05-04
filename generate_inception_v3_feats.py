import os
import glob
import math
import time
import random
import pickle
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

import lasagne
from lasagne.updates import adam
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, FlattenLayer, helper
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

import theano
from theano import tensor as T

from utils import load_train, load_test
from models import inception_v3

"""
Set up all theano functions
"""

X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
net = inception_v3(X)
# load network weights
d = pickle.load(open('data/pre_trained_weights/inception_v3.pkl'))
helper.set_all_param_values(net['softmax'], d['param values'])

inception_v3_feats = lasagne.layers.get_output(FlattenLayer(net['mixed_10/join'], outdim=2), deterministic=True)
get_feats = theano.function(inputs=[X], outputs=inception_v3_feats)

'''
load training data and start training
'''
encoder = LabelEncoder()

# load data
X_train, y_train, encoder = load_train(encoder, cache=True)

# loop over training functions for however many iterations, print information while training
# last hidden layer
TEST_BATCHSIZE = 1
pool5_features = []
for j in range((X_train.shape[0] + TEST_BATCHSIZE -1) // TEST_BATCHSIZE):
    sl = slice(j * TEST_BATCHSIZE, (j + 1) * TEST_BATCHSIZE)
    X_batch = X_train[sl]
    pool5_features.extend(get_feats(X_batch))

pool5_features = np.array(pool5_features, dtype='float32')
# (22424, 2048)
print pool5_features.shape
np.save('data/cache/inception_v3_features_train.npy', pool5_features)

X_train = None
y_train = None

X_test, X_test_id = load_test(cache=True)

# loop over training functions for however many iterations, print information while training
# last hidden layer
pool5_features_test = []
for j in range((X_test.shape[0] + TEST_BATCHSIZE -1) // TEST_BATCHSIZE):
    sl = slice(j * TEST_BATCHSIZE, (j + 1) * TEST_BATCHSIZE)
    X_batch = X_test[sl]
    pool5_features_test.extend(get_feats(X_batch))

pool5_features_test = np.array(pool5_features_test, dtype='float32')
# (79726, 2048)
print pool5_features_test.shape
np.save('data/cache/inception_v3_features_test.npy', pool5_features_test)
