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
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, FlattenLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

import theano
from theano import tensor as T

from utils import load_train, load_test

# Download a pickle containing the pretrained weights
#print('downloading pretrained vgg16 ... ')
#!wget -N https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

# Model definition for VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

def build_model(input_var=None):
    net = {}
    net['input'] = InputLayer((None, 3, 128, 128), input_var=input_var)
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

"""
Set up all theano functions
"""

X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
net = build_model(X)
# load network weights
d = pickle.load(open('data/pre_trained_weights/vgg16.pkl'))
lasagne.layers.set_all_param_values(net['prob'], d['param values'])

vgg_feats = lasagne.layers.get_output(FlattenLayer(net['pool5'], outdim=2), deterministic=True)
get_feats = theano.function(inputs=[X],outputs=vgg_feats)

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
print pool5_features.shape
np.save('data/cache/vgg16_features_train.npy', pool5_features)

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
# (22424, 8192)
print pool5_features_test.shape
np.save('data/cache/vgg16_features_test.npy', pool5_features_test)
