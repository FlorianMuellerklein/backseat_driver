import os
import gzip
import time
import pickle
import datetime
import random
import numpy as np
import pandas as pd

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum, adam
from lasagne.layers import helper

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE

from skimage.io import imshow, imsave, imread
from skimage.util import crop
from skimage import transform, filters, exposure

from models import ResNet_FullPre
from utils import load_test, batch_iterator_train, batch_iterator_valid

from matplotlib import pyplot

# testing params
BATCHSIZE = 1
PIXELS = 128
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)


'''
Set up all theano functions
'''
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = ResNet_FullPre(X, n=5)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

output_class = T.argmax(output_test, axis=1)

# set up training and prediction functions
predict_proba = theano.function(inputs=[X], outputs=output_test)
predict_class = theano.function(inputs=[X], outputs=output_class)

'''
Load data and make predictions
'''
# load data
X_train = np.load('data/cache/X_train_128_f32_bw.npy')
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], 1, PIXELS, PIXELS).astype('float32')

# load network weights
f = gzip.open('data/weights/ResNet42_7x7_BN_ortho_bw_128_fold00_last.pklz', 'rb')
all_params = pickle.load(f)
f.close()
helper.set_all_param_values(output_layer, all_params)

#make predictions
new_labels = []
for j in range((X_train.shape[0] + BATCHSIZE - 1) // BATCHSIZE):
    sl = slice(j * BATCHSIZE, (j + 1) * BATCHSIZE)
    X_batch = X_train[sl]
    new_labels.extend(predict_class(X_batch))

new_labels = np.array(new_labels)
print new_labels.shape

#np.save('data/cache/y_train_small_relabel.npy', new_labels)

'''
Compare differences
'''
#data = pd.read_csv('subm/pretrained_vgg_resnet42.csv')
#data = data.drop(['img'], axis=1)
#data = data.values
#new_labels = np.argmax(data, axis=1).astype('int32')

old_labels = np.load('data/cache/y_train_128_f32_bw.npy').astype('int32')
train_id = np.load('data/cache/X_train_id_128_f32_bw.npy')
same = 0
for i in range(new_labels.shape[0]):
    if old_labels[i] == new_labels[i]:
        same += 1
    else:
        print train_id[i]

print('Percent same, ', (float(same) / float(new_labels.shape[0])))
