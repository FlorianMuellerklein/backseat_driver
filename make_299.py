import os
import glob
import math
import random
import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE

from matplotlib import pyplot
from skimage.io import imshow, imsave, imread
from skimage.util import crop
from skimage import transform, filters, exposure, img_as_ubyte

import theano
from theano import tensor as T

import argparsing
args, unknown_args = argparsing.parse_args()

PIXELS = 299
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

def load_train_cv():
    X_train = np.empty(shape=(21794, num_features), dtype='float32')
    X_train_id = []
    y_train = np.empty(shape=(21794,1), dtype='int')
    print('Read train images')
    file_count = 0
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('data', 'imgs', 'train_cleaned', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            print fl
            flbase = os.path.basename(fl)
            img = imread(fl)
            img = transform.resize(img, output_shape=(PIXELS, PIXELS, 3), preserve_range=True)
            img = img.transpose(2, 0, 1)
            img = np.reshape(img, (1, num_features)).astype('float32')
            X_train[file_count] = img
            y_train[file_count] = j
            file_count += 1
            X_train_id.append('c' + str(j) + '/' + str(flbase))

    X_train_id = np.array(X_train_id)
    y_train = np.array(y_train)

    np.save('data/cache/X_train_%d_f32_clean.npy'%PIXELS, X_train)
    np.save('data/cache/y_train_%d_f32_clean.npy'%PIXELS, y_train)
    np.save('data/cache/X_train_id_%d_f32_clean.npy'%PIXELS, X_train_id)

load_train_cv()
