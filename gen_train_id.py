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

PIXELS = args.pixels
PAD_CROP = int(PIXELS * 0.125)
PAD_PIXELS = PIXELS + (PAD_CROP * 2)
imageSize = PIXELS * PIXELS
num_features = imageSize #* 3


def generate_train_id():
    X_train = []
    y_train = []
    X_test_id = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('data', 'imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            print fl
            flbase = os.path.basename(fl)
            img = imread(fl, as_grey = True)
            img = transform.resize(img, output_shape=(PIXELS, PIXELS), preserve_range=True)
            #img = img.transpose(2, 0, 1)
            img = np.reshape(img, (1, num_features))
            X_train.append(img)
            y_train.append(j)
            X_test_id.append('c' + str(j) + '/' + str(flbase))

    X_train = np.array(X_train, dtype='float32')
    y_train = np.array(y_train)
    X_test_id = np.array(X_test_id)

    np.save('data/cache/X_train_%d_f32_bw.npy'%PIXELS, X_train)
    np.save('data/cache/y_train_%d_f32_bw.npy'%PIXELS, y_train)
    np.save('data/cache/X_train_id_%d_f32_bw.npy'%PIXELS, X_test_id)

generate_train_id()
