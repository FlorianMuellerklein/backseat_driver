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
from lasagne.layers import helper, DenseLayer
from lasagne.nonlinearities import softmax

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE

from skimage.io import imshow, imsave, imread
from skimage.util import crop
from skimage import transform, filters, exposure

from models import ResNet_FullPre, ResNet_BttlNck_FullPre, blvc_googlenet, inception_v3, ResNet_Orig_ELU
from utils import load_test, batch_iterator_train, batch_iterator_valid

import argparsing
args, unknown_args = argparsing.parse_args()

experiment_label = args.label
PIXELS = args.pixels
BATCHSIZE = args.batchsize

imageSize = PIXELS * PIXELS
num_features = imageSize * 3

'''
Set up all theano functions
'''
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = ResNet_FullPre(X, n=5)
#net = inception_v3(X)

# stack our own softmax onto the final layer
#output_layer = DenseLayer(net['pool3'], num_units=10, W=lasagne.init.HeNormal(), nonlinearity=softmax)

output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up training and prediction functions
predict_proba = theano.function(inputs=[X], outputs=output_test)
'''
Load data and make predictions
'''
# load data
X_test, X_test_id = load_test(cache=True)
print 'Test shape:', X_test.shape
print np.amax(X_test), np.amin(X_test), np.mean(X_test)

# load network weights
#f = gzip.open('data/weights/resnet45_fullpre_more_L2.pklz', 'rb')
f = gzip.open('data/weights/%s.pklz'%experiment_label, 'rb')
all_params = pickle.load(f)
f.close()
helper.set_all_param_values(output_layer, all_params)
'''
#make predictions
predictions = []
for j in range((X_test.shape[0] + BATCHSIZE - 1) // BATCHSIZE):
    sl = slice(j * BATCHSIZE, (j + 1) * BATCHSIZE)
    X_batch = X_test[sl]
    predictions.extend(predict_proba(X_batch))

predictions = np.array(predictions)
print predictions.shape
'''
#Test Time Augmentations

PAD_CROP = 8
PAD_PIXELS = PIXELS + (PAD_CROP * 2)
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

tta_iter = 1
for _ in range(15):
    predictions_tta = []
    for i in range((X_test.shape[0] + BATCHSIZE - 1) // BATCHSIZE):
        sl = slice(i * BATCHSIZE, (i + 1) * BATCHSIZE)
        X_batch = X_test[sl]

        # pad and crop settings
        trans_1 = random.randint(0, (PAD_CROP*2))
        trans_2 = random.randint(0, (PAD_CROP*2))
        crop_x1 = trans_1
        crop_x2 = (PIXELS + trans_1)
        crop_y1 = trans_2
        crop_y2 = (PIXELS + trans_2)

        # brightness settings
        bright = random.uniform(0.95,1.05)

        r_intensity = random.randint(0,1)
        g_intensity = random.randint(0,1)
        b_intensity = random.randint(0,1)
        intensity_scaler = random.randint(-5, 5)

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.copy(X_batch)

        # for each image in the batch do the augmentation
        for j in range(X_batch.shape[0]):
            # for each image channel
            for k in range(X_batch.shape[1]):
                # pad and crop images
                img_pad = np.pad(X_batch_aug[j,k], pad_width=((PAD_CROP,PAD_CROP), (PAD_CROP,PAD_CROP)), mode='constant')
                X_batch_aug[j,k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                # adjust brightness
                X_batch_aug[j,k] = X_batch_aug[j,k] * bright

            if r_intensity == 1:
                X_batch_aug[j][0] += intensity_scaler
            if g_intensity == 1:
                X_batch_aug[j][1] += intensity_scaler
            if b_intensity == 1:
                X_batch_aug[j][2] += intensity_scaler

        # print statements for debugging post augmentation
        #img_max =  np.amax(X_batch_aug)
        #plot_sample(X_batch_aug[0] / img_max)

        # fit model on each batch
        predictions_tta.extend(predict_proba(X_batch_aug))

    predictions_tta = np.array(predictions_tta)
    np.save('data/tta_temp/predictions_tta_' + str(tta_iter) + '.npy', predictions_tta)

    tta_iter += 1

# average all TTA predictions
tta_sub_1 = np.load('data/tta_temp/predictions_tta_1.npy')
tta_sub_2 = np.load('data/tta_temp/predictions_tta_2.npy')
tta_sub_3 = np.load('data/tta_temp/predictions_tta_3.npy')
tta_sub_4 = np.load('data/tta_temp/predictions_tta_4.npy')
tta_sub_5 = np.load('data/tta_temp/predictions_tta_5.npy')
tta_sub_6 = np.load('data/tta_temp/predictions_tta_6.npy')
tta_sub_7 = np.load('data/tta_temp/predictions_tta_7.npy')
tta_sub_8 = np.load('data/tta_temp/predictions_tta_8.npy')
tta_sub_9 = np.load('data/tta_temp/predictions_tta_9.npy')
tta_sub_10 = np.load('data/tta_temp/predictions_tta_10.npy')
tta_sub_11 = np.load('data/tta_temp/predictions_tta_11.npy')
tta_sub_12 = np.load('data/tta_temp/predictions_tta_12.npy')
tta_sub_13 = np.load('data/tta_temp/predictions_tta_13.npy')
tta_sub_14 = np.load('data/tta_temp/predictions_tta_14.npy')
tta_sub_15 = np.load('data/tta_temp/predictions_tta_15.npy')

predictions = (tta_sub_1 + tta_sub_2 + tta_sub_3 + tta_sub_4 + tta_sub_5 + tta_sub_6 + tta_sub_7 + tta_sub_8 + tta_sub_9 + tta_sub_10 + tta_sub_11 + tta_sub_12 + tta_sub_13 + tta_sub_14 + tta_sub_15) / 15.0

'''
Make submission file
'''
def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', '%s.csv'%experiment_label)
    result1.to_csv(sub_file, index=False)

create_submission(predictions, X_test_id)
