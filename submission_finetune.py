import os
import gzip
import time
import glob
import math
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

from models import ResNet_FullPre, ResNet_FullPre_Wide, ST_ResNet_FullPre, bvlc_googlenet_submission
from utils import load_test, batch_iterator_train, batch_iterator_valid

import argparsing
args, unknown_args = argparsing.parse_args()

experiment_label = args.label
PIXELS = args.pixels
BATCHSIZE = 1

imageSize = PIXELS * PIXELS
num_features = imageSize * 3

'''
Set up all theano functions
'''
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = bvlc_googlenet_submission(X)

output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up training and prediction functions
predict_proba = theano.function(inputs=[X], outputs=output_test)


# load network weights
f = gzip.open('data/weights/%s_last.pklz'%experiment_label, 'rb')
#f = gzip.open('data/weights/%s.pklz'%experiment_label, 'rb')
all_params = pickle.load(f)
f.close()
helper.set_all_param_values(output_layer, all_params)



#make predictions
predictions = []
# load each image
path = os.path.join('data', 'imgs', 'test', '*.jpg')
files = glob.glob(path)
for fl in files:
    flbase = os.path.basename(fl)
    img = imread(fl)
    img = transform.resize(img, output_shape=(PIXELS, PIXELS, 3), preserve_range=True)
    img = img.transpose(2, 0, 1)
    img = img.astype('float32')

    img_pred = np.ones(shape=(1,3,224,224), dtype='float32')
    img_pred[0] = img
    img_pred = img_pred[:, [2,1,0], :, :]

    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        img_pred[:, c, :, :] = img_pred[:, c, :, :] - mean_pixel[c]

    predictions.extend(predict_proba(img_pred))

predictions = np.array(predictions)
print predictions.shape


'''
#Test Time Augmentations

PAD_CROP = 8
PAD_PIXELS = PIXELS + (PAD_CROP * 2)
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

def fast_warp(img, tf, output_shape, mode='constant', cval=0.0):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

print 'Running TTA ... '
tta_iter = 1
for _ in range(20):
    print 'tta count', tta_iter
    # load each image
    path = os.path.join('data', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)
    predictions_tta = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = imread(fl)
        img = transform.resize(img, output_shape=(PIXELS, PIXELS, 3), preserve_range=True)
        img = img.transpose(2, 0, 1)
        img = img.astype('float32')
        X_test_id.append(flbase)

        # pad and crop settings
        trans_1 = random.randint(0, (PAD_CROP*2))
        trans_2 = random.randint(0, (PAD_CROP*2))
        crop_x1 = trans_1
        crop_x2 = (PIXELS + trans_1)
        crop_y1 = trans_2
        crop_y2 = (PIXELS + trans_2)

        # random zooms
        zoom = random.uniform(0.95, 1.05)

        # shearing
        shear_deg = random.uniform(-5,5)

        # random rotations betweein -10 and 10 degrees
        dorotate = random.randint(-10,10)

        # set the transform parameters for skimage.transform.warp
        # have to shift to center and then shift back after transformation otherwise
        # rotations will make image go out of frame
        center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center   = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_aug = transform.AffineTransform(shear = np.deg2rad(shear_deg),
                                              scale = (1/zoom, 1/zoom),
                                              rotation = np.deg2rad(dorotate))

        tform = tform_center + tform_aug + tform_uncenter

        img_pred = np.ones(shape=(1,3,224,224), dtype='float32')

        img_pred[0] = img

        # for each image in the batch do the augmentation
        for j in range(img_pred.shape[0]):
            for k in range(img_pred.shape[1]):
                #X_batch_aug[j,k] = fast_warp(X_batch_aug[j,k], tform, output_shape=(PIXELS,PIXELS))
                # pad and crop images
                img_pad = np.pad(img[k], pad_width=((PAD_CROP,PAD_CROP), (PAD_CROP,PAD_CROP)), mode='constant')
                img_pred[j,k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                #if r_intensity == 1:
                #    X_batch_aug[j][0] += intensity_scaler
                #if g_intensity == 1:
                #    X_batch_aug[j][1] += intensity_scaler
                #if b_intensity == 1:
                #    X_batch_aug[j][2] += intensity_scaler

                # adjust brightness
                #X_batch_aug[j] = X_batch_aug[j] * bright

        # print statements for debugging post augmentation
        #img_max =  np.amax(X_batch_aug)
        #plot_sample(X_batch_aug[0] / img_max)

        # fit model on each batch
        predictions_tta.extend(predict_proba(img_pred))

    X_test_id = np.array(X_test_id)
    np.save('data/cache/X_test_id_%d_f32.npy'%PIXELS, X_test_id)

    predictions_tta = np.array(predictions_tta)
    np.save('data/tta_temp/predictions_tta_' + str(tta_iter) + '.npy', predictions_tta)

    tta_iter += 1

# load data
#X_test_id = np.load('data/cache/X_test_id_128_f32.npy')

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


predictions = (tta_sub_1 + tta_sub_2 + tta_sub_3 + tta_sub_4 + tta_sub_5 + tta_sub_6 + tta_sub_7 + tta_sub_8 + tta_sub_9 + tta_sub_10) / 10.0
'''
test_id = np.load('data/cache/X_test_id_%d_f32.npy'%PIXELS)

'''
Make submission file
'''
print 'Generating submission for', str(experiment_label)


def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', '%s_finetune.csv'%experiment_label)
    result1.to_csv(sub_file, index=False)

create_submission(predictions, test_id)
