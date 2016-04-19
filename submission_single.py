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

from models import vgg16, ResNet_Orig, ResNet_FullPreActivation
from utils import load_test, batch_iterator_train, batch_iterator_valid

from matplotlib import pyplot

# testing params
BATCHSIZE = 32
PIXELS = 64
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
output_layer = ResNet_FullPreActivation(X)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up training and prediction functions
predict_proba = theano.function(inputs=[X], outputs=output_test)

'''
Load data and make predictions
'''
# load data
X_test, X_test_id = load_test(cache=True)

# load network weights
f = gzip.open('data/weights/weights_resnet110_16ch_relabel.pklz', 'rb')
all_params = pickle.load(f)
f.close()
helper.set_all_param_values(output_layer, all_params)

#make predictions
predictions = []
for j in range((X_test.shape[0] + BATCHSIZE - 1) // BATCHSIZE):
    sl = slice(j * BATCHSIZE, (j + 1) * BATCHSIZE)
    X_batch = X_test[sl]
    predictions.extend(predict_proba(X_batch))

predictions = np.array(predictions)
print predictions.shape

''' Test Time Augmentations
tta_iter = 1
for _ in range(5):
    predictions_tta = []
    for i in range((X_test.shape[0] + BATCHSIZE - 1) // BATCHSIZE):
        sl = slice(i * BATCHSIZE, (i + 1) * BATCHSIZE)
        X_batch = X_test[sl]

        # random rotations betweein -8 and 8 degrees
        dorotate = random.randint(-15,15)

        # random translations
        trans_1 = random.randint(-6,6)
        trans_2 = random.randint(-6,6)
        crop_amt = ((4 - trans_1, 4 + trans_1), (4 - trans_2, 4 + trans_2))

        # random zooms
        zoom = random.uniform(0.8, 1.2)

        # shearing
        shear_deg = random.uniform(-5,5)

        # set the transform parameters for skimage.transform.warp
        # have to shift to center and then shift back after transformation otherwise
        # rotations will make image go out of frame
        center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
        tform_center   = transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = transform.SimilarityTransform(translation=center_shift)

        tform_aug = transform.AffineTransform(translation = (trans_1, trans_2),
                                              shear = np.deg2rad(shear_deg),
                                              scale = (1/zoom, 1/zoom),
                                              rotation = np.deg2rad(dorotate))

        tform = tform_center + tform_aug + tform_uncenter

        r_intensity = random.randint(0,1)
        g_intensity = random.randint(0,1)
        b_intensity = random.randint(0,1)
        intensity_scaler = random.randint(-25, 25) / 255.

        # print statements for debugging pre-augmentation
        #print X_batch.shape
        #plot_sample(X_batch[0])

        # set empty copy to hold augmented images so that we don't overwrite
        X_batch_aug = np.copy(X_batch)

        # for each image in the batch do the augmentationp
        for j in range(X_batch.shape[0]):
            # for each image channel
            for k in range(X_batch.shape[1]):
                X_batch_aug[j,k] = fast_warp(X_batch[j,k], tform, output_shape=(PIXELS,PIXELS))

            if r_intensity == 1:
                X_batch_aug[j][0] = X_batch_aug[j][0] + intensity_scaler
            if g_intensity == 1:
                X_batch_aug[j][1] = X_batch_aug[j][1] + intensity_scaler
            if b_intensity == 1:
                X_batch_aug[j][2] = X_batch_aug[j][2] + intensity_scaler

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

predictions = (tta_sub_1 + tta_sub_2 + tta_sub_3 + tta_sub_4 + tta_sub_5) / 5.0
'''

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
    sub_file = os.path.join('subm', 'submission_resnet110_16ch_relabel.csv')
    result1.to_csv(sub_file, index=False)

create_submission(predictions, X_test_id)
