import os
import gzip
import time
import pickle
import random
import datetime
import numpy as np
import pandas as pd

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum, adam
from lasagne.layers import helper

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE

from models import vgg16, ResNet_Orig, ResNet_FullPre, ResNet_BttlNck_FullPre
from utils import load_test, batch_iterator_train, batch_iterator_valid

from matplotlib import pyplot

# testing params
BATCHSIZE = 1

'''
Set up all theano functions
'''
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = ResNet_FullPre(X, n=5)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up training and prediction functions
predict_proba = theano.function(inputs=[X], outputs=output_test)

'''
Load data and make predictions
'''
# load data
X_test, X_test_id = load_test(cache=True)

nn_count = 1
for ensb in range(19):
    # load network weights
    f = gzip.open('data/weights/resnet32_fullpre_' + str(nn_count) + '.pklz', 'rb')
    all_params = pickle.load(f)
    f.close()
    helper.set_all_param_values(output_layer, all_params)

    '''
    # make regular predictions
    predictions = []
    for j in range((X_test.shape[0] + BATCHSIZE - 1) // BATCHSIZE):
        sl = slice(j * BATCHSIZE, (j + 1) * BATCHSIZE)
        X_batch = X_test[sl]
        predictions.extend(predict_proba(X_batch))

    predictions = np.array(predictions)
    '''

    #Test Time Augmentations

    PIXELS = 64
    PAD_CROP = 4
    PAD_PIXELS = PIXELS + (PAD_CROP * 2)
    imageSize = PIXELS * PIXELS
    num_features = imageSize * 3

    tta_iter = 1
    for _ in range(5):
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

            # set empty copy to hold augmented images so that we don't overwrite
            X_batch_aug = np.copy(X_batch)

            # for each image in the batch do the augmentation
            for j in range(X_batch.shape[0]):
                # for each image channel
                for k in range(X_batch.shape[1]):
                    # pad and crop images
                    img_pad = np.pad(X_batch_aug[j,k], pad_width=((PAD_CROP,PAD_CROP), (PAD_CROP,PAD_CROP)), mode='constant')
                    X_batch_aug[j,k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]


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

    print('Prediction ' + str(nn_count) + ' done, Writing file ... ')

    '''
    make submission file
    '''
    def create_submission(predictions, test_id):
        result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
        result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
        now = datetime.datetime.now()
        if not os.path.isdir('subm'):
            os.mkdir('subm')
        suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
        sub_file = os.path.join('subm', 'resnet32_fullpre_' + str(nn_count) + '_tta.csv')
        result1.to_csv(sub_file, index=False)

    create_submission(predictions, X_test_id)

    nn_count += 1
