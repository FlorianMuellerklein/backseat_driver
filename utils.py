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
from skimage import transform, filters, exposure

import theano
from theano import tensor as T

PIXELS = 64
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

tsne = TSNE(verbose=1)

def load_train_cv(encoder, cache=False):
    if cache:
        X_train = np.load('data/cache/X_train_small.npy')
        y_train = np.load('data/cache/y_train_small.npy')
    else:
        X_train = []
        y_train = []
        print('Read train images')
        for j in range(10):
            print('Load folder c{}'.format(j))
            path = os.path.join('data', 'imgs', 'train', 'c' + str(j), '*.jpg')
            files = glob.glob(path)
            for fl in files:
                print(fl)
                img = imread(fl)
                img = transform.resize(img, output_shape=(PIXELS, PIXELS, 3))
                img = img.transpose(2, 0, 1)
                img = np.reshape(img, (1, num_features))
                X_train.append(img)
                y_train.append(j)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        np.save('data/cache/X_train_small.npy', X_train)
        np.save('data/cache/y_train_small.npy', y_train)

        plot_sample(X_train[0])

    y_train = encoder.fit_transform(y_train).astype('int32')

    X_train, y_train = shuffle(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    X_train = X_train.reshape(X_train.shape[0], 3, PIXELS, PIXELS).astype('float32') #/ 255.
    X_test = X_test.reshape(X_test.shape[0], 3, PIXELS, PIXELS).astype('float32') #/ 255.

    return X_train, y_train, X_test, y_test, encoder

def load_test(cache=False):
    if cache:
        X_test = np.load('data/cache/X_test_small.npy')
        X_test_id = np.load('data/cache/X_test_id_small.npy')
    else:
        print('Read test images')
        path = os.path.join('data', 'imgs', 'test', '*.jpg')
        files = glob.glob(path)
        X_test = []
        X_test_id = []
        total = 0
        thr = math.floor(len(files)/10)
        for fl in files:
            print(fl)
            flbase = os.path.basename(fl)
            img = imread(fl)
            img = transform.resize(img, output_shape=(PIXELS, PIXELS, 3))
            img = img.transpose(2, 0, 1)
            img = np.reshape(img, (1, num_features))
            X_test.append(img)
            X_test_id.append(flbase)
            total += 1
            if total%thr == 0:
                print('Read {} images from {}'.format(total, len(files)))

        X_test = np.array(X_test)
        X_test_id = np.array(X_test_id)

        np.save('data/cache/X_test_small.npy', X_test)
        np.save('data/cache/X_test_id_small.npy', X_test_id)

    X_test = X_test.reshape(X_test.shape[0], 3, PIXELS, PIXELS).astype('float32')

    return X_test, X_test_id

def plot_sample(img):
    img = img.reshape(PIXELS, PIXELS, 3)
    imshow(img)
    #img = img / 290.
    pyplot.show(block=True)

def fast_warp(img, tf, output_shape, mode='nearest'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

def batch_iterator_train(data, y, batchsize, train_fn, leftright=True):
    '''
    Data augmentation batch iterator for feeding images into CNN.
    This example will randomly rotate all images in a given batch between -30 and 30 degrees
    and to random translations between -24 and 24 pixels in all directions.
    Random zooms between 1 and 1.3.
    Random shearing between -10 and 10 degrees.
    '''
    n_samples = data.shape[0]
    data, y = shuffle(data, y)
    loss = []
    acc_train = 0.
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = data[sl]
        y_batch = y[sl]

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

        # random clips bool
        if leftright:
            flip_lr = random.randint(0,1)
        else:
            flip_lr = 0
        #flip_ud = random.randint(0,1)

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
                if flip_lr == 1:
                    X_batch_aug[j,k] = np.fliplr(X_batch[j,k])
                #if flip_ud == 1:
                #    X_batch_aug[j,k] = np.flipud(X_batch[j,k])

                #img_crop = crop(X_batch_aug[j,k], crop_amt)
                #X_batch_aug[j,k] = transform.resize(img_crop, output_shape=(PIXELS, PIXELS))

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
        loss.append(train_fn(X_batch_aug, y_batch))

    return np.mean(loss)

def batch_iterator_valid(data_test, y_test, batchsize, valid_fn):
    '''
    Batch iterator for fine tuning network, no augmentation.
    '''
    n_samples_valid = data_test.shape[0]
    loss_valid = []
    acc_valid = 0.
    for i in range((n_samples_valid + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch_test = data_test[sl]
        y_batch_test = y_test[sl]

        loss_vv, acc_vv = valid_fn(X_batch_test, y_batch_test)
        loss_valid.append(loss_vv)
        acc_valid += acc_vv

    # print statements for debugging, check if test images look like train images
    #plot_sample((X_batch_test[0] / np.amax(X_batch_test)))

    return np.mean(loss_valid), acc_valid / n_samples_valid
