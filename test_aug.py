import os
import random
import numpy as np
import pandas as pd

from random import randint, uniform

from skimage.util import crop
from skimage.io import imshow, imread, imsave
from skimage import transform, filters, exposure

PIXELS = 64
PAD_CROP = int(PIXELS * 0.125)
PAD_PIXELS = PIXELS + (PAD_CROP * 2)
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

def plot_sample(img):
    pyplot.gray()
    imshow(img)
    pyplot.show()

def fast_warp(img, tf, output_shape, mode='constant'):
    return transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)


def gen_aug_images(img):
    '''
    Data augmentation batch iterator for feeding images into CNN.
    This example will randomly rotate all images in a given batch between -9 and 9 degrees
    and to random translations between -2 and 2 pixels in all directions.
    Be sure to remove part that displays the augmented images, it's only there to check
    that everything works correctly.
    '''

    img_aug = np.zeros((PIXELS, PIXELS, 3))
    # color intensity augmentation
    r_intensity = random.randint(0,1)
    g_intensity = random.randint(0,1)
    b_intensity = random.randint(0,1)
    intensity_scaler = random.randint(-20, 20) / 255.

    # pad and crop settings
    trans_1 = random.randint(0, (PAD_CROP*2))
    trans_2 = random.randint(0, (PAD_CROP*2))
    crop_x1 = trans_1
    crop_x2 = (PIXELS + trans_1)
    crop_y1 = trans_2
    crop_y2 = (PIXELS + trans_2)

    # shearing
    shear_deg = random.uniform(-5,5)

    # random rotations betweein -15 and 15 degrees
    dorotate = random.randint(-15,15)

    # brightness settings
    bright = random.uniform(0.8,1.2)

    # flip left-right choice
    #flip_lr = random.randint(0,1)

    # set the transform parameters for skimage.transform.warp
    # have to shift to center and then shift back after transformation otherwise
    # rotations will make image go out of frame
    center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
    tform_center   = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_aug = transform.AffineTransform(shear = np.deg2rad(shear_deg),
                                          rotation = np.deg2rad(dorotate))

    tform = tform_center + tform_aug + tform_uncenter

    # images in the batch do the augmentation
    #img = crop(img, crop_amt)

    for i in range(img_aug.shape[2]):
        img_aug[:, :, i] = fast_warp(img[:, :, i], tform, output_shape = (PIXELS, PIXELS))
        # pad and crop images
        img_pad = np.pad(img_aug[:,:,i], pad_width=((PAD_CROP,PAD_CROP), (PAD_CROP,PAD_CROP)), mode='constant')
        img_aug[:,:,i] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

    # adjust brightness
    img_aug = img_aug * bright

    if r_intensity == 1:
        img_aug[:,:,0] += intensity_scaler
    if g_intensity == 1:
        img_aug[:,:,1] += intensity_scaler
    if b_intensity == 1:
        img_aug[:,:,2] += intensity_scaler

    img_aug[img_aug > 1] = 1.0

    #choice = randint(0,2)
    #if choice == 1:
    #    img_aug = np.fliplr(img_aug)

    return img_aug

img_raw = imread('data/imgs/train/c0/img_34.jpg')
img_raw = transform.resize(img_raw, output_shape=(PIXELS, PIXELS))

for i in range(0,50):
    img_aug = gen_aug_images(img_raw)
    imsave(('Aug/' + str(i) + '_aug.png'), img_aug)

imsave('orig.png', transform.resize(img_raw, output_shape=(PIXELS, PIXELS)))

# make gif
# convert -delay 30 -loop 0 *.png noise.gif

os.chdir('Aug')
os.system('convert -delay 30 -loop 0 *.png noise.gif')
