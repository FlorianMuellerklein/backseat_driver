import os
import numpy as np
import pandas as pd

from random import randint, uniform

from skimage.util import crop
from skimage.io import imshow, imread, imsave
from skimage import transform, filters, exposure

PIXELS = 128
cropPIXELS = 112

def plot_sample(img):
    pyplot.gray()
    imshow(img)
    pyplot.show()

def fast_warp(img, tf, output_shape, mode='nearest'):
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
    # random rotations betweein -8 and 8 degrees
    dorotate = randint(-20,20)

    # random translations
    trans_1 = randint(-8,8)
    trans_2 = randint(-8,8)
    crop_amt = ((8 - trans_1, 8 + trans_1), (8 - trans_2, 8 + trans_2), (0,0))

    # random zooms
    zoom = uniform(1, 1.3)

    # shearing
    shear_deg = uniform(-10, 10)

    # set the transform parameters for skimage.transform.warp
    # have to shift to center and then shift back after transformation otherwise
    # rotations will make image go out of frame
    center_shift   = np.array((PIXELS, PIXELS)) / 2. - 0.5
    tform_center   = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)

    tform_aug = transform.AffineTransform(rotation = np.deg2rad(dorotate),
                                          #translation = (trans_1, trans_2),
                                          shear = np.deg2rad(shear_deg),
                                          scale = (1/zoom, 1/zoom))

    tform = tform_center + tform_aug + tform_uncenter

    r_intensity = randint(0,1)
    g_intensity = randint(0,1)
    b_intensity = randint(0,1)
    intensity_scaler = randint(-30, 30) / 255.

    # images in the batch do the augmentation
    #img = crop(img, crop_amt)

    for i in range(img_aug.shape[2]):
        img_aug[:, :, i] = fast_warp(img[:, :, i], tform, output_shape = (PIXELS, PIXELS))

    img_aug = crop(img_aug, crop_amt)

    if r_intensity == 1:
        img_aug[:, :, 0] = img_aug[:, :, 0] + intensity_scaler
    if g_intensity == 1:
        img_aug[:, :, 1] = img_aug[:, :, 1] + intensity_scaler
    if b_intensity == 1:
        img_aug[:, :, 2] = img_aug[:, :, 2] + intensity_scaler

    img_aug[img_aug > 1] = 1

    choice = randint(0,2)
    if choice == 1:
        img_aug = np.fliplr(img_aug)

    return img_aug

img_raw = imread('18.jpg') / 255.

for i in range(0,50):
    img_aug = gen_aug_images(img_raw)
    imsave(('Aug/' + str(i) + '_aug.png'), img_aug)

imsave('orig.png', transform.resize(img_raw, output_shape = (112, 112)))

# make gif
# convert -delay 30 -loop 0 *.png noise.gif

os.chdir('Aug')
os.system('convert -delay 30 -loop 0 *.png noise.gif')
