#!/usr/bin/env python

import os
import glob
import numpy as np
import argparse
import gzip
import cPickle as pickle

from sklearn.cross_validation import KFold
from skimage import transform
from skimage.io import imshow, imsave, imread
from sklearn.utils import shuffle

'''
    Instead of using filenames, use indices
    that way we can just read one data cache, 
    and then separate it into train/test
    
    filenames are stored just for sanity's sake
'''

import argparsing
args, unknown_args = argparsing.parse_args()

PIXELS = args.pixels

IMG_FOLDER = os.path.join('data', 'imgs')
RANDOM_STATE = 20


def load_cv_fold(encoder, fold_idx=0):
    global IMG_FOLDER
    X_data_filename = 'data/cache/X_data_%d.npy'%(PIXELS)
    y_data_filename = 'data/cache/y_data_%d.npy'%(PIXELS)   

    X_data = np.load(X_data_filename)
    y_data = np.load(y_data_filename)
    
    with gzip.open(os.path.join('cv_folds.pklz')) as f:
        cv_folds = pickle.load(f)
            
    X_train = X_data[..., cv_folds[fold_idx]['train']]
    y_train = y_data[..., cv_folds[fold_idx]['train']]
    
    X_test = X_data[..., cv_folds[fold_idx]['test']]
    y_test = y_data[..., cv_folds[fold_idx]['test']]
    
    y_train = encoder.fit_transform(y_train).astype('int32')
    y_test = encoder.fit_transform(y_test).astype('int32')

    X_train = X_train.transpose(3, 2, 0, 1) / 255.
    X_test = X_test.transpose(3, 2, 0, 1) / 255.

    X_train, y_train = shuffle(X_train, y_train)
    
    return X_train, y_train, X_test, y_test, encoder

def get_driver_data():
    dr = dict()
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

def get_driver_indices():
    global IMG_FOLDER
    driver_id = []
    filenames = []
    
    driver_data = get_driver_data()

    print('Read training driver ids')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join(IMG_FOLDER, 'train', 'c' + str(j), '*.jpg')
        files = sorted(glob.glob(path))
        for fl in files:
            flbase = os.path.basename(fl)
            filenames.append(fl)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    
    # For easier indexing
    driver_id = np.array(driver_id)
    filenames = np.array(filenames)
    
    
    driver_indices = {}
    for driver in unique_drivers:
        driver_indices[driver] = np.where(driver_id == driver)[0]
    
    counts = {}
    for driver,indices in driver_indices.iteritems():
        counts[driver] = len(indices)
        
    print counts    
    
    return unique_drivers, driver_indices, filenames

# TODO: check to make sure there are no duplicates
def create_cv_fold_yamls(nfolds=10):
    global RANDOM_STATE
    global IMG_FOLDER
    
    unique_drivers, driver_indices, filenames = get_driver_indices()
    
    kf = KFold(len(driver_indices.keys()), n_folds=nfolds,
               shuffle=True, random_state=RANDOM_STATE)
    
    cv_folds = {}
    for fold_idx, (train_drivers_indices, test_drivers_indices) in enumerate(kf):
        cv_folds[fold_idx] = {
                              'train': [],
                              'test': [],
                              'train_filenames': [],
                              'test_filenames': [],
                              }
        
        for driver_idx in train_drivers_indices:
            np_list = driver_indices[unique_drivers[driver_idx]]
            cv_folds[fold_idx]['train'].extend(np_list.tolist())
            cv_folds[fold_idx]['train_filenames'].extend(filenames[np_list].tolist())
            
        for driver_idx in test_drivers_indices:
            np_list = driver_indices[unique_drivers[driver_idx]]
            cv_folds[fold_idx]['test'].extend(np_list.tolist())
            cv_folds[fold_idx]['test_filenames'].extend(filenames[np_list].tolist())
                    
    with gzip.open(os.path.join('cv_folds.pklz'), 'w') as f:
        pickle.dump(cv_folds, f)
           
def create_cv_cache(args):
    global IMG_FOLDER
    PIXELS = args.pixels
    channels = 3
    
    path = os.path.join(IMG_FOLDER, 'train', 'c*', '*.jpg')
    total_files = len(glob.glob(path))

    # preallocate to avoid double memory usage
    X_data = np.empty((PIXELS, PIXELS, channels, total_files), dtype=np.float32)
    y_data = np.empty(total_files, dtype=np.float32)

    print('Read training images')
    
    count = 0
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join(IMG_FOLDER, 'train', 'c' + str(j), '*.jpg')
        files = sorted(glob.glob(path))
        
        
        for idx, fl in enumerate(files):
            if idx%100 == 0:
                print('%d of %d'%(idx, len(files)))
                
            img = imread(fl)
            X_data[..., count] = transform.resize(img, output_shape=(PIXELS, PIXELS, channels), preserve_range=True)
            y_data[count] = j
            
            count += 1
                        
            
    np.save('data/cache/X_data_%d.npy'%(PIXELS), X_data)
    np.save('data/cache/y_data_%d.npy'%(PIXELS), y_data)
        
def get_label(fl): 
    return int(fl.split(os.sep)[-2][1:])

def get_distribution():
    unique_drivers, driver_indices, filenames = get_driver_indices()
    counts = {}

    for driver in unique_drivers:
        np_list = driver_indices[driver]
        for filename in filenames[np_list].tolist():
            if driver not in counts:
                counts[driver] = {k:0 for k in range(10)}
                
            label = get_label(filename)
            counts[driver][label] += 1
            
    return counts
            
            
if __name__ == '__main__':    
    #import ipdb; ipdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_yamls', action='store_true', help='create cv yamls')
    parser.add_argument('--create_cache', action='store_true', help='create cv cache')
    parser.add_argument('-p', '--pixels', type=int, default=64, help='pixels')
    args_file = parser.parse_args(unknown_args)
    
    if args_file.create_yamls:
        create_cv_fold_yamls()
        
    if args_file.create_cache:
        create_cv_cache(args)
    

            


