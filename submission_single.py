import os
import gzip
import time
import pickle
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

from models import vgg16, ResNet34
from utils import load_test, batch_iterator_train, batch_iterator_valid, iterate_minibatches

from matplotlib import pyplot

# testing params
BATCHSIZE = 1

'''
Set up all theano functions
'''
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = ResNet34(X)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up training and prediction functions
predict_proba = theano.function(inputs=[X], outputs=output_test)

'''
Load data and make predictions
'''
# load data
X_test, X_test_id = load_test(cache=True)


# load network weights
f = gzip.open('data/weights/weights_resnet56_flipud.pklz', 'rb')
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
    sub_file = os.path.join('subm', 'submission_resnet56_flipud.csv')
    result1.to_csv(sub_file, index=False)

create_submission(predictions, X_test_id)
