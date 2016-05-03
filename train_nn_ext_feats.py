import gzip
import time
import pickle
import numpy as np
import pandas as pd

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum, adam
from lasagne.layers import helper, DenseLayer, InputLayer, DropoutLayer, batch_norm, NonlinearityLayer
from lasagne.init import Orthogonal, HeNormal, GlorotNormal
from lasagne.nonlinearities import softmax, rectify

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from utils import batch_iterator_train_noaug, batch_iterator_valid

from matplotlib import pyplot

# training params
ITERS = 60
BATCHSIZE = 32
LR_SCHEDULE = {
    0: 0.0001,
    30: 0.00001
}

'''
Set up Network
'''
ortho = Orthogonal(gain='relu')
he_norm = HeNormal(gain='relu')

def build_model(input_var=None):
    net = {}
    l_input = InputLayer((None, 8192), input_var=input_var)
    l_dropout1 = DropoutLayer(l_input, p=0.5)

    l_hidden1 = batch_norm(DenseLayer(l_dropout1, W=he_norm, num_units=1024, nonlinearity=rectify))
    l_dropout2 = DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = batch_norm(DenseLayer(l_dropout2, W=he_norm, num_units=1024, nonlinearity=rectify))
    l_dropout3 = DropoutLayer(l_hidden2, p=0.5)

    l_output = DenseLayer(l_dropout3, num_units=10, W=HeNormal(), nonlinearity=softmax)
    return l_output

'''
Set up Theano functions
'''

X = T.fmatrix('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
# load model
output_layer = build_model(X)

# create outputs
output_train = lasagne.layers.get_output(output_layer)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up the loss that we aim to minimize when using cat cross entropy our Y should be ints not one-hot
loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
loss = loss.mean()

# set up loss functions for validation dataset
test_loss = lasagne.objectives.categorical_crossentropy(output_test, Y)
test_loss = test_loss.mean()

test_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y), dtype=theano.config.floatX)

# get parameters from network and set up sgd with nesterov momentum to update parameters, l_r is shared var so it can be changed
l_r = theano.shared(np.array(LR_SCHEDULE[0], dtype=theano.config.floatX))
params = lasagne.layers.get_all_params(output_layer, trainable=True)
#updates = nesterov_momentum(loss, params, learning_rate=l_r, momentum=0.9)
updates = adam(loss, params, learning_rate=l_r)

# set up training and prediction functions
train_fn = theano.function(inputs=[X,Y], outputs=loss, updates=updates)
valid_fn = theano.function(inputs=[X,Y], outputs=[test_loss, test_acc])

'''
load training data
'''
encoder = LabelEncoder()

# load the training and validation data sets
X_train = np.load('data/cache/vgg16_features_train.npy')
y_train = np.load('data/cache/y_train_128_f32.npy')

y_train = encoder.fit_transform(y_train).astype('int32')

X_train, y_train = shuffle(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

'''
Start training
'''
def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]

# loop over training functions for however many iterations, print information while training
best_acc = 0.0
best_vl = 1.0
try:
    for epoch in range(ITERS):
        if epoch in LR_SCHEDULE:
            l_r.set_value(LR_SCHEDULE[epoch])
        # do the training
        start = time.time()
        # training batches
        train_loss = []
        for batch in iterate_minibatches(X_train, y_train, BATCHSIZE):
            inputs, targets = batch
            train_loss.append(train_fn(inputs, targets))
        train_loss = np.mean(train_loss)
        # validation batches
        valid_loss = []
        valid_acc = []
        for batch in iterate_minibatches(X_test, y_test, BATCHSIZE):
            inputs, targets = batch
            valid_eval = valid_fn(inputs, targets)
            valid_loss.append(valid_eval[0])
            valid_acc.append(valid_eval[1])
        valid_loss = np.mean(valid_loss)
        valid_acc = np.mean(valid_acc)
        # get ratio of TL to VL
        ratio = train_loss / valid_loss
        end = time.time() - start
        # print training details
        print 'iter:', epoch, '| TL:', np.round(train_loss,decimals=3), '| VL:', np.round(valid_loss, decimals=3), '| Vacc:', np.round(valid_acc, decimals=3), '| Ratio:', np.round(ratio, decimals=2), '| Time:', np.round(end, decimals=1)

        if valid_loss < best_vl:
            best_vl = valid_loss
            best_params = helper.get_all_param_values(output_layer)

except KeyboardInterrupt:
    pass

print "Best Valid Loss:", best_vl

# save weights
f = gzip.open('data/weights/vgg16_features_mlp.pklz', 'wb')
pickle.dump(best_params, f)
f.close()

'''
Set up all prediction theano functions
'''
output_test = lasagne.layers.get_output(output_layer, deterministic=True)
predict_proba = theano.function(inputs=[X], outputs=output_test)

'''
Load Test data
'''
X_test_final = np.load('data/cache/vgg16_features_test.npy')
X_test_id = np.load('data/cache/X_test_id_128_f32.npy')

'''
Make prediction
'''
#make predictions
print 'Making predictions ... '
PRED_BATCH = 32
def iterate_pred_minibatches(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

predictions = []
for pred_batch in iterate_pred_minibatches(X_test_final, PRED_BATCH):
    predictions.extend(predict_proba(pred_batch))

predictions = np.array(predictions)

print 'pred shape'
print predictions.shape

print 'Creating Submission ... '
def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1.to_csv('subm/submission_vgg16_features.csv', index=False)

create_submission(predictions, X_test_id)
