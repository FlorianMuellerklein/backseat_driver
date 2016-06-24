import gzip
import time
import pickle
import numpy as np

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum, adam
from lasagne.layers import helper, DenseLayer
from lasagne.nonlinearities import softmax

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from models import ST_ResNet_FullPre, ResNet_FullPre, ResNet_FullPre_Wide
from utils import load_train_cv, batch_iterator_train_pseudo_label, batch_iterator_valid, load_pseudo
from crossvalidation import load_cv_fold

from matplotlib import pyplot
import warnings
warnings.filterwarnings("ignore")

import argparsing
args, unknown_args = argparsing.parse_args()

# training params
experiment_label = args.label
PIXELS = args.pixels
ITERS = args.epochs
BATCHSIZE = args.batchsize

LR_SCHEDULE = {
    0: 0.01,
    120: 0.001,
    180: 0.0001
}

#encoder = LabelEncoder()
encoder = LabelBinarizer()

"""
Set up all theano functions
"""
X = T.tensor4('X')
Y = T.fmatrix('y') # use this for pseudo-label

# custom log loss for pseudo label soft-targets, takes vector as target input instead of label
def pseudo_log_loss(pred, y, eps=1e-15):
    '''
    pred: predictions
    y: true value, training targets will be one hot, testing-pseudo will be soft-targets
    '''
    pred = T.clip(pred, eps, 1 - eps)
    losses = -T.sum(y * T.log(pred), axis=1)
    return losses

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
# load model
output_layer = ResNet_FullPre_Wide(X, n=5, k=3)

# create outputs
output_train = lasagne.layers.get_output(output_layer)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up the loss that we aim to minimize when using cat cross entropy our Y should be ints not one-hot
#loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
loss = pseudo_log_loss(output_train, Y)
loss = loss.mean()

# if using ResNet use L2 regularization
all_layers = lasagne.layers.get_all_layers(output_layer)
l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
loss = loss + l2_penalty

# set up loss functions for validation dataset
#test_loss = lasagne.objectives.categorical_crossentropy(output_test, Y)
test_loss = pseudo_log_loss(output_test, Y)
test_loss = test_loss.mean()

#test_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y), dtype=theano.config.floatX)
test_acc = T.mean(T.eq(T.argmax(output_test, axis=1), T.argmax(Y, axis=1)), dtype=theano.config.floatX)

# get parameters from network and set up sgd with nesterov momentum to update parameters, l_r is shared var so it can be changed
l_r = theano.shared(np.array(LR_SCHEDULE[0], dtype=theano.config.floatX))
params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = nesterov_momentum(loss, params, learning_rate=l_r, momentum=0.9)
#updates = adam(loss, params, learning_rate=l_r)

# set up training and prediction functions
train_fn = theano.function(inputs=[X,Y], outputs=loss, updates=updates)
valid_fn = theano.function(inputs=[X,Y], outputs=[test_loss, test_acc])

'''
load training data and start training
'''

# load the training and validation data sets
train_X, train_y, test_X, test_y, encoder = load_train_cv(encoder, cache=True, relabel=False)
train_y = train_y.astype('float32')
test_y = test_y.astype('float32')
pseudo_X, pseudo_labels = load_pseudo(cache=True)
pseudo_labels = pseudo_labels.astype('float32')
print 'Train shape:', train_X.shape, 'Test shape:', test_X.shape
print 'Train y shape:', train_y.shape, 'Test y shape:', test_y.shape
print 'Pseudo X shape:', pseudo_X.shape, 'pseudo y shape:', pseudo_labels.shape
print np.amax(train_X), np.amin(train_X), np.mean(train_X)

# loop over training functions for however many iterations, print information while training
train_eval = []
valid_eval = []
valid_acc = []
best_vl = 3.0
try:
    for epoch in range(ITERS):
        # change learning rate according to schedules
        if epoch in LR_SCHEDULE:
            l_r.set_value(LR_SCHEDULE[epoch])
        # do the training
        start = time.time()

        train_loss = batch_iterator_train_pseudo_label(train_X, train_y, pseudo_X, pseudo_labels, BATCHSIZE, train_fn)
        train_eval.append(train_loss)

        valid_loss, acc_v = batch_iterator_valid(test_X, test_y, BATCHSIZE, valid_fn)
        valid_eval.append(valid_loss)
        valid_acc.append(acc_v)

        ratio = train_loss / valid_loss
        end = time.time() - start
        # print training details
        print 'iter:', epoch, '| TL:', np.round(train_loss,decimals=3), '| VL:', np.round(valid_loss,decimals=3), '| Vacc:', np.round(acc_v,decimals=3), '| Ratio:', np.round(ratio,decimals=2), '| Time:', np.round(end,decimals=1)

        if valid_loss < best_vl:
            best_vl = valid_loss
            best_params = helper.get_all_param_values(output_layer)

except KeyboardInterrupt:
    pass

print "Best Valid Loss:", best_vl

# save weights
f = gzip.open('data/weights/%s_best.pklz'%experiment_label, 'wb')
pickle.dump(best_params, f)
f.close()

last_params = helper.get_all_param_values(output_layer)
f = gzip.open('data/weights/%s_last.pklz'%experiment_label, 'wb')
pickle.dump(last_params, f)
f.close()

# plot loss and accuracy
train_eval = np.array(train_eval)
valid_eval = np.array(valid_eval)
valid_acc = np.array(valid_acc)
pyplot.plot(train_eval, label='Train loss', color='#707070')
pyplot.plot(valid_eval, label='Valid loss', color='#3B91CF')
pyplot.ylabel('Categorical Cross Entropy Loss')
pyplot.xlabel('Epoch')
pyplot.legend(loc=2)
pyplot.ylim([0,1.5])
pyplot.twinx()
pyplot.ylabel('Valid Acc (%)')
pyplot.grid()
pyplot.plot(valid_acc, label='Valid classification accuracy (%)', color='#ED5724')
pyplot.legend(loc=1)
pyplot.savefig('plots/%s.png'%experiment_label)
pyplot.clf()
#pyplot.show()
