import gzip
import time
import pickle
import numpy as np

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum, adam
from lasagne.layers import helper

from sklearn.preprocessing import StandardScaler, LabelEncoder

from models import vgg16, ResNet, ResNet_test
from utils import load_train_cv, batch_iterator_train, batch_iterator_valid

from matplotlib import pyplot

# training params
ITERS = 80
BATCHSIZE = 32
LR_SCHEDULE = {
    0: 0.001,
    25: 0.0001,
    50: 0.00001,
    65: 0.000001
}

encoder = LabelEncoder()

"""
Set up all theano functions
"""
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = ResNet_test(X)
output_train = lasagne.layers.get_output(output_layer)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up the loss that we aim to minimize when using cat cross entropy our Y should be ints not one-hot
loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
loss = loss.mean()

# if using ResNet use L2 regularization
all_layers = lasagne.layers.get_all_layers(output_layer)
l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
loss = loss + l2_penalty

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
train_fn = theano.function(inputs=[X,Y], outputs=[loss, test_acc], updates=updates)
valid_fn = theano.function(inputs=[X,Y], outputs=[test_loss, test_acc])

'''
load training data and start training
'''
encoder = LabelEncoder()

# load the training and validation data sets
train_X, train_y, test_X, test_y, encoder = load_train_cv(encoder, cache=True)
print 'Train shape:', train_X.shape, 'Test shape:', test_X.shape
print 'Train y shape:', train_y.shape, 'Test y shape:', test_y.shape
print np.amax(train_X)

# loop over training functions for however many iterations, print information while training
train_eval = []
valid_eval = []
valid_acc = []
best_acc = 0.0
try:
    for epoch in range(ITERS):
        # change learning rate according to schedules
        if epoch in LR_SCHEDULE:
            l_r.set_value(LR_SCHEDULE[epoch])
        # do the training
        start = time.time()

        train_loss = batch_iterator_train(train_X, train_y, BATCHSIZE, train_fn)
        train_eval.append(train_loss)

        valid_loss, acc_v = batch_iterator_valid(test_X, test_y, valid_fn)
        valid_eval.append(valid_loss)
        valid_acc.append((1.0 - acc_v))

        ratio = train_loss / valid_loss
        end = time.time() - start
        # print training details
        print 'iter:', epoch, '| TL:', np.round(train_loss,decimals=3), '| VL:', np.round(valid_loss,decimals=3), '| Vacc:', np.round(acc_v,decimals=3), '| Ratio:', np.round(ratio,decimals=2), '| Time:', np.round(end,decimals=1)

        if acc_v > best_acc:
            best_acc = acc_v
            best_params = helper.get_all_param_values(output_layer)

except KeyboardInterrupt:
    pass

print "Final Acc:", best_acc

# save weights
all_params = helper.get_all_param_values(output_layer)
f = gzip.open('data/weights/weights_resnet56_32ch.pklz', 'wb')
pickle.dump(best_params, f)
f.close()

# plot loss and accuracy
train_eval = np.array(train_eval)
valid_eval = np.array(valid_eval)
valid_acc = np.array(valid_acc)
pyplot.plot(train_eval, label='Train loss', color='#707070')
pyplot.plot(valid_err, label='Valid loss', color='#3B91CF')
pyplot.ylabel('Categorical Cross Entropy Loss')
pyplot.legend(loc=2)
pyplot.twinx()
ax.yaxis.set_label_position("right")
pyplot.ylabel('Valid Error (%)')
pyplot.plot(valid_acc, label='Valid classification error', color='#ED5724')
pyplot.xlabel('Epoch')
pyplot.legend(loc=3)
pyplot.savefig('plots/training_ResNet56_l2_16ch.png')
pyplot.clf()
#pyplot.show()
