'''
Tweaked from https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/ImageNet%20Pretrained%20Network%20(ResNet-50).ipynb

To use with the State Farm Distracted Driver Competition

Need to download Caffe files form https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777
And place them in 'caffe_resnet' folder within script directory
'''

import caffe

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer # can be replaced with dnn layers
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax

import numpy as np
import matplotlib.pyplot as plt

import io
import urllib
import skimage.transform
from IPython.display import Image
import pickle

import argparsing
args, unknown_args = argparsing.parse_args()

# training params
experiment_label = args.label
PIXELS = args.pixels
ITERS = args.epochs
BATCHSIZE = args.batchsize

LR_SCHEDULE = {
    0: 0.001,
    10: 0.0001,
    20: 0.00001
}

PIXELS = 224
imageSize = PIXELS * PIXELS
num_features = imageSize

'''
Functions to create residual blocks that mimic the Caffe ones.
'''
def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    names : list of string
        Names of the layers in block

    num_filters : int
        Number of filters in convolution layer

    filter_size : int
        Size of filters in convolution layer

    stride : int
        Stride of convolution layer

    pad : int
        Padding of convolution layer

    use_bias : bool
        Whether to use bias in conlovution layer

    nonlin : function
        Nonlinearity type of Nonlinearity layer

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]

simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block

    ratio_size : float
        Scale factor of filter size

    has_left_branch : bool
        if True, then left branch contains simple block

    upscale_factor : float
        Scale factor of filter bank at the output of residual block

    ix : int
        Id of residual block

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix

# Build the Network

net = {}
net['input'] = InputLayer((None, 3, 224, 224))
sub_net, parent_layer_name = build_simple_block(
    net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
    64, 7, 3, 2, use_bias=True)
net.update(sub_net)
net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)

block_size = list('abc')
parent_layer_name = 'pool1'
for c in block_size:
    if c == 'a':
        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
    else:
        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
    net.update(sub_net)

block_size = list('abcd')
for c in block_size:
    if c == 'a':
        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
    else:
        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
    net.update(sub_net)

block_size = list('abcdef')
for c in block_size:
    if c == 'a':
        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
    else:
        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
    net.update(sub_net)

block_size = list('abc')
for c in block_size:
    if c == 'a':
        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
    else:
        sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
    net.update(sub_net)


net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                         mode='average_exc_pad', ignore_border=False)
net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)

print 'Total number of layers:', len(lasagne.layers.get_all_layers(net['prob']))

# load the caffe weights
net_caffe = caffe.Net('caffe_resnet/ResNet-50-deploy.prototxt', 'caffe_resnet/ResNet-50-model.caffemodel', caffe.TEST)
layers_caffe = dict(zip(list(net_caffe._layer_names), net_caffe.layers))
print 'Number of layers: %i' % len(layers_caffe.keys())

# copy the caffe weights to the lasagne ResNet-50

for name, layer in net.items():
    if name not in layers_caffe:
        print name, type(layer).__name__
        continue
    if isinstance(layer, BatchNormLayer):
        layer_bn_caffe = layers_caffe[name]
        layer_scale_caffe = layers_caffe['scale' + name[2:]]
        layer.gamma.set_value(layer_scale_caffe.blobs[0].data)
        layer.beta.set_value(layer_scale_caffe.blobs[1].data)
        layer.mean.set_value(layer_bn_caffe.blobs[0].data)
        layer.inv_std.set_value(1/np.sqrt(layer_bn_caffe.blobs[1].data) + 1e-4)
        continue
    if isinstance(layer, DenseLayer):
        layer.W.set_value(layers_caffe[name].blobs[0].data.T)
        layer.b.set_value(layers_caffe[name].blobs[1].data)
        continue
    if len(layers_caffe[name].blobs) > 0:
        layer.W.set_value(layers_caffe[name].blobs[0].data)
    if len(layers_caffe[name].blobs) > 1:
        layer.b.set_value(layers_caffe[name].blobs[1].data)

'''
Set up Theano Funcitons for FineTuning
'''
X = T.tensor4('X')
Y = T.ivector('y')

# stack our own softmax onto the final layer
output_layer = DenseLayer(net['pool5'], num_units=10, W=lasagne.init.HeNormal(), nonlinearity=softmax)

# standard output functions
output_train = lasagne.layers.get_output(output_layer)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

# set up the loss that we aim to minimize, when using cat cross entropy our Y should be ints not one-hot
loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
loss = loss.mean()

# set up loss functions for validation dataset
valid_loss = lasagne.objectives.categorical_crossentropy(output_test, Y)
valid_loss = valid_loss.mean()

valid_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y), dtype=theano.config.floatX)

# get parameters from network and set up sgd with nesterov momentum to update parameters
l_r = theano.shared(np.array(LR_SCHEDULE[0], dtype=theano.config.floatX))
params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = nesterov_momentum(loss, params, learning_rate=l_r)

# set up training and prediction functions
train_fn = theano.function(inputs=[X,Y], outputs=loss, updates=updates)
valid_fn = theano.function(inputs=[X,Y], outputs=[valid_loss, valid_acc])

# set up prediction function
predict_proba = theano.function(inputs=[X], outputs=output_test)


'''
load training data and start training
'''
encoder = LabelEncoder()

train_X, train_y, test_X, test_y, encoder = load_train_cv(encoder, cache=True, relabel=False)
print 'Train shape:', train_X.shape, 'Test shape:', test_X.shape
print 'Train y shape:', train_y.shape, 'Test y shape:', test_y.shape
print np.amax(train_X), np.amin(train_X), np.mean(train_X)

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

        valid_loss, acc_v = batch_iterator_valid(test_X, test_y, BATCHSIZE, valid_fn)
        valid_eval.append(valid_loss)
        valid_acc.append(acc_v)

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
f = gzip.open('data/weights/%s_best.pklz'%experiment_label, 'wb')
pickle.dump(best_params, f)
f.close()

last_params = helper.get_all_param_values(output_layer)
f = gzip.open('data/weights/%s_last.pklz'%experiment_label, 'wb')
pickle.dump(last_params, f)
f.close()
