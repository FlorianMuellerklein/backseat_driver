import sys
sys.setrecursionlimit(10000)
import numpy as np

import theano
import lasagne
from lasagne.nonlinearities import rectify, softmax, sigmoid, linear, elu, very_leaky_rectify, leaky_rectify, identity
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper, batch_norm, BatchNormLayer
from lasagne.layers import Conv2DLayer, ConcatLayer, TransformerLayer
# for ResNet
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer, ElemwiseSumLayer, NonlinearityLayer, PadLayer, GlobalPoolLayer, ExpressionLayer
from lasagne.init import Orthogonal, HeNormal, GlorotNormal, HeUniform
# for BLVC Googlenet
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer

import argparsing
args, unknown_args = argparsing.parse_args()

PIXELS = args.pixels

imageSize = PIXELS * PIXELS
num_features = imageSize * 3

ortho = Orthogonal(gain='relu')
he_norm = HeNormal(gain='relu')
xavier_norm = GlorotNormal(gain=1.0)

def vgg16_old(input_var=None):
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    l_conv1a = batch_norm(Conv2DLayer(l_in, num_filters=64, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_conv1b = batch_norm(Conv2DLayer(l_conv1a, num_filters=64, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=2) # feature maps 64x64

    l_conv2a = batch_norm(Conv2DLayer(l_pool1, num_filters=128, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_conv2b = batch_norm(Conv2DLayer(l_conv2a, num_filters=128, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=2) # feature maps 32x32

    l_conv3a = batch_norm(Conv2DLayer(l_pool2, num_filters=256, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_conv3b = batch_norm(Conv2DLayer(l_conv3a, num_filters=256, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_conv3c = batch_norm(Conv2DLayer(l_conv3b, num_filters=256, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_pool3 = MaxPool2DLayer(l_conv3c, pool_size=2) # feature maps 16x16

    l_conv4a = batch_norm(Conv2DLayer(l_pool3, num_filters=512, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_conv4b = batch_norm(Conv2DLayer(l_conv4a, num_filters=512, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_conv4c = batch_norm(Conv2DLayer(l_conv4b, num_filters=512, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_pool4 = MaxPool2DLayer(l_conv4c, pool_size=2) # feature maps 8x8

    l_conv5a = batch_norm(Conv2DLayer(l_pool4, num_filters=512, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_conv5b = batch_norm(Conv2DLayer(l_conv5a, num_filters=512, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_conv5c = batch_norm(Conv2DLayer(l_conv5b, num_filters=512, filter_size=3, pad=1, W=he_norm, nonlinearity=rectify))
    l_pool5 = MaxPool2DLayer(l_conv5c, pool_size=2) # feature maps 4x4
    l_conv5_dropout = DropoutLayer(l_pool5, p=0.5)

    l_hidden1 = batch_norm(DenseLayer(l_conv5_dropout, num_units=1024, W=he_norm, nonlinearity=rectify))
    l_dropout1 = DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = batch_norm(DenseLayer(l_dropout1, num_units=1024, W=he_norm, nonlinearity=rectify))
    l_dropout2 = DropoutLayer(l_hidden2, p=0.5)

    l_out = DenseLayer(l_dropout2, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return l_out

# ========================================================================================================================

def vgg16(input_var=None):
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    l_conv1a = batch_norm(Conv2DLayer(l_in, num_filters=32, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_conv1b = batch_norm(Conv2DLayer(l_conv1a, num_filters=32, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=(3,2), stride=2) # feature maps 64x64

    l_conv2a = batch_norm(Conv2DLayer(l_pool1, num_filters=128, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_conv2b = batch_norm(Conv2DLayer(l_conv2a, num_filters=128, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=(3,2), stride=2) # feature maps 32x32

    l_conv3a = batch_norm(Conv2DLayer(l_pool2, num_filters=128, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_conv3b = batch_norm(Conv2DLayer(l_conv3a, num_filters=128, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_conv3c = batch_norm(Conv2DLayer(l_conv3b, num_filters=128, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_pool3 = MaxPool2DLayer(l_conv3c, pool_size=(3,2), stride=2) # feature maps 16x16

    l_conv4a = batch_norm(Conv2DLayer(l_pool3, num_filters=256, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_conv4b = batch_norm(Conv2DLayer(l_conv4a, num_filters=256, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_conv4c = batch_norm(Conv2DLayer(l_conv4b, num_filters=256, filter_size=3, pad=1, W=he_norm, nonlinearity=very_leaky_rectify))
    l_pool4 = MaxPool2DLayer(l_conv4c, pool_size=(3,2), stride=2) # feature maps 8x8

    l_hidden1 = batch_norm(DenseLayer(l_pool4, num_units=1024, W=he_norm, nonlinearity=very_leaky_rectify))
    l_dropout1 = DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = batch_norm(DenseLayer(l_dropout1, num_units=1024, W=he_norm, nonlinearity=very_leaky_rectify))
    l_dropout2 = DropoutLayer(l_hidden2, p=0.5)

    l_out = DenseLayer(l_dropout2, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return l_out

# ========================================================================================================================

def vgg16_fc7(input_var=None):
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    l_conv1a = batch_norm(Conv2DLayer(l_in, num_filters=64, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv1b = batch_norm(Conv2DLayer(l_conv1a, num_filters=64, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=2) # feature maps 32x32
    l_conv1_dropout = DropoutLayer(l_pool1, p=0.25)

    l_conv2a = batch_norm(Conv2DLayer(l_conv1_dropout, num_filters=128, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv2b = batch_norm(Conv2DLayer(l_conv2a, num_filters=128, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=2) # feature maps 16x16
    l_conv2_dropout = DropoutLayer(l_pool2, p=0.35)

    l_conv3a = batch_norm(Conv2DLayer(l_conv2_dropout, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv3b = batch_norm(Conv2DLayer(l_conv3a, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv3c = batch_norm(Conv2DLayer(l_conv3b, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool3 = MaxPool2DLayer(l_conv3c, pool_size=2) # feature maps 8x8
    l_conv3_dropout = DropoutLayer(l_pool3, p=0.45)

    l_conv4a = batch_norm(Conv2DLayer(l_conv3_dropout, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv4b = batch_norm(Conv2DLayer(l_conv4a, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv4c = batch_norm(Conv2DLayer(l_conv4b, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool4 = MaxPool2DLayer(l_conv4c, pool_size=2) # feature maps 4x4
    l_conv4_dropout = DropoutLayer(l_pool4, p=0.5)

    l_conv5a = batch_norm(Conv2DLayer(l_conv4_dropout, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv5b = batch_norm(Conv2DLayer(l_conv5a, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv5c = batch_norm(Conv2DLayer(l_conv5b, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool5 = MaxPool2DLayer(l_conv5c, pool_size=2) # feature maps 2x2
    l_conv5_dropout = DropoutLayer(l_pool5, p=0.5)

    l_hidden1 = batch_norm(DenseLayer(l_conv5_dropout, num_units=1024, W=he_norm, nonlinearity=rectify))
    l_dropout1 = DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = batch_norm(DenseLayer(l_dropout1, num_units=1024, W=he_norm, nonlinearity=rectify))
    l_dropout2 = DropoutLayer(l_hidden2, p=0.5)

    l_out = DenseLayer(l_dropout2, num_units=10, W=xavier_norm, nonlinearity=softmax)

    return l_out, l_hidden2

# ========================================================================================================================

def ResNet_Orig(input_var=None, n=9):
    '''
    Stolen from from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning ;-)
    '''
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=True):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(Conv2DDNNLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))

        stack_2 = batch_norm(Conv2DDNNLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu')))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(Conv2DDNNLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # first layer, output is 64 x 32 x 32
    l = batch_norm(Conv2DDNNLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))

    # first stack of residual blocks, output is 32 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 64 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 128 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
            l, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network

# ========================================================================================================================

def ResNet_Orig_ELU(input_var=None, n=5):
    '''
    Stolen from from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning ;-)
    '''
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=elu, pad='same', W=lasagne.init.HeNormal(gain='relu'))

        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu')))

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([stack_2, projection])
        else:
            block = ElemwiseSumLayer([stack_2, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # first layer, output is 64 x 32 x 32
    l = ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=elu, pad='same', W=lasagne.init.HeNormal(gain='relu'))

    # first stack of residual blocks, output is 32 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 64 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 128 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
            l, num_units=10,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

    return network


# ========================================================================================================================

def ResNet_FullPre(input_var=None, n=5):
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    Formula to figure out depth: 8n+2
    '''
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=True, first=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        # contains the last weight portion, step 6
        conv_2 = ConvLayer(conv_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])

            # identity shortcut, as option A in paper
            #identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
            #padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
            #block = ElemwiseSumLayer([conv_2, padding])
        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # first layer, output is 16 x 64 x 64
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(5,5), stride=(1,1), nonlinearity=rectify, pad='same', W=he_norm))
    l = MaxPool2DLayer(l, pool_size=2)

    # first stack of residual blocks, output is 16 x 64 x 64
    l = residual_block(l, first=True)
    for _ in range(1,n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 32 x 32
    l = residual_block(l, increase_dim=True)
    for _ in range(1,(n+2)):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,(n+2)):
        l = residual_block(l)

    # third stack of residual blocks, output is 128 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # FC should be alternative to avg_pool
    #l_hidden1 = batch_norm(DenseLayer(avg_pool, num_units=1024, W=he_norm, nonlinearity=rectify))
    #l_hidden2 = batch_norm(DenseLayer(l_hidden1, num_units=1024, W=he_norm, nonlinearity=rectify))

    # dropout
    #dropout = DropoutLayer(avg_pool, p=0.25)

    # fully connected layer
    network = DenseLayer(avg_pool, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return network

# ========================================================================================================================

def ResNet_FullPre_Wide(input_var=None, n=5, k=2):
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    Formula to figure out depth: 8n+2
    '''
    n_filters = {0:16, 1:16*k, 2:32*k, 3:64*k, 4:128*k}

    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=True, first=False, filters=16):
        if increase_dim:
            first_stride = (2,2)
            #out_num_filters = input_num_filters
        else:
            first_stride = (1,1)
            #out_num_filters = input_num_filters

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        dropout = DropoutLayer(conv_1, p=0.35)

        # contains the last weight portion, step 6
        conv_2 = ConvLayer(dropout, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])

        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])

        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # first layer, output is 16 x 64 x 64
    l = batch_norm(ConvLayer(l_in, num_filters=n_filters[0], filter_size=(7,7), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))
    l = MaxPool2DLayer(l, pool_size=(3,3), stride=2, pad=0)

    # first stack of residual blocks, output is 32 x 64 x 64
    l = residual_block(l, first=True, filters=n_filters[1])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[1])

    # second stack of residual blocks, output is 64 x 32 x 32
    l = residual_block(l, increase_dim=True, filters=n_filters[2])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[2])

    # third stack of residual blocks, output is 128 x 16 x 16
    l = residual_block(l, increase_dim=True, filters=n_filters[3])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[3])

    # third stack of residual blocks, output is 256 x 8 x 8
    l = residual_block(l, increase_dim=True, filters=n_filters[4])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[4])

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # dropout
    dropout_last = DropoutLayer(avg_pool, p=0.25)

    # fully connected layer
    network = DenseLayer(dropout_last, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return network

# ========================================================================================================================

def ST_ResNet_FullPre(input_var=None, n=5, k=2):
    '''
    Spatial Transformer ResNet
    'Spatial Transformer Networks', Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu (https://arxiv.org/pdf/1506.02025v3.pdf)
    Adapted from https://github.com/skaae/transformer_network

    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    Formula to figure out depth: 8n+2
    '''
    st_filters = {0:16, 1:32, 2:64, 3:128}
    n_filters = {0:16, 1:16*k, 2:32*k, 3:64*k, 4:128*k}

    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=True, first=False, filters=16):
        if increase_dim:
            first_stride = (2,2)
            #out_num_filters = input_num_filters
        else:
            first_stride = (1,1)
            #out_num_filters = input_num_filters

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

        dropout = DropoutLayer(conv_1, p=0.35)

        # contains the last weight portion, step 6
        conv_2 = ConvLayer(dropout, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=he_norm)

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])

        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])

        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # Localization network
    # same architecture as svhn localization network from paper
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()

    loc_conv1 = batch_norm(ConvLayer(l_in, num_filters=st_filters[0], filter_size=(7,7), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))
    loc_pool = MaxPool2DLayer(loc_conv1, pool_size=(3,3))

    loc_conv2 = residual_block(loc_pool, first=True, filters=st_filters[0])
    loc_conv3 = residual_block(loc_conv2, filters=st_filters[0])
    loc_conv4 = residual_block(loc_conv3, filters=st_filters[0])

    loc_conv5 = residual_block(loc_conv4, increase_dim=True, filters=st_filters[1])
    loc_conv6 = residual_block(loc_conv5, filters=st_filters[1])
    loc_conv7 = residual_block(loc_conv6, filters=st_filters[1])

    loc_conv8 = residual_block(loc_conv7, increase_dim=True, filters=st_filters[2])
    loc_conv9 = residual_block(loc_conv8, filters=st_filters[2])
    loc_conv10 = residual_block(loc_conv9, filters=st_filters[2])

    loc_bn_post_conv = BatchNormLayer(loc_conv10)
    loc_bn_post_relu = NonlinearityLayer(loc_bn_post_conv, rectify)

    # average pooling
    loc_avg_pool = GlobalPoolLayer(loc_bn_post_relu)

    loc_out = DenseLayer(loc_avg_pool, num_units=6, b=b, W=lasagne.init.Constant(0.0), nonlinearity=None)

    # Transformer network
    l_trans1 = TransformerLayer(l_in, loc_out, downsample_factor=1.0)

    # Classifier Network
    # first layer, output is 16 x 64 x 64
    l = batch_norm(ConvLayer(l_trans1, num_filters=n_filters[0], filter_size=(7,7), stride=(2,2), nonlinearity=rectify, pad='same', W=he_norm))
    l = MaxPool2DLayer(l, pool_size=(3,3), stride=2, pad=2)

    # first stack of residual blocks, output is 32 x 64 x 64
    l = residual_block(l, first=True, filters=n_filters[1])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[1])

    # second stack of residual blocks, output is 64 x 32 x 32
    l = residual_block(l, increase_dim=True, filters=n_filters[2])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[2])

    # third stack of residual blocks, output is 128 x 16 x 16
    l = residual_block(l, increase_dim=True, filters=n_filters[3])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[3])

    # third stack of residual blocks, output is 256 x 8 x 8
    l = residual_block(l, increase_dim=True, filters=n_filters[4])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[4])

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # dropout
    dropout_last = DropoutLayer(avg_pool, p=0.25)

    # fully connected layer
    network = DenseLayer(dropout_last, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return network


# ========================================================================================================================

def ResNet_FullPre_ELU(input_var=None, n=5):
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    Forumala to figure out depth: 8n+2
    '''
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=True, first=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            #bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(l, elu)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = ConvLayer(bn_pre_relu, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=elu, pad='same', W=ortho)

        # contains the last weight portion, step 6
        conv_2 = ConvLayer(conv_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=ortho)

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])
        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # first layer, output is 16 x 128 x 128
    l = ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=elu, pad='same', W=ortho)

    # first stack of residual blocks, output is 16 x 128 x 128
    l = residual_block(l, first=True)
    for _ in range(1,n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 64 x 64
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 32 x 32
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 128 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    #bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(l, elu)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # fully connected layer
    network = DenseLayer(avg_pool, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return network


# ========================================================================================================================

def ResNet_BttlNck_FullPre(input_var=None, n=18):
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    Formula to figure out depth: 9n + 2
    '''
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_bottleneck_block(l, increase_dim=False, first=False):
        input_num_filters = l.output_shape[1]

        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        bottleneck_filters = out_num_filters / 4

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=bottleneck_filters, filter_size=(1,1), stride=(1,1), nonlinearity=rectify, pad=0, W=he_norm))

        conv_2 = batch_norm(ConvLayer(conv_1, num_filters=bottleneck_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad=1, W=he_norm))

        # contains the last weight portion, step 6
        conv_3 = ConvLayer(conv_2, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad=0, W=he_norm)

        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad=0, b=None)
            block = ElemwiseSumLayer([conv_3, projection])
        else:
            block = ElemwiseSumLayer([conv_3, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    # first layer, output is 64 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad=1, W=he_norm))

    # first stack of residual blocks, output is 128 x 32 x 32
    l = residual_bottleneck_block(l, first=True)
    for _ in range(1,n):
        l = residual_bottleneck_block(l)

    # second stack of residual blocks, output is 256 x 16 x 16
    l = residual_bottleneck_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_bottleneck_block(l)

    # third stack of residual blocks, output is 512 x 8 x 8
    l = residual_bottleneck_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_bottleneck_block(l)

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # fully connected layer
    network = DenseLayer(avg_pool, num_units=10, W=HeNormal(), nonlinearity=softmax)

    return network

# ========================================================================================================================
#                                                   Pre-trained Models Below
# ========================================================================================================================
# BLVC Googlenet, model from the paper:
# "Going Deeper with Convolutions"
# Original source:
# https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
# License: unrestricted use

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl


def build_inception_module(name, input_layer, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(
        net['pool'], nfilters[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

    net['3x3_reduce'] = ConvLayer(
        input_layer, nfilters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(
        net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

    net['5x5_reduce'] = ConvLayer(
        input_layer, nfilters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(
        net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
        ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def bvlc_googlenet(input_var=None):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1/7x7_s2'] = ConvLayer(
        net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = MaxPool2DLayer(
        net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(
        net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(
        net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = MaxPool2DLayer(
      net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = MaxPool2DLayer(
      net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = MaxPool2DLayer(
      net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                         num_units=1000,
                                         nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                    nonlinearity=softmax)
    return net

def bvlc_googlenet_submission(input_var=None):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1/7x7_s2'] = ConvLayer(
        net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = MaxPool2DLayer(
        net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(
        net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(
        net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = MaxPool2DLayer(
      net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = MaxPool2DLayer(
      net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = MaxPool2DLayer(
      net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])

    output_layer = DenseLayer(net['pool5/7x7_s1'], num_units=10, W=lasagne.init.HeNormal(), nonlinearity=softmax)

    return output_layer

# ========================================================================================================================

# Inception-v3, model from the paper:
# "Rethinking the Inception Architecture for Computer Vision"
# http://arxiv.org/abs/1512.00567
# Original source:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
# License: http://www.apache.org/licenses/LICENSE-2.0

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/inception_v3.pkl


def preprocess(im):
    # Expected input: RGB uint8 image
    # Input to network should be bc01, 299x299 pixels, scaled to [-1, 1].
    import skimage.transform
    import numpy as np

    im = skimage.transform.resize(im, (299, 299), preserve_range=True)
    im = (im - 128) / 128.
    im = np.rollaxis(im, 2)[np.newaxis].astype('float32')

    return im


def bn_conv(input_layer, **kwargs):
    l = Conv2DLayer(input_layer, **kwargs)
    l = batch_norm(l, epsilon=0.001)
    return l


def inceptionA(input_layer, nfilt):
    # Corresponds to a modified version of figure 5 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionB(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionC(input_layer, nfilt):
    # Corresponds to figure 6 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
    l3 = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionD(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    l1 = bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    l2 = bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionE(input_layer, nfilt, pool_mode):
    # Corresponds to figure 7 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2a = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
    l2b = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3a = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
    l3b = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])


def inception_v3(input_var=None):
    net = {}

    net['input'] = InputLayer((None, 3, None, None), input_var=input_var)
    net['conv'] = bn_conv(net['input'],
                          num_filters=32, filter_size=3, stride=2)
    net['conv_1'] = bn_conv(net['conv'], num_filters=32, filter_size=3)
    net['conv_2'] = bn_conv(net['conv_1'],
                            num_filters=64, filter_size=3, pad=1)
    net['pool'] = Pool2DLayer(net['conv_2'], pool_size=3, stride=2, mode='max')

    net['conv_3'] = bn_conv(net['pool'], num_filters=80, filter_size=1)

    net['conv_4'] = bn_conv(net['conv_3'], num_filters=192, filter_size=3)

    net['pool_1'] = Pool2DLayer(net['conv_4'],
                                pool_size=3, stride=2, mode='max')
    net['mixed/join'] = inceptionA(
        net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
    net['mixed_1/join'] = inceptionA(
        net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_2/join'] = inceptionA(
        net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_3/join'] = inceptionB(
        net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

    net['mixed_4/join'] = inceptionC(
        net['mixed_3/join'],
        nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

    net['mixed_5/join'] = inceptionC(
        net['mixed_4/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_6/join'] = inceptionC(
        net['mixed_5/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_7/join'] = inceptionC(
        net['mixed_6/join'],
        nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

    net['mixed_8/join'] = inceptionD(
        net['mixed_7/join'],
        nfilt=((192, 320), (192, 192, 192, 192)))

    net['mixed_9/join'] = inceptionE(
        net['mixed_8/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='average_exc_pad')

    net['mixed_10/join'] = inceptionE(
        net['mixed_9/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='max')

    net['pool3'] = GlobalPoolLayer(net['mixed_10/join'])

    net['softmax'] = DenseLayer(
        net['pool3'], num_units=1008, nonlinearity=softmax)

    return net
