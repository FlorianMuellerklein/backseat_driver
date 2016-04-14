import sys
sys.setrecursionlimit(10000)
import lasagne
from lasagne.nonlinearities import rectify, softmax, sigmoid
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper, batch_norm
# for ResNet
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer, ElemwiseSumLayer, NonlinearityLayer, PadLayer, GlobalPoolLayer, ExpressionLayer
from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.init import Orthogonal, HeNormal, GlorotNormal

PIXELS = 64
imageSize = PIXELS * PIXELS
num_features = imageSize * 3

ortho = Orthogonal(gain='relu')
he_norm = HeNormal(gain='relu')
xavier_norm = GlorotNormal(gain=1.0)

def vgg16(input_var=None):
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    l_conv1a = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv1b = batch_norm(ConvLayer(l_conv1a, num_filters=64, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=2) # feature maps 32x32
    l_conv1_dropout = DropoutLayer(l_pool1, p=0.25)

    l_conv2a = batch_norm(ConvLayer(l_conv1_dropout, num_filters=128, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv2b = batch_norm(ConvLayer(l_conv2a, num_filters=128, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=2) # feature maps 16x16
    l_conv2_dropout = DropoutLayer(l_pool2, p=0.35)

    l_conv3a = batch_norm(ConvLayer(l_conv2_dropout, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv3b = batch_norm(ConvLayer(l_conv3a, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv3c = batch_norm(ConvLayer(l_conv3b, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool3 = MaxPool2DLayer(l_conv3c, pool_size=2) # feature maps 8x8
    l_conv3_dropout = DropoutLayer(l_pool3, p=0.45)

    l_conv4a = batch_norm(ConvLayer(l_conv3_dropout, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv4b = batch_norm(ConvLayer(l_conv4a, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv4c = batch_norm(ConvLayer(l_conv4b, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool4 = MaxPool2DLayer(l_conv4c, pool_size=2) # feature maps 4x4
    l_conv4_dropout = DropoutLayer(l_pool4, p=0.5)

    l_conv5a = batch_norm(ConvLayer(l_conv4_dropout, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv5b = batch_norm(ConvLayer(l_conv5a, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv5c = batch_norm(ConvLayer(l_conv5b, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool5 = MaxPool2DLayer(l_conv5c, pool_size=2) # feature maps 2x2
    l_conv5_dropout = DropoutLayer(l_pool5, p=0.5)

    l_hidden1 = batch_norm(DenseLayer(l_conv5_dropout, num_units=1024, W=he_norm, nonlinearity=rectify))
    l_dropout1 = DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = batch_norm(DenseLayer(l_dropout1, num_units=1024, W=he_norm, nonlinearity=rectify))
    l_dropout2 = DropoutLayer(l_hidden2, p=0.5)

    l_out = DenseLayer(l_dropout2, num_units=10, W=xavier_norm, nonlinearity=softmax)

    return l_out

def vgg16_fc7(input_var=None):
    l_in = InputLayer(shape=(None, 3, PIXELS, PIXELS), input_var=input_var)

    l_conv1a = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv1b = batch_norm(ConvLayer(l_conv1a, num_filters=64, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool1 = MaxPool2DLayer(l_conv1b, pool_size=2) # feature maps 32x32
    l_conv1_dropout = DropoutLayer(l_pool1, p=0.25)

    l_conv2a = batch_norm(ConvLayer(l_conv1_dropout, num_filters=128, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv2b = batch_norm(ConvLayer(l_conv2a, num_filters=128, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool2 = MaxPool2DLayer(l_conv2b, pool_size=2) # feature maps 16x16
    l_conv2_dropout = DropoutLayer(l_pool2, p=0.35)

    l_conv3a = batch_norm(ConvLayer(l_conv2_dropout, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv3b = batch_norm(ConvLayer(l_conv3a, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv3c = batch_norm(ConvLayer(l_conv3b, num_filters=256, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool3 = MaxPool2DLayer(l_conv3c, pool_size=2) # feature maps 8x8
    l_conv3_dropout = DropoutLayer(l_pool3, p=0.45)

    l_conv4a = batch_norm(ConvLayer(l_conv3_dropout, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv4b = batch_norm(ConvLayer(l_conv4a, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv4c = batch_norm(ConvLayer(l_conv4b, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool4 = MaxPool2DLayer(l_conv4c, pool_size=2) # feature maps 4x4
    l_conv4_dropout = DropoutLayer(l_pool4, p=0.5)

    l_conv5a = batch_norm(ConvLayer(l_conv4_dropout, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv5b = batch_norm(ConvLayer(l_conv5a, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_conv5c = batch_norm(ConvLayer(l_conv5b, num_filters=512, filter_size=3, pad=1, W=ortho, nonlinearity=rectify))
    l_pool5 = MaxPool2DLayer(l_conv5c, pool_size=2) # feature maps 2x2
    l_conv5_dropout = DropoutLayer(l_pool5, p=0.5)

    l_hidden1 = batch_norm(DenseLayer(l_conv5_dropout, num_units=1024, W=he_norm, nonlinearity=rectify))
    l_dropout1 = DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = batch_norm(DenseLayer(l_dropout1, num_units=1024, W=he_norm, nonlinearity=rectify))
    l_dropout2 = DropoutLayer(l_hidden2, p=0.5)

    l_out = DenseLayer(l_dropout2, num_units=10, W=xavier_norm, nonlinearity=softmax)

    return l_out, l_hidden2

def ResNet56(input_var=None, n=9):
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
    l = batch_norm(Conv2DDNNLayer(l_in, num_filters=32, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))

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
