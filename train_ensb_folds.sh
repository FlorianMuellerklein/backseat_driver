#!/bin/bash

for i in {0..15}
do
    python train_nn.py --label ResNet82_5x5_BN_rgb_clean --pixels 128 --fold $i --batchsize 32
done
