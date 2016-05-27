#!/bin/bash

for i in {0..9}
do
    python train_nn_pseudo.py --label ResNet82_5x5_BN_rgb_pseudo --pixels 128 --fold $i --batchsize 64
done
