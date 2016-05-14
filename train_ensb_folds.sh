#!/bin/bash

for i in {0..10}
do
    python train_nn.py --label ResNet42_5x5 --pixels 128 --fold $i --batchsize 32
done
