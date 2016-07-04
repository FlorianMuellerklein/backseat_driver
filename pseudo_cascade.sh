#!/bin/bash

for i in {0..9}
do
    python train_nn_pseudo.py --label wide_resnet_n5_k4_7x7_pseudo --pixels 128 --fold $i --batchsize 32 --epochs 200
    python gen_pseudo_cascade.py --label wide_resnet_n5_k4_7x7_pseudo --pixels 128 --fold $i
done
