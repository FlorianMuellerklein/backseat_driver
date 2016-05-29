#!/bin/bash

for i in {0..9}
do
    python submission_single.py --label ResNet82_5x5_BN_rgb_pseudo --pixels 128 --fold $i
done
