#!/bin/bash

for i in {0..9}
do
    python finetune_bvlc_googlenet.py --label GoogLeNet --pixels 224 --fold $i --batchsize 32
done
