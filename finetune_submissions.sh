#!/bin/bash

for i in {0..9}
do
    echo $i
    python submission_finetune.py --label GoogLeNet --pixels 224 --fold $i
done
