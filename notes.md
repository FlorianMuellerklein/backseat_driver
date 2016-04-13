# Distracted Driver State Farm Kaggle Challenge

Given a dataset of 2D dashboard camera images, State Farm is challenging Kagglers to classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the backseat?

## Current solution

### Ensemble of 15 ResNet34
* Initial filter num - 16
* No L2 regularization
* Option A (Identity shortcuts)
* trained with ADAM for 60 epoch- lr_schedule = {0:0.003, 15:0.0003, 35:0.00003, 50:0.00001}
* Individual models get 99.5-99.9%
* Submission score - 1.03804

### Current Data augmentation
* rotation - (-15,15)
* translation - (-6,6)
* zoom - (0.8,1.2)
* flip_lr - True (50%)
* flip_ud - False
* RGB intensity - (-25,25)

## Things that have worked well

### ResNet56
* Initial filter num - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 60 epoch - lr_schedule = {0:0.001, 15:0.0001, 30:0.00001, 50:0.000001}
* Projection option
* With flip_ud augmentation
* Individual accuracy - 99.7%
* Submission score - 0.63255

### ResNet56
* Initial filter num - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 60 epoch - lr_schedule = {0:0.001, 15:0.0001, 30:0.00001, 50:0.000001}
* Projection option
* Individual accuracy - 99.7%
* Submission score - 0.71352

### ResNet34
* Initial filter num - 16
* ADAM for 60 epoch - lr_schedule = {0:0.001, 15:0.0001, 30:0.00001, 50:0.000001}
* Identity option
* Individual models get 99.5-99.9%
* Submission score - 1.03804


### vgg16
* Initial filter num -64
* ADAM for 60 epoch - lr_schedule = {0:0.003, 15:0.0003, 30:0.00003, 50:0.00001}
* Increasing dropout per pool & hidden - (0.25,0.35,0.45,0.5,0.5,0.5,0.5)
* CNN filters init - Orthogonal
* hidden init - HeNormal
Submission score - 1.05536

## Things currently trying

* add flipud to current best ResNet try

## To try later

* SGD instead of ADAM, SGD should reach better accuracy but slower training.
* Random forest output split, like in class project
* Test whether LR flips hurt scores
* Start testing single models instead of ensembles, then ensemble best singles for final submission
