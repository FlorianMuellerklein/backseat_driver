# Distracted Driver State Farm Kaggle Challenge

Given a dataset of 2D dashboard camera images, State Farm is challenging Kagglers to classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the backseat?

## Current solution

### Ensemble of 15 ResNet56
* Initial filter num - 16
* L2 regularization - 0.0001 (same as paper)
* Option B (Projection shortcuts)
* Batch size - 32
* Trained with ADAM for 75 epoch- lr_schedule = {0:0.001, 30:0.0001, 65:0.00001}
* Individual models get 99.5-99.9% (random 10% valid on each training net)
* Single model submission score - 0.71352
* Ensemble submission score - 0.30114
* Including batch iterator bug

## Past solution

### Ensemble of 15 ResNet34
* Initial filter num - 16
* No L2 regularization
* Option A (Identity shortcuts)
* Trained with ADAM for 60 epoch- lr_schedule = {0:0.003, 15:0.0003, 35:0.00003, 50:0.00001}
* Individual models get 99.5-99.9%
* Single model submission score - 1.03804
* Ensemble submission score - 0.38794

### Crop Pad Color augmentation
* Pad images with 16 PIXELS
* Random crops of original image size
* random color intensity for each channel

### Old augmentation
* rotation - (-15,15)
* translation - (-6,6)
* zoom - (0.8,1.2)
* flip_lr - False (50%)
* flip_ud - False
* RGB intensity - (-25,25)

## Things that have worked well

### ResNet-42 (Bug Free)
* Initial filter num - 16
* FullPreActivation
* Batch size - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 100 epoch - lr_schedule = {0:0.001, 60:0.0001}
* Pad Crop Color Augmentations
* Mean centered image scaling
* Projection option
* Individual accuracy - 99.8%
* Submission score - 0.46547

### ResNet-42 very LeakyReLU(Bug Free)
* Initial filter num - 16
* FullPreActivation
* Batch size - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 100 epoch - lr_schedule = {0:0.001, 60:0.0001}
* Pad Crop Color Augmentations
* Mean centered image scaling
* Projection option
* Individual accuracy - 99.8%
* Submission score - 0.60455

## Everything below this had bugs in the batch iterator and ResNet Architecture :-/

### ResNet-110
* Initial filter num - 16
* FullPreActivation
* Batch size - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 75 epoch - lr_schedule = {0:0.001, 30:0.0001, 65:0.00001}
* Projection option
* Individual accuracy - 99.7%
* Submission score - 0.60583

### ResNet-110 - Relabeled Training
* Initial filter num - 16
* FullPreActivation
* Batch size - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 75 epoch - lr_schedule = {0:0.001, 30:0.0001, 65:0.00001}
* Projection option
* Individual accuracy - 99.7%
* Submission score - 0.63784

### ResNet-56
* Initial filter num - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 60 epoch - lr_schedule = {0:0.001, 15:0.0001, 30:0.00001, 50:0.000001}
* Projection option
* With flip_ud augmentation
* Individual accuracy - 99.5%
* Submission score - 0.63255
* Batch size - 32

### ResNet56
* Initial filter num - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 60 epoch - lr_schedule = {0:0.001, 15:0.0001, 30:0.00001, 50:0.000001}
* Projection option
* Individual accuracy - 99.7%
* Submission score - 0.71352
* Batch size - 32

### ResNet56
* Initial filter num - 64
* L2 regularization - 0.0001 (same as paper)
* ADAM for 80 epoch - lr_schedule = {0: 0.001, 25: 0.0001, 50: 0.00001, 65: 0.000001}
* Projection option
* With flip_ud augmentation
* Individual accuracy - 99.6%
* Submission score - 0.73603
* Batch size - 32

### ResNet56
* Initial filter num - 64
* L2 regularization - 0.0001 (same as paper)
* ADAM for 60 epoch - lr_schedule = {0:0.001, 15:0.0001, 30:0.00001, 50:0.000001}
* Projection option
* With flip_ud augmentation
* Individual accuracy - 99.1%
* Submission score - 0.82991
* Batch size - 16

### ResNet-34
* Initial filter num - 16
* ADAM for 60 epoch - lr_schedule = {0:0.001, 15:0.0001, 30:0.00001, 50:0.000001}
* Identity option
* Individual models get 99.5-99.9%
* Submission score - 1.03804
* Batch size - 32

### VGG16
* Initial filter num - 64
* ADAM for 60 epoch - lr_schedule = {0:0.003, 15:0.0003, 30:0.00003, 50:0.00001}
* Increasing dropout per pool & hidden - (0.25,0.35,0.45,0.5,0.5,0.5,0.5)
* CNN filters init - Orthogonal
* Hidden init - HeNormal
* Submission score - 1.05536
* Batch size - 32

## Things currently trying

* Ensemble of 10 ResNet32 with pad, crop, color augmentation (no flip_lr), test-time-augmentations.

## To try later

* Extract features from scratch and Pretrained models, concat features then train MLP on those.
* MLP at the end of ResNet
* Whatever my best ensemble is, add test time augmentations (without flipping)
* SGD instead of ADAM, SGD should reach better accuracy but slower training.
* Test whether LR flips hurt scores
* Start testing single models instead of ensembles, then ensemble best singles for final submission
* Try increased number of filters for ResNets, 16 might be low
* Train any models for longer than 60 to see if the scores become more stable.
* Train models without lr_decay and look at plots to get an idea about the best decay points. (ResNets use decay points from VGG training, might not be optimal)
* VGG with L2 regularization

## Things that didn't really work out

* Ensemble of 19 ResNet-110 with lr and old augmentations.
* Random forest on fc7 features. The thought was that it might draw very different decision boundaries through the learned ConvNet feature representations than the softmax. So it might be useful to blend with the softmax predictions.
