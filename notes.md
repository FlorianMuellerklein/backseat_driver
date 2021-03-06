# Distracted Driver State Farm Kaggle Challenge

Given a dataset of 2D dashboard camera images, State Farm is challenging Kagglers to classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the backseat?

## Current solution

### ResNet-82 with Pseudo Label (Soft Targets)
* 128x128 Image size
* Initial filter num - 16
* Initial filter size - 5x5, stride 1
* Maxpool after first filter
* Soft targets generated from ensemble 10 finetuned (ImageNet) VGG with data aug
* FullPreActivation
* Batch size - 64 (44 train / 20 test)
* L2 regularization - 0.0001
* ADAM for 100 epoch - lr_schedule = {0:0.001, 60:0.0001, 80:0.00001}
* Heavy Train Augmentations
* Trans TTA
* Mean-pixel centering
* Projection option
* Individual (naive cv) accuracy - 99.7%
* Local CV (naive) loss - 0.006 to 0.013764
* Submission single model LB score - 0.19
* Submission ensemble of 10 - 0.18454
* Time per epoch - 131 seconds

### Current Train Augmentation
* Pad images with 16 pixels
* Random crops of original image size
* Random color intensity for each channel -20,20 (uniform)
* Random rotations between -15,15 degrees (uniform)
* Random shear between -5,5 degrees (uniform)
* Random image brightness aug 90-110% (uniform)

### Current Test Augmentation x15
* Pad images with 8 pixels
* Random crops of original image size

## Past solution

### Ensemble of 15 ResNet34
* Initial filter num - 16
* No L2 regularization
* Option A (Identity shortcuts)
* Trained with ADAM for 60 epoch- lr_schedule = {0:0.003, 15:0.0003, 35:0.00003, 50:0.00001}
* Individual models get 99.5-99.9%
* Single model submission score - 1.03804
* Ensemble submission score - 0.38794

### Old augmentation
* rotation - (-15,15)
* translation - (-6,6)
* zoom - (0.8,1.2)
* flip_lr - False (50%)
* flip_ud - False
* RGB intensity - (-25,25)

## Things that have worked well

### ResNet-82
* 128x128 Image size
* Initial filter num - 16
* Initial filter size - 5x5, stride 1
* No maxpool after first filter
* FullPreActivation
* Batch size - 64
* L2 regularization - 0.00001 (less than paper)
* Dropout after GlobalPoolLayer p=0.25
* ADAM for 100 epoch - lr_schedule = {0:0.001, 60:0.0001, 80:0.00001}
* Heavy Train Augmentations
* Trans TTA
* Mean-pixel centering
* Projection option
* Individual (old local cv) accuracy - 99.9%
* Local CV loss - 0.003972 -> 0.023
* Submission score - 0.41797 -> 0.49150
* Time per epoch - 174.7 seconds

### ST-ResNet-82
* 128x128 Image size
* Initial filter num - 16
* Initial filter size - 5x5, stride 1
* No maxpool after first filter
* FullPreActivation
* Batch size - 64
* L2 regularization - 0.00001 (less than paper)
* Dropout after GlobalPoolLayer p=0.25
* ADAM for 100 epoch - lr_schedule = {0:0.001, 60:0.0001, 80:0.00001}
* Heavy Train Augmentations
* Trans TTA
* Mean-pixel centering
* Projection option
* ST layer is PreResNet with initial filter num 16, 7x7 stride 2, followed by maxpool 3x3
* ST Net has four residual blocks
* Individual LB score with pseudo labels is 0.187
* Time per epoch - 174.7 seconds

### ResNet-42 More train and test aug (best single model so far)
* 128x128 Image size
* Initial filter num - 16
* Initial filter size - 3x3, stride 1
* No maxpool after first filter
* FullPreActivation
* Batch size - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 100 epoch - lr_schedule = {0:0.001, 60:0.0001}
* Heavy Train Augmentations
* Trans, color, brightness TTA
* Mean-pixel centering
* Projection option
* Individual (new local cv) accuracy - 89.9%
* Local CV loss - 0.35086
* Submission score - 0.43872
* Time per epoch - 230 seconds

### ResNet-42
* 128x128 Image size
* Initial filter num - 16
* Initial filter size - 5x5, stride 1
* Maxpool after initial filter - 2x2
* FullPreActivation
* Batch size - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 100 epoch - lr_schedule = {0:0.001, 60:0.0001, 85:0.00001}
* Heavy Train Augmentations
* Trans, color, brightness TTA
* Mean-pixel centering
* Projection option
* Individual (old cv) accuracy - 99.8%
* Local old CV loss - 0.00548 to 0.01809
* Submission score - 0.46059 to 0.54379
* Time per epoch - 82 seconds!

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

### ResNet-42 with FC (Bug Free)
* Initial filter num - 16
* FullPreActivation
* Batch size - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 100 epoch - lr_schedule = {0:0.001, 60:0.0001}
* Pad Crop Color Augmentations
* Mean centered image scaling
* Projection option
* Individual accuracy - 99.8%
* Submission score - 0.88490


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
