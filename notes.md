# Distracted Driver State Farm Kaggle Challenge nonlinearities

Given a dataset of 2D dashboard camera images, State Farm is challenging Kagglers to classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the backseat?

## Current solution

### Ensemble of 15 ResNet34
* Initial filter num - 16
* No L2 regularization
* Option A (Identity shortcuts)
* trained with ADAM for 60 epoch- lr_schedule = {0:0.003, 15:0.0003, 35:0.0003, 50:0.0001}
* Individual models get 99.5-99.9%
* Submission score - 1.03804

## Things currently trying

### ResNet56
* Initial filter num - 32
* L2 regularization - 0.0001 (same as paper)
* ADAM for 60 epoch - lr_schedule = {0:0.001, 15:0.0001, 30:0.0001, 50:0.00001}
* Individual accuracy - 99.7%
* Submission score - TBD (do this today)

## To try later

* SGD instead of ADAM, SGD should reach better accuracy but slower training.
* Random forest output split, like in class project
