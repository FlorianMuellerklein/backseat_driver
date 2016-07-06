import numpy as np
import pandas as pd

sub_1 = pd.read_csv('subm/GoogLeNet_224_fold00_finetune.csv')
sub_2 = pd.read_csv('subm/GoogLeNet_224_fold01_finetune.csv')
sub_3 = pd.read_csv('subm/GoogLeNet_224_fold02_finetune.csv')
sub_4 = pd.read_csv('subm/GoogLeNet_224_fold03_finetune.csv')
sub_5 = pd.read_csv('subm/GoogLeNet_224_fold04_finetune.csv')
sub_6 = pd.read_csv('subm/GoogLeNet_224_fold05_finetune.csv')
sub_7 = pd.read_csv('subm/GoogLeNet_224_fold06_finetune.csv')
sub_8 = pd.read_csv('subm/GoogLeNet_224_fold07_finetune.csv')
sub_9 = pd.read_csv('subm/GoogLeNet_224_fold08_finetune.csv')
sub_10 = pd.read_csv('subm/GoogLeNet_224_fold09_finetune.csv')
'''
sub_11 = pd.read_csv('subm/resnet110_fullpre_11_tta.csv')
sub_12 = pd.read_csv('subm/resnet110_fullpre_12_tta.csv')
sub_13 = pd.read_csv('subm/resnet110_fullpre_13_tta.csv')
sub_14 = pd.read_csv('subm/resnet110_fullpre_14_tta.csv')
sub_15 = pd.read_csv('subm/resnet110_fullpre_15_tta.csv')
sub_16 = pd.read_csv('subm/resnet110_fullpre_16_tta.csv')
sub_17 = pd.read_csv('subm/resnet110_fullpre_17_tta.csv')
sub_18 = pd.read_csv('subm/resnet110_fullpre_18_tta.csv')
sub_19 = pd.read_csv('subm/resnet110_fullpre_19_tta.csv')
#sub_20 = pd.read_csv('subm/20_tta.csv')
'''
#vgg_preds = pd.read_csv('subm/vgg_preds.csv', index_col=10)
#vgg_aug = pd.read_csv('subm/pretrained_aug.csv', index_col=10)

image_labels = sub_1['img']
#image_reindex = sub_1['img'].as_matrix()

sub_1 = sub_1.drop(['img'], axis=1)
col_names = sub_1.columns.values
sub_2 = sub_2.drop(['img'], axis=1)
sub_3 = sub_3.drop(['img'], axis=1)
sub_4 = sub_4.drop(['img'], axis=1)
sub_5 = sub_5.drop(['img'], axis=1)
sub_6 = sub_6.drop(['img'], axis=1)
sub_7 = sub_7.drop(['img'], axis=1)
sub_8 = sub_8.drop(['img'], axis=1)
sub_9 = sub_9.drop(['img'], axis=1)
sub_10 = sub_10.drop(['img'], axis=1)
#vgg_preds = vgg_preds.set_index(['img'])

#vgg_preds = vgg_preds.reindex(image_reindex)
#vgg_aug = vgg_aug.reindex(image_reindex)

#pseudo_labels_test = vgg_aug.values
#np.save('data/pseudo_labels_test.npy', pseudo_labels_test)

'''
sub_11 = sub_11.drop(['img'], axis=1)
sub_12 = sub_12.drop(['img'], axis=1)
sub_13 = sub_13.drop(['img'], axis=1)
sub_14 = sub_14.drop(['img'], axis=1)
sub_15 = sub_15.drop(['img'], axis=1)
sub_16 = sub_16.drop(['img'], axis=1)
sub_17 = sub_17.drop(['img'], axis=1)
sub_18 = sub_18.drop(['img'], axis=1)
sub_19 = sub_19.drop(['img'], axis=1)
#sub_20 = sub_20.drop(['img'], axis=1)
'''

combo = (sub_1.values + sub_2.values + sub_3.values + sub_4.values + sub_5.values + sub_6.values + sub_7.values + sub_8.values + sub_9.values + sub_10.values) / 10.0

#print combo.shape
#np.save('data/cache/pseudo_labels_test.npy', combo)

#vgg_resnet = (combo + (vgg_aug.values * 3.0)) / 4.0

avg = pd.DataFrame(combo, index=image_labels, columns=col_names)
avg.to_csv('subm/GoogleNet_finetune_ensemble.csv', index_label='img')
