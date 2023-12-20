# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:18:18 2019

@author: Administrator
"""

import numpy as np
from scipy import io as sio
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import cv2
import os
import csv
import pandas as pd

# 读取csv至字典
#

impath = 'E:\\Database\\IQA Database\\## KonIQ-10k Image Database\\koniq10k_1024x768\\koniq10k_1024x768\\'
all_lbs = pd.read_csv('E:\Database\IQA Database\## KonIQ-10k Image Database/koniq10k_scores_and_distributions.csv')
name = np.array(all_lbs['image_name'])
inds = np.argsort(name)
num = len(name)
name=name[inds]
mos = np.array(all_lbs['MOS'])[inds]
mos_zscore = np.array(all_lbs['MOS_zscore'])[inds]



ind = np.arange(num)
np.random.seed(0)
np.random.shuffle(ind)
ind_train = ind[:int(len(ind) * 0.8)]
ind_test = ind[int(len(ind) * 0.8):]

imgs_all = np.zeros((num, 3, 224, 224), dtype=np.uint8)


for i in np.arange(0, num):
    im = cv2.cvtColor(cv2.imread(impath + '\\' + name[i], 1), cv2.COLOR_BGR2RGB)
    # plt.imshow(im)
    # plt.show()
    imgs_all[i] = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

sio.savemat('Koniq_224.mat', {'X': imgs_all[ind_train], 'Y': mos[ind_train], 'Xtest': imgs_all[ind_test], 'Ytest': mos[ind_test]})

