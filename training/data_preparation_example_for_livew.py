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


all_lbs =sio.loadmat('LIVEW_mos')
name = np.array(all_lbs['LIVEW_name'])
num = len(name)

mos = np.array(all_lbs['LIVEW_mos'])[:,0]
mos_std = np.array(all_lbs['LIVEW_mos'])[:,1]


ind = np.arange(num)
np.random.seed(0)
np.random.shuffle(ind)
ind_train = ind[:int(len(ind) * 0.8)]
ind_test = ind[int(len(ind) * 0.8):]

imgs_all = np.zeros((num, 3, 244, 244), dtype=np.uint8)

impath = 'E:\Database\LIVEW\Images'

for i in np.arange(0, num):
    if i<7:
        im = cv2.cvtColor(cv2.imread(impath + '\\' + name[i,0][0][28:], 1), cv2.COLOR_BGR2RGB)
    else:
        im = cv2.cvtColor(cv2.imread(impath + '\\' + name[i][0][0][13:], 1), cv2.COLOR_BGR2RGB)

    # plt.imshow(im)
    # plt.show()
    imgs_all[i] = cv2.resize(im, (244, 244), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

sio.savemat('livew_244.mat', {'X': imgs_all[ind_train], 'Y': mos[ind_train], 'Xtest': imgs_all[ind_test], 'Ytest': mos[ind_test]})

