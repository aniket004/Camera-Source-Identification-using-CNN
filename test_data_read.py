#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:29:07 2017

@author: aniketr
"""

# data prepare for CNN

import numpy as np
import pandas as pd
import cv2
import glob

import os

# modified COVERAGE have 100 forged image
files = os.listdir('/home/ms/aniketr/Aniket/python/COVERAGE_Modified/image_tampered')


test_img = []
test_mask = []
for i in range(0,100):
    img = cv2.imread('/home/ms/aniketr/Aniket/python/COVERAGE_Modified/image_tampered/'+str(i+1)+'t.tif')
    test_img.append(img)
    mask = cv2.imread('/home/ms/aniketr/Aniket/python/COVERAGE_Modified/mask/'+str(i+1)+'forged.tif')
    test_mask.append(mask)


np.save('test_img',np.array(test_img))
np.save('test_mask',np.array(test_mask))



#def read_img_from_folder():
#    iamge_stack = []
#    mask_stack = []
#    for img in glob.glob('path/*.jpg'):
#        image_stack.append(cv2.imread(img))
#    return image_stack 

