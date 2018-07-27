#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:11:20 2017

@author: aniketr
"""

# importing required libraries

from keras.models import Sequential
#from scipy.misc import imread
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
#train=pd.read_csv("R/Data/Train/train.csv")
#test=pd.read_csv("R/Data/test.csv")
#train_path="R/Data/Train/Images/train/"
#test_path="R/Data/Train/Images/test/"

from scipy.misc import imresize
# preparing the train dataset

# loaded image and reshape
loaded_image = np.load('/home/ms/aniketr/Aniket/python/training/fake/image.npy')
loaded_mask = np.load('/home/ms/aniketr/Aniket/python/training/fake/mask.npy')

size = np.shape(loaded_image)[0]

from scipy import misc

reshaped_image = []
reshaped_mask = []

for i in range(0,size):
    reshaped_new_image = misc.imresize(loaded_image[i],(224,224,3))
    reshaped_image.append(reshaped_new_image)
    
    reshaped_new_mask = misc.imresize(loaded_mask[i],(224,224))
    reshaped_mask.append(reshaped_new_mask)


## Load test samples for COVERAGE dataset

test_img = np.load('/home/ms/aniketr/Aniket/python/COVERAGE_Modified/image_tampered/test_img.npy')
test_mask = np.load('/home/ms/aniketr/Aniket/python/COVERAGE_Modified/image_tampered/test_mask.npy')

size = np.shape(test_img)[0]

reshaped_test_img =[]
reshaped_test_mask =[]

for i in range(0,size):
    reshaped_test_image = misc.imresize(test_img[i],(224,224,3))
    reshaped_test_img.append(reshaped_test_image)
    
    reshaped_tst_mask = np.invert(misc.imresize(test_mask[i],(224,224)))
    reshaped_test_mask.append(reshaped_tst_mask)


##converting train images to array and applying mean from keras.models import Sequential
#   ...: #from scipy.misc import imread
#   ...: #get_ipython().magic('matplotlib inline')
#   ...: import matplotlib.pyplot as plt
#   ...: import numpy as np
#   ...: import keras
#   ...: from keras.layers import Dense
#   ...: import pandas as pd
#   ...: 
#   ...: from keras.applications.vgg16 import VGG16
#   ...: from keras.preprocessing import image
#   ...: from keras.applications.vgg16 import preprocess_input #subtraction processing

# Assigning traning and test images 

train_img = np.array(reshaped_image,dtype = 'float64')
train_img=preprocess_input(train_img)

test_img = np.array(reshaped_test_img,dtype='float64')
test_img=preprocess_input(test_img)


#train_img=[]
#for i in range(len(train)):
#
#    temp_img=image.load_img(train_path+train['filename'][i],target_size=(224,224))
#
#    temp_img=image.img_to_array(temp_img)
#
#    train_img.append(temp_img)
#
###converting train images to array and applying mean from keras.models import Sequential
##   ...: from scipy.misc import imread
##   ...: get_ipython().magic('matplotlib inline')
##   ...: import matplotlib.pyplot as plt
##   ...: import numpy as np
##   ...: import keras
##   ...: from keras.layers import Dense
##   ...: import pandas as pd
##   ...: 
##   ...: from keras.applications.vgg16 import VGG16
##   ...: from keras.preprocessing import image
##   ...: from keras.applications.vgg16 import preprocess_inputsubtraction processing
#
#train_img=np.array(train_img) 
#train_img=preprocess_input(train_img)
## applying the same procedure with the test dataset
#
#test_img=[]
#for i in range(len(test)):
#
#    temp_img=image.load_img(test_path+test['filename'][i],target_size=(224,224))
#
#    temp_img=image.img_to_array(temp_img)
#
#    test_img.append(temp_img)
#
#test_img=np.array(test_img) 
#test_img=preprocess_input(test_img)

# loading VGG16 model weights
model = VGG16(weights='imagenet', include_top=False)
# Extracting features from the train dataset using the VGG16 pre-trained model

features_train=model.predict(train_img)
# Extracting features from the train dataset using the VGG16 pre-trained model

features_test=model.predict(test_img)

# flattening the layers to conform to MLP input

train_x=features_train.reshape(449,7*7*512)
# converting target variable to array

train_y=np.asarray(train['label'])
# performing one-hot encoding for the target variable

train_y=pd.get_dummies(train_y)
train_y=np.array(train_y)
# creating training and validation set

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(train_x,train_y,test_size=0.3, random_state=42)

 

# creating a mlp model
from keras.layers import Dense, Activation
model=Sequential()

model.add(Dense(1000, input_dim=7*7*512, activation='relu',kernel_initializer='uniform'))
keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

model.add(Dense(500,input_dim=1000,activation='sigmoid'))
keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

model.add(Dense(150,input_dim=500,activation='sigmoid'))
keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# fitting the model 

model.fit(X_train, Y_train, epochs=20, batch_size=128,validation_data=(X_valid,Y_valid))