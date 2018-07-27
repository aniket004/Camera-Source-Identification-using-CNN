# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:22:41 2017

@author: aniketr
"""

#import files
from keras.models import Sequential
from keras.layers import Dense , Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import SGD
from keras import regularizers
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from skimage.util.shape import view_as_blocks
from scipy.signal import convolve2d

#patch_size = 64
#def make_patch(y):
#    x = y[0]
#    print(x.shape)
#    patches = []
#    
#    for i in range(0,x.shape[0],patch_size):
#        for j in range(0,x.shape[1],patch_size):
#            if (i+patch_size) < x.shape[0]:                
#                patches.append(x[i:(i+patch_size-1),j:(j+patch_size-1)])
#                i = i+patch_size-1
#               # print(patches.size)
#            else:
#                if (j+patch_size) < x.shape[1]:                    
#                    patches.append(x[i:(i+patch_size-1),j:(j+patch_size-1)])
#                    j = j+patch_size-1
#                    
#                    return np.array(patches)
#                #else:
#                #    patches.append(x[i:(i+patch_size-1),j:(j+patch_size-1)])                                      

                    




#import data
df_data = pd.read_csv('Dresden_10_cam_100_gray_img_126_crop_img.csv',header = None)
df_data = df_data.values
#df_data = df_data.values/255.0
df_data  = df_data.reshape((1000,1,256,256))
df_label = pd.read_csv('Dresden_10_cam_100_gray_img_label.csv',header = None)
df_label = df_label.values
#labels = df_label.copy()

from skimage import io 


# high pass filtering
filter_basis = np.array([[ -1, 2, -2, 2, -1],
                    [ 2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [ 2, -6, 8, -6, 2],
                    [ -1, 2, -2, 2, -1] ])

filter = 0.083*filter_basis

new_data = []
new_label = []
for x in range(0,1000):
    img_label = df_label[0][x]
    img_data = df_data[x][0]
    conv_data = convolve2d(img_data,filter,mode = 'same')
#    max_val = np.amax(conv_data)
#    print(max_val,conv_data.max(),conv_data.min())
#    conv_data = conv_data/max_val
    #conv_data = conv_data/255.0
    B = view_as_blocks(conv_data, block_shape = (64,64))
    #B = view_as_blocks(df_data[x][0], block_shape = (64,64))
    # each image is divided into 16 blocks of size 64X64
    C = B.reshape(16,1,64,64)
    #np.append(new_data,C)
    new_data.append(C)
    mod_label = img_label* np.ones(16)
    #np.append(new_label,mod_label)
    new_label.append(mod_label)


new_data = np.array(new_data)
new_label = np.array(new_label)

new_data = new_data.reshape((16000,1,64,64))
new_label = new_label.reshape((16000,1))

del df_data
del df_label

df_data = new_data

#df_data = (df_data-df_data.min())/(df_data.max()-df_data.min())
df_data = (df_data - np.mean(df_data,axis=0))/np.std(df_data,axis=0)
df_label = new_label
labels = new_label.copy()

#new_data = np.zeros((16000,1,64,64))
#new_label = np.zeros((1600,1))
#
#for x in range(0,999):
#    img = df_data[x]
#    img_label = labels[x]
#    for y in range(0,15):
#        new_img = img.reshape(16,1,64,64)
#        #new_data =np.hstack(new_img)
#         print(x,y)
#        new_data[(16*x+y):(16*(x+1)+y),:,:,:] = new_img
#        new_label[(16*x+y):(16*(x+1)+y)] = img_label
#       



## randomly generate 60% training and 40% test index
#train_index = np.random.randint(0,1000, size=(1,600))
#test_index = ~train_index
#train_id, test_id = train_test_split(df_label[:],test_size=0.6, random_state = 1)

#train_id = df_label.sample(frac=0.6,random_state=1)
#test_id = df_label.drop(train_id)
#
## convert dataframe to arrays
#df_data = df_data.values
#df_label = df_label.values

# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)



# size of imges and num_class
num_img = df_label.size  # number of image
num_class = np.max(df_label[:]) + 1  # as classes start from 0

#normalize input 0-255 to 0-1.

# create the model
model = Sequential()
model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode = 'same', input_shape = (1,64,64), activation = 'relu'))
model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode = 'same', activation = 'relu' ))
model.add(Convolution2D(32,3,3, subsample = (1,1), border_mode = 'same', activation = 'relu' ))
model.add(MaxPooling2D( pool_size = (2,2), border_mode = 'same' ))
model.add(Flatten())
model.add(Dense(256, activation = 'relu' ))
#model.add(Dense(4096, activation = 'relu' ))
model.add(Dense(num_class, activation = 'softmax'))

#Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, decay = 1e-5, momentum = 0.9, nesterov= True )
#model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#print(model.summary())

# Fit the model
#model.fit(X_train,y_train, validation_data = (X_test,y_test), nb_epoch = epochs, batch_size = 32 )

# Final evaluation of the model
#scores = model.evaluate(X_test,y_test,verbose=0)

# 10 fold cross valiation
folds = StratifiedKFold(df_label[:,0], n_folds=5)   #0,:

from sklearn import preprocessing
one=preprocessing.OneHotEncoder(sparse=False)
df_label = one.fit_transform(df_label)  #df_label.T

val_acc=[]
test_acc=[]
for tr_ix, test_ix in folds:
#    print(df[tr_ix,:-1],)
    print(len(tr_ix),len(test_ix))
    #print(df_data[tr_ix*256:(tr_ix+1)*256,:].shape,df_label[tr_ix].shape)
    model = Sequential()
    model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode = 'same', input_shape = (1,64,64), activation = 'relu'))
    model.add(MaxPooling2D( pool_size = (2,2), border_mode = 'same' ))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode = 'same', activation = 'relu' ))
    model.add(MaxPooling2D( pool_size = (2,2), border_mode = 'same' ))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(32,3,3, subsample = (1,1), border_mode = 'same', activation = 'relu' ))
    model.add(MaxPooling2D( pool_size = (2,2), border_mode = 'same' ))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    #model.add(Dense(4096, activation = 'relu' ))
    model.add(Dense(num_class, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    model.fit(df_data[tr_ix], df_label[tr_ix], nb_epoch=25,verbose =1)

    y_pred = model.predict(df_data[test_ix,:])

    print (classification_report(labels[test_ix], np.argmax(y_pred,axis=1)))