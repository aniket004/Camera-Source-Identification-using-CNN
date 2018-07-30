# Camera-Source-Identification-using-CNN

1. Code implemented in Keras.

2. cnn.py and cnn_256.py are files implemented for camera source identification.

3. CNN model using 3 convolution layer.

4. Image patches from dataset are collected in .csv file.

5. In cnn.py, preprocessing, i.e., highpass filtering has been performed before feeding data to CNN module. 

6. Input : Image patches from 10 camera models from Dresden Image Dataset.

7. Output: camera source of test image.

8. Model gives good classification accuracy, i.e., ~96 % for 10 camera models. 
