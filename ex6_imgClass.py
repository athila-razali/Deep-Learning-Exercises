# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 07:24:13 2022

@author: user

Exercise 6: To perform image classification using CNN

1. We are going to use the example dataset from tensorflow:
    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10
2. Once you have loaded the dataset, you will need to normalize the images 
by simply dividing by 255(image pixel format is uint8)
3. Construct a CNN to perform image classification, your model structure
should follow what we have learnt from the theory:
    a. Input Layer
    b. Feature Extraction Layers: Conv + ReLu + Pooling
     (use layers.Conv2D, layers.MaxPool2D to build these layers)
    c. Classification layers: Flatten + Fully-connected
    (use layers.Flatten and layers.Dense)
4. Important hyperparameters to consider:
    a. no. of nodes in Conv layers
    b. kernel size in Conv layers
    c. type of padding in conv layers
    d. kernel size in pooling layers
    e. hyperparameters we have practiced in previous exercises
5. Important note: be careful about the input shape. Input shape should follow 
the shape of the image. Usual image shape format is (width, height, channels)

We are using the CIFAR10 dataset, which contain 10 classes of images:
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
"""

#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, callbacks
#from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, datetime

#2. Load data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

#%%

#3. Create a list for the classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
nClass = len(class_names)

#%%

#4. Display some images as example
plt.figure(figsize=(10,10))

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[y_train[i][0]])
    
plt.show()

#%%
"""
For this images, their pixel format is in uint8, with a range of values
0-255

To normalized the pixel value, we can just use min-max method by dividing
the pixels by 255.
"""

#5. Perform pixel normalization
x_train, x_test = x_train/255.0, x_test/255.0
#Data is ready

#%%

#6. Build the CNN model
img_shape = x_test[0].shape

model = keras.Sequential()
#Add the input layer
model.add(layers.InputLayer(input_shape=img_shape))
#Add the feature extraction layers (Conv + Relu + Pooling)
model.add(layers.Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
#Add classification layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#Add output layer (softmax)
model.add(layers.Dense(nClass, activation='softmax'))

#Print out the structure of the model
model.summary()

#%%

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%

#Define the callbacks
base_log_path = r"C:\Users\user\Desktop\AI07\DeepLearning\tb_logs"
log_path = os.path.join(base_log_path, 'exercise_6', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_dir=log_path)

#%%

#Train the model
EPOCHS = 32
BATCH_SIZE = 10

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb])

#%%

#6. Save an entire model
root_path = r"C:\Users\user\Desktop\AI07\DeepLearning\saved_models"
save_path = os.path.join(root_path, 'simple_CNN.h5')
model.save(save_path)

#%%

#7. load your saved model
new_model = keras.models.load_model(save_path)





