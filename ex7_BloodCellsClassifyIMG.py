# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:10:29 2022

@author: user

Exercise 7: To create a CNN to classify the blood cells - Loading data externally
for image classification

1. Use this dataset:
https://www.kaggle.com/datasets/paultimothymooney/blood-cells
2. We just need to use the dataset2-master folder(don't need to care about the csv file)
3. To load the data fromafolder,you can use this method:
https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
4. Construct a CNN to perform image classification
5. Refer back to your Exercise 6 for the model

The main point is to practice how to load data from the usual format

The usual format of data annotation for image classification is:
    1. One folder for each class(folder name will be the name of class)
    2. Images of the corresponding classes are stored within their respective folder
    
Tensorflow provides a method for us to load image dataset from such format.
"""

#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, datetime, pathlib

#2. Load image from the files
root_path = r"C:\Users\user\Desktop\AI07\DeepLearning\dataset2-master\dataset2-master\images"
train_path = os.path.join(root_path, "TRAIN")
val_path = os.path.join(root_path, "TEST")
test_path = os.path.join(root_path, "TEST_SIMPLE")

train_dir = pathlib.Path(train_path)
val_dir = pathlib.Path(val_path)
test_dir = pathlib.Path(test_path)

SEED = 12345
IMG_SIZE = (160,160)
BATCH_SIZE = 16
train_data = keras.utils.image_dataset_from_directory(train_dir, seed=SEED, 
                                                      image_size=IMG_SIZE, 
                                                      batch_size=BATCH_SIZE)
val_data = keras.utils.image_dataset_from_directory(val_dir, seed=SEED, 
                                                      image_size=IMG_SIZE, 
                                                      batch_size=BATCH_SIZE)
test_data = keras.utils.image_dataset_from_directory(test_dir, seed=SEED, 
                                                      image_size=IMG_SIZE, 
                                                      batch_size=4)

#%%

#The dataset is in BatchDataset, now convert it into PrefetchDataset

AUTOTUNE = tf.data.AUTOTUNE
train_pf = train_data.prefetch(buffer_size=AUTOTUNE)
val_pf = val_data.prefetch(buffer_size=AUTOTUNE)
test_pf = test_data.prefetch(buffer_size=AUTOTUNE)

#%%

#3. Create the model
class_names = train_data.class_names
IMG_SHAPE = IMG_SIZE + (3,)
nClasses = len(class_names)

model = keras.Sequential()
#The rescaling performs normalization, it will also be the input layer
model.add(layers.Rescaling(1./255, input_shape=IMG_SHAPE))
#Feature Extraction layers
model.add(layers.Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32,(3,3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2,2)))
#Add the classification layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#Add the output layer
model.add(layers.Dense(nClasses, activation='softmax'))

model.summary()

#%%

#Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%

#Define the callbacks
base_log_path = r"C:\Users\user\Desktop\AI07\DeepLearning\tb_logs"
log_path = os.path.join(base_log_path, 'exercise_7', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_dir=log_path)

#%%

#train the model
EPOCHS = 10
history = model.fit(train_pf, validation_data=val_pf, epochs=EPOCHS)

#%%


