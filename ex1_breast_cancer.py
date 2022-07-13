# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 14:45:45 2022

@author: user

Exercise 1: Construct a FeedForward Neural Network to predict cancer.

1. You are going to use the breast cancer dataset from sklearn.datasets
2. Create a neural network with 3 hidden layers
3. Output layer you can use either softmax or sigmoid activation
4. Display your training result using matplotlib

"""
#1. Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn.datasets as skdatasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2. Import datasets from scikit learn
(bc_features, bc_labels) = skdatasets.load_breast_cancer(return_X_y=True, as_frame=True)

#%%

#3. Perfrom a train test split to obtain training and testing datasets
SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(bc_features, bc_labels, 
                                                    test_size=0.3, random_state=SEED)

#%%

#4. Perform Data Normalization
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)

#%%

#5. Define your Neural Network model
nClass = len(np.unique(y_test)) #If classification, better define the nClass
model = keras.Sequential()

#Start input layer, in this case, we can use normalize layer
model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
#Now we can add the hidden layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
#Add the output layer (be careful about the number of nodes and activation function)
model.add(layers.Dense(nClass, activation='softmax'))

#%%

model.summary() #show the structure of the model

#%%

#6. Compile your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%

#7.Perform model training
BATCH_SIZE = 32
EPOCHS = 20

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

#%%

#8. Visualize the result of model training
training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis , training_loss, label='Training Loss')
plt.plot(epochs_x_axis, val_loss, label='Validation Loss')
plt.title("Training vs Validation loss") # easier to detect over/under fitting using validation plot
plt.legend()
plt.figure()

plt.plot(epochs_x_axis, training_acc, label='Training accuracy')
plt.plot(epochs_x_axis, val_acc, label='validation accuracy')
plt.title('Training vs Validation accuracy')
plt.legend()
plt.figure()

plt.show()

#%%

#9. Make Prediction with your model
predictions = np.argmax(model.predict(X_test), axis=1)

#%%