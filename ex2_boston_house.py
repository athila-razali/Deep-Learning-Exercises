# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 16:20:55 2022

@author: user

Exercise 2: Construct a model to predict house price in Boston.

1. Use the sklearn.datasets load_boston
2. Create a FeedForward NN with 3 hidden layers, use Functional API to create model.
3. Some points to take note:
    a. This is a regression problem
    b. Need to carefully choose your activation functions, 
    no. of nodes in output layer, evaluation metrics, and loss function.
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

#2. Data Preparation - load datasets from scikit-learn
(boston_features, boston_target) = skdatasets.load_boston(return_X_y=True)

#%%

#3. Perform a train test split to obtain training and testing dataset
SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(boston_features, boston_target, test_size=0.2, random_state=SEED)

#%%

#4. Data Normalization
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)

#%%

#5. Define your Neural Network model - Functional API
#Start with input layer
nIn = X_train.shape[1]
inputs = keras.Input(shape=(nIn,))
h1 = layers.Dense(128, activation='elu')
h2 = layers.Dense(64, activation='elu')
h3 = layers.Dense(32, activation='elu')
out_layer = layers.Dense(1)

#Use Functional API to link layers together
x = h1(inputs)
x = h2(x)
x = h3(x)
outputs = out_layer(x)

#Create a model by using the model object
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
#%%

#6. Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#%%

#7.Perform model training
BATCH_SIZE = 32
EPOCHS = 30

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

#%%

#8. Visualize the result of model training
training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_acc = history.history['mae']
val_acc = history.history['val_mae']
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis , training_loss, label='Training Loss')
plt.plot(epochs_x_axis, val_loss, label='Validation Loss')
plt.title("Training vs Validation loss") # easier to detect over/under fitting using validation plot
plt.legend()
plt.figure()

plt.plot(epochs_x_axis, training_acc, label='Training mae')
plt.plot(epochs_x_axis, val_acc, label='Validation mae')
plt.title('Training vs Validation mae')
plt.legend()
plt.figure()

plt.show()

#%%

#9. Make Prediction with your model
predictions = model.predict(X_test)
pred_vs_label = np.concatenate((predictions, np.expand_dims(y_test, axis=1)), axis=1)

#%%