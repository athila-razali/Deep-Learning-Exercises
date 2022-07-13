# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 03:09:21 2022

@author: user

Exercise 5 : To predict if someone jas a good credit score. But this time,
we want to practice handling data that overfits easily

1. to make your model overfit easily, you are going to one hot encode all the
categorical features in the data(get_dummies method)
2. Build a model that can overfit easily:
    a. 6 hidden layers, each layer with more than 200 nodes
3. Train the model, but dont apply EarlyStopping yet, set an epoch no. of 100,
then observe the graph in TensorBoard
4. Apply the techniques to minimize overfitting:
    a. reduce no. of layers and nodes
    b. apply dropout layer (layers.Dropout)
    c. apply weight decay/regularization*keras.regularizers->apply in hidden layers
    d. apply Earlystopping callback
5. Retain your model, and observe the graph
"""
#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os, datetime

#2. Load the csv data
file_path = r"C:\Users\user\Desktop\AI07\DeepLearning\germanCredit.csv"
data = pd.read_csv(file_path, sep=" ", header=None)

#%%

#Inspect for missing values
print(data.isna().sum())

#%%

#3. Data Preparation
#(a) Change our label so that it is  the sameas label encoding
data[20] = data[20] - 1

#(b) Split the data into features and labels
features = data.copy()
labels = features.pop(20)

#%%

#(c) One-hot encode all the categorical features
features = pd.get_dummies(features)

#%%

#(d) Perform train-test split
SEED = 12345
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=SEED)

#%%

#(e) Data Normalization / Feature Scaling
standardizer = StandardScaler()
standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)

#%%

#4. Model Creation
#Define the number of Inputs and Outputs
nIn = X_test.shape[1]
nClass = len(np.unique(y_test))

#Create themodel with Sequential API
l1 = keras.regularizers.L1(l1=0.001)
l2 = keras.regularizers.L2(l2=0.001)

model = keras.Sequential()
#Now we can add the hidden layers
model.add(layers.InputLayer(input_shape=(nIn,)))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=l1))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=l1))
model.add(layers.Dropout(0.3))
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
#Add the output layer (be careful about the number of nodes and activation function)
model.add(layers.Dense(nClass, activation='softmax'))

model.summary() #to check structure of the model
#%%

#Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%%

#Early stopping callback
es = EarlyStopping(patience=10,verbose=1,restore_best_weights=True)
#TensorBard callback
base_log_path = r"C:\Users\user\Desktop\AI07\DeepLearning\tb_logs"
log_path = os.path.join(base_log_path, 'exercise_5', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=log_path)

#%%

# Train the model
EPOCHS = 100
BATCH_SIZE = 32

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb, es])

#%%