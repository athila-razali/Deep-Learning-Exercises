# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:54:56 2022

@author: user

Exercise 3: To predict diabetes using neural network.

1. Use the sklearn.dataset to load_diabetes (take note, it is a regression pronlem)
2. You are going to perform k-fold cross validation, with k=5
3. Record the evaluation for each fold, and calculate the average scores for both loss
and evaluation metrics.

"""
#1. Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn.datasets as skdatasets
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2. Import datasets from scikit-learn
(dbt_features, dbt_target) = skdatasets.load_diabetes(return_X_y=True, as_frame=True)

#%%

#3. Prepare k-fold data
SEED = 12345
kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)

#%%

#4. Define model
nIn= dbt_features.shape[1]

# Create model with Functional API
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

#%%
from sklearn.preprocessing import StandardScaler
#5. Loop through KFold
features = np.array(dbt_features)
labels = np.array(dbt_target)
nfold = 1

#empty list to hold the score
loss_list = []
mae_list = []

for train, test in kfold.split(features, labels):
    train_features = features[train]
    train_labels = labels[train]
    test_features = features[test]
    test_labels = labels[test]
    
    #Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    #Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #perform model training
    print('#######################################') 
    print(f'Training for fold number {nfold}...')
    history = model.fit(train_features,train_labels,validation_data=(test_features,test_labels),batch_size=32, epochs=10)
    #Evaluate model
    scores = model.evaluate(test_features, test_labels)
    #generate the score
    for metric_name,score in zip(model.metrics_names,scores):
        print(f'{metric_name}:{score}')
    loss_list.append(scores[0])
    mae_list.append(scores[1])
    nfold+=1
    keras.backend.clear_session()
    
    
#%%

#6. Print the average scores
print('average loss: ',np.mean(loss_list))
print('average mae: ', np.mean(mae_list))