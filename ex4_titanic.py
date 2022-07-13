# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 01:17:24 2022

@author: user

Exercise 4:To predict if a passenger survives in the Titanic incident.

1. Use the titanic dataset from Kaggle: https://www.kaggle.com/competitions/titanic/data
2. Use train.csv todo everything
3. Perform necessary stepsfor data processing
4. Construct a neural network with 5 hidden layers.
5. Apply early stopping to prevent the model to be overfitting
6. Observe the model training result with Tensorboard
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
base_path = r"C:\Users\user\Desktop\AI07\DeepLearning\titanic data"
train_path = os.path.join(base_path, 'train.csv')
test_path = os.path.join(base_path,'test.csv')
test_label_path = os.path.join(base_path, 'gender_submission.csv')

train_data = pd.read_csv(train_path)
test_features = pd.read_csv(test_path)
test_labels = pd.read_csv(test_label_path)

#%%

#3. Data preparation
print("Train data: \n",train_data.isna().sum())
print("Test features: \n",test_features.isna().sum())
print("Test labels: \n",test_labels.isna().sum())

#%%

#(a) Drop unwanted features
train_data = train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#(b) Drop missing values from 'Embarked'
train_data = train_data.dropna(subset=['Embarked'])

#(c) Do the same thing for test data
test_features = test_features.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


#%%

#(d) One hot encode the categorical features
train_data = pd.get_dummies(data=train_data)
test_features = pd.get_dummies(data=test_features)

#%%

#(e) Impute data to fill up the missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
train_data_imputed = imputer.fit_transform(train_data)
test_features_imputed = imputer.fit_transform(test_features)

#%%

#(f) extract out the training labels
train_features = train_data_imputed[:,1:] 
train_labels = train_data_imputed[:,0]

#%%

#(g) Perform train test split on train data (for train-validation split)
SEED = 12345
x_train, x_test, y_train, y_test = train_test_split(train_features,train_labels,test_size=0.2,random_state=SEED)

#%% 

#(h) Data normalization
standardizer = StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)
wild_test = standardizer.transform(test_features_imputed)

#%%

#4. Model creation
nClass = len(np.unique(y_test))
nIn = x_train.shape[1]

#Use functional API to build NN
inputs = keras.Input(shape=(nIn,))
h1 = layers.Dense(128,activation='relu')
h2 = layers.Dense(128,activation='relu')
h3 = layers.Dense(64,activation='relu')
h4 = layers.Dense(64,activation='relu')
h5 = layers.Dense(32,activation='relu')
out_layer = layers.Dense(nClass,activation='softmax')

#Chain the layers with functional API
x = h1(inputs)
x = h2(x)
x = h3(x)
x = h4(x)
x = h5(x)
outputs = out_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs,name='titanic_model')
model.summary()

#%%

#Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#%%

#Early stopping callback
es = EarlyStopping(patience=10,verbose=1,restore_best_weights=True)
#TensorBard callback
base_log_path = r"C:\Users\user\Desktop\AI07\DeepLearning\tb_logs"
log_path = os.path.join(base_log_path, 'exercise_4', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=log_path)

#%%
BATCH_SIZE = 32
EPOCHS = 200
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[es,tb])

#%%

#Define model
predictions = np.argmax(model.predict(wild_test),axis=1)
labels = np.array(test_labels['Survived'])
prediction_vs_labels = np.transpose(np.vstack((predictions,labels)))
