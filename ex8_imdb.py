# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:46:42 2022

@author: user

Exercise 8: Sentiment analysis with RNN

Text is a type of sequential data.

We are using the IMDB dataset, which containts movie reviews with either
positive or negative review

We are going to perform sentiment analysis by using LSTM
"""

#1. Import the necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,optimizers,callbacks
import tensorflow_datasets as tfds
import numpy as np
import io

#2. Load the data 
(train_data, test_data), info = tfds.load('imdb_reviews',split=(tfds.Split.TRAIN,tfds.Split.TEST)
                                          ,with_info=True,as_supervised=True)

#%%

#Show an example data sample
for feature, label in train_data.take(1):
    print('Text: ', feature.numpy())
    print('Label: ', label.numpy())
    
#%%

#Change the configuration of the dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_dataset = train_dataset.cache()
test_dataset = test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache()

#%%
"""
Sentiment analysis is a type of NLP task. Generally for NLP task, these processes
are involved:
    1. Texts are in strings, we need to convert them into numerical representations
       by carrying out two processes.
    2. First process is called tokenization, this converts each word in your dataset
       into a number. The number depends on your size of vocabulary.
       Vocabulary: The number of different words you want to represent as numbers.
       Any words that are not within your vocabulary are treated equally as
       'Out of Vocabulary (OOV)'.
    3. Second process is word embedding. This converts the word tokens into
       a high dimensional vector. This helps to provide every different word a 
       meaning using a vector as representation.
       
In this example, we are going to include the tokenization and embedding process
as port of the deep learning model. So your model will look like this

Text input --> Tokenization --> Embedding --> LSTM --> Output (Classification)
"""

#2. Define the tokenization layer
VOCAB_SIZE = 1000
tokenization = layers.TextVectorization(max_tokens=VOCAB_SIZE)
tokenization.adapt(train_dataset.map(lambda text, label: text))

#%%

#Show some examples of the vocabulary
vocab = np.array(tokenization.get_vocabulary())
print(vocab[:30])

#%%

#We can pass some text data into the layer and see the result of tokenization process
for  features, labels in train_dataset.take(1):
    print('Text: ', features[0])
    print('Label: ',labels[0])
    
#%%    

for i in range(3):
    print("---------------------Review-------------------------------")
    print(features[i])
    print("---------------------------Tokenized Result--------------------------")
    print(tokenization(features)[i].numpy())
    
#%%

#3. Define the embedding layer
embedding = layers.Embedding(input_dim=len(tokenization.get_vocabulary()),
                             output_dim=64,mask_zero=True)
#Show example of embedding
embedding_example = embedding(tokenization(features)[:3].numpy()).numpy()
print(embedding_example[0])

#%%

#4. Create the model
"""
The model should start with
1. TextVectorization layer
2. Embedding layer
3. LSTM layer
4. Classification layers (Dense layer)
"""

model = keras.Sequential()
model.add(tokenization)
model.add(embedding)
model.add(layers.LSTM(64))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

#%%

#5. Compile model
optimizer = optimizers.Adam(0.0001)
loss = losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

#%%

#6.Train model
history = model.fit(train_dataset, validation_data=test_dataset, validation_steps=30, epochs=10)

#%%

own_review = np.array('This movie is amazing! The story is very well written')
review_np = np.expand_dims(own_review, axis=0)

#%%

prediction = np.argmax(model.predict(review_np))
print(prediction)



