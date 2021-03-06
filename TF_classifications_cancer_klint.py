# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:29:32 2021

@author: Klint Mane

This code follows the lecture on TF classifications. I try to use neural networks to classify types of cancer from real data.

"""

##=========================================================================##
# Imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
##=========================================================================##


##=========================================================================##
# Data

os.chdir("C:/Users/klint/Desktop/Ph.D/Python")  ## Setting the directory
df = pd.read_csv("TensorFlow_FILES/DATA/cancer_classification.csv")
##=========================================================================##


##=========================================================================##
# Exploratory analysis

df.describe().transpose()
sns.countplot(x='benign_0__mal_1', data=df)
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
sns.heatmap(df.corr())
##=========================================================================##


##=========================================================================##
# Preparing the data

X = df.drop('benign_0__mal_1', axis=1).values  ## Remember we need the .values so that we do not input a panda frame but a simple numpy array
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
##=========================================================================##


##=========================================================================##
# Preprocessing the data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
##=========================================================================##


##=========================================================================##
# Training the Model 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid')) ## For Binary classification we want sigmoid to be our activation function
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))
##=========================================================================##


##=========================================================================##
# Evaluating the model

losses = pd.DataFrame(model.history.history)
losses.plot()  ## After 20 epochs or so we overfit the training data
##=========================================================================##


##=========================================================================##
# Using callbacks to stop overfitting

model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1,activation='sigmoid')) ## For Binary classification we want sigmoid to be our activation function
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()  ## After 20 epochs or so we overfit the training data
##=========================================================================##


##=========================================================================##
# Using dropout to stop overfitting

from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid')) ## For Binary classification we want sigmoid to be our activation function
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()  ## After 20 epochs or so we overfit the training data
##=========================================================================##


##=========================================================================##
# Evaluation

predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))
##=========================================================================##
