# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:34:17 2021
This code trains and evaluates a neural network model to predict house prices in Kansas City. The goal is to predict the price of a new house based on how we trained the model with the existing data.

@author: Klint Mane
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Preparing Data

# Uploading the data
df = pd.read_csv("C:/Users/klint/Desktop/Ph.D/Python/TensorFlow_FILES/DATA/kc_house_data.csv")

# Checking missing data
df.isnull().sum()

# Describing the data 
df.describe().transpose()
sns.displot(df['price'])
sns.countplot(df['bedrooms'])
df.corr()['price'].sort_values()
sns.scatterplot(x='price', y='sqft_living' , data=df)
sns.boxplot(x='bedrooms', y='price', data=df)
sns.scatterplot(x='price', y='long', data=df)
sns.scatterplot(x='price', y='lat', data=df)
sns.scatterplot(x='long', y='lat', data=df, hue='price')

# Cleaning outliers
df.sort_values('price', ascending=False)['price'].head(20)
non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]
sns.scatterplot(x='long', y='lat', data=non_top_1_perc, edgecolor=None, alpha=0.2, palette='RdYlGn', hue='price')

# Cleaning the dataset
df = df.drop('id', axis=1)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)  ## lamda is a short function that we covered
df['month'] = df['date'].apply(lambda date: date.month) 

sns.boxplot(x='month', y='price', data = df)
df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()
df = df.drop('date', axis=1)
df['zipcode'].value_counts()
df = df.drop('zipcode', axis=1)
df['yr_renovated'].value_counts()

#%% Deep Learning Model

# Training the model

X = df.drop('price', axis=1).values ## Remember, .values is reqired to turn it into a numpy array not panda series
y = df['price'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  ## We do not fit on the test set
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(19, activation='relu')) ## Adding 4 layers of 19 nodes
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1)) ## output layer
model.compile(optimizer='adam', loss='mse')
model.fit(x = X_train, y = y_train, validation_data=(X_test, y_test), batch_size=128, epochs= 400) ## We use batches when the data set is large

losses = pd.DataFrame(model.history.history)
losses.plot() ## Since the validation loss is not going up yet, that means that we can actually continue training more, even though it does not look like we are improving much after 50 epochs

# Evaluating the model

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, predictions)))
mean_absolute_error(y_test, predictions) ## Looks like we are off by 100k where prices are about 500k. That is about 20% so I would say we are not doing very well.
explained_variance_score(y_test, predictions)
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')

# Predicting a brand new house

single_house = df.drop('price', axis=1).iloc[0] ## Our model is trained on scaled version of the data not raw
single_house = scaler.transform(single_house.values.reshape(-1,19))
print("Our prediction for the house is: " + str(int(model.predict(single_house))))
print("The real price it sold at was: " + str(df['price'][0]))

