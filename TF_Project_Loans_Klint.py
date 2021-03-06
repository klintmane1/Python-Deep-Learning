# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:05:32 2021

@author: Klint Mane

This code uses real world data from bank loans. Using tenserflow, I try to create a model that predicts the probability of an individual to pay back his loan. The aim is to train a model with the purpose of predicting the probability of a new costumer to pay back his loan based on his data.

"""

#%% Imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


#%% Uploading data

os.chdir("C:/Users/klint/Desktop/Ph.D/Python")  ## Setting the directory
df = pd.read_csv('TensorFlow_FILES/DATA/lending_club_loan_two.csv')
df.info()


#%% Data Exploration

sns.countplot('loan_status', data=df)  ## More loans are payed back which makes sense and is expected
plt.show()
sns.displot(x='loan_amnt',bins =25 , data=df)  ## Most loans are around 10k$.
plt.show()
sns.heatmap(data = df.corr(), cmap="coolwarm", annot = True)  ## Installment and total loan amount are highly correlated which makes a lot of sense
plt.show()
sns.scatterplot(x='installment', y='loan_amnt', data =df)  ##  I think there is duplicate information here and would probably keep only one of the variables
plt.show()
sns.boxplot(x='loan_status' , y='loan_amnt', data = df)  ## Looks like loans that failed were a bit higher on average
plt.show()
df['loan_amnt'].groupby(df["loan_status"]).describe()

sorted(df['grade'].unique())
sorted(df['sub_grade'].unique())

sns.countplot(x='grade', hue='loan_status', data = df)  ## Really cool graph. We can see that loans of grade E,F,G have a high probability of failure
plt.show()
sns.countplot(x='sub_grade', order= sorted(df['sub_grade'].unique()), palette= "light:#5A9" ,  data=df)
plt.show()
sns.countplot(x='sub_grade', order= sorted(df['sub_grade'].unique()), hue='loan_status' ,  data=df)
plt.show()

df1 = df[df['grade'].isin(['F','G'])]
sns.countplot(x='sub_grade', order= sorted(df1['sub_grade'].unique()), hue='loan_status',  data = df1)
plt.show()

#%% Data Engineering


df['loan_repaid'] = df['loan_status'].isin(['Fully Paid'])*1  ## Times 1 is a cool trick to make the column zero or one istead of true and false

df[['loan_repaid','loan_status']].head(10)

pd.DataFrame(df.corr()['loan_repaid'][:-1]).sort_values('loan_repaid').plot.bar()
plt.show()

#%% Data PreProcessing

# Missing Data
len(df)  ## Checking the length of the dataframe
df.isnull().sum().sort_values()*100/len(df)  ## Employment title has 5% missing values
len(df['emp_title'].unique())  ## Unique emp titles
df['emp_title'].value_counts()  ## Too many unique values to be useful

df = df.drop('emp_title', axis=1)  ##  Have to assing the operation to df otherwise the change does not happen
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']
sns.countplot(x = 'emp_length', order= emp_length_order, data=df)
plt.show()
sns.countplot(x = 'emp_length', order= emp_length_order, hue='loan_status' , data=df)
plt.show()

# Getting the percentage of failed loans based on years of employment
emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_total = df.groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_total
print(emp_len)  ## Basically the number of years the person has been working seem to have no effect on whether they pay the loan back
emp_len.plot(kind='bar')
df = df.drop('emp_length', axis=1)

# What missing data is left
df.isnull().sum().sort_values()
df[['title', 'purpose']].head(10)  ## Title and purpose seem to have the same info and purpose has no missing data
df.drop('title', axis=1, inplace=True)
df['purpose'].value_counts()
sns.countplot(x = 'purpose',  hue='loan_status' , data=df)  ## Trying to see whether purpose matters (it probably should)
plt.show()

# Dealing with mort_acc (number of mortgage accounts)
df['mort_acc'].value_counts()
df.corr()['mort_acc'].sort_values()
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']

# # My solution works as well. His is more efficient I believe.

# import math
# for index, item in enumerate(df['mort_acc']):
#     if math.isnan(item):
#         items_total_acc = df.iloc[index, 18]
#         df.iloc[index, 21] = total_acc_avg[items_total_acc]
        
# df.isnull().sum().sort_values()

# Letcure solution
def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping            of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc] 
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

df = df.dropna()
df.isnull().sum().sort_values()  ## No more missing values now


#%% String variables to dummies


df.select_dtypes(['object']).columns

df['term'] = df['term'].apply(lambda term: int(term[:3]))

df = df.drop('grade', axis = 1)  ## Since we will already use subgrade

# Converting sub_grade to dummies
subgrade_dummies = pd.get_dummies(data = df['sub_grade'], drop_first=True)
df = pd.concat([df.drop('sub_grade', axis=1), subgrade_dummies], axis=1)
df.columns

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

df['home_ownership'].value_counts()
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

# Address
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
df['address'].head()
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

# Issue date
df = df.drop('issue_d', axis=1)

# Earliest cr line
df['earliest_cr_line'].head()
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x: int(x[-4:]))
df = df.drop('earliest_cr_line', axis=1)

df.select_dtypes(['object']).columns

df = df.drop('loan_status', axis=1)
## No more string values


#%% Train Test Split

X = df.drop('loan_repaid', axis=1).values
y = df['loan_repaid'].values

y = y.astype(np.float)  ## Model fit was not working since y was int32 type instead of float64.


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

#%% Normalizing Data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  ## We do not fit on the test set

#%% Creating the model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation
from tensorflow.keras.constraints import max_norm

model = Sequential()
# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.fit(x=X_train, 
          y=y_train, 
          epochs=100,
          batch_size=256,
          validation_data=(X_test, y_test), 
          callbacks=[early_stop],
          )
## Model fit was not working since y was int32 type instead of float64.

#%% Saving model
from tensorflow.keras.models import load_model
model.save('full_data_project_model.h5')  

#%% Evaluating

losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()

from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)


#%% Given the customer below, would you offer this person a loan?

import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer

print("We would do this action" + str(model.predict_classes(new_customer.values.reshape(1,78))))
df.iloc[random_ind]['loan_repaid']