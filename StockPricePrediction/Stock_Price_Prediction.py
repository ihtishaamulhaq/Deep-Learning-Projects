# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 12:04:50 2023

@author: Ihtishaam
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

import warnings
warnings.filterwarnings('ignore')

# Read the csv file and show some rows
stock_data=pd.read_csv('C:\\Users\Ihtishaam\Deep-Learning-Projects\StockPricePrediction\Stocks_dataset.csv')
stock_data_head=stock_data.head()
print(stock_data_head)

# Data shape
data_shape=stock_data.shape
print(data_shape)

# Dataset Summary
stock_info=stock_data.info()
print(stock_info)

# Statistical info of the data
stock_des=stock_data.describe()
print(stock_des)

# Check null values
null_values=stock_data.isnull().sum()
print(null_values)

# Show the columns name
columns_names=stock_data.columns
print(columns_names)

# Extracting the require columns
dataset=stock_data[['date','open','close']]
print(dataset)

# Convert it to datetime dtype
dataset['date']=pd.to_datetime(dataset['date'].apply(lambda Y: Y.split()[0]))
dataset.set_index('date', drop=True, inplace=True)
print(dataset.head())

#  Plotting the open and close price on date index
plt.figure(figsize=(8,8))
plt.plot(dataset['open'], label='Open Price', color='green')
plt.xlabel('Date',size=14)
plt.ylabel('Price',size=14)
plt.legend(loc='upper left')
plt.savefig('ActualOpenPrice.png')
plt.show()

plt.figure(figsize=(8,8))
plt.plot(dataset['close'], label='Close Price', color='blue')
plt.xlabel('Date',size=14)
plt.ylabel('Price',size=14)
plt.legend(loc='upper left')
plt.savefig('ActualClosingPrice.png')
plt.show()

# Perform preprocessing on the data
mmscaler=MinMaxScaler()
dataset[dataset.columns]=mmscaler.fit_transform(dataset)
print(dataset)

#split data into Training and testing
train_size = round(len(dataset) * 0.75)


train_data = dataset[:train_size]
test_data = dataset[train_size:]
print(train_data.shape)
print(test_data.shape)

def create_sequence(dataset):
    sequences = []
    label = []
    size=60
    data=len(dataset)
    sr_index = 0
    
    for e_idx in range(size, data): 
        sequences.append(dataset.iloc[sr_index:e_idx])
        label.append(dataset.iloc[e_idx])
        sr_index= sr_index + 1
    return (np.array(sequences), np.array(label))



train_data_seq, train_label = create_sequence(train_data)
test_data_seq, test_label = create_sequence(test_data)
print(train_data_seq.shape)
print(train_label.shape)
print(test_data_seq.shape) 
print(test_label.shape)

# Building the model
regressor_model=Sequential()

regressor_model.add(LSTM(units=40, return_sequences=True, input_shape=(train_data_seq.shape[1], train_data_seq.shape[2])))
regressor_model.add(Dropout(0.1))

regressor_model.add(LSTM(units=40))



regressor_model.add(Dense(units=2))

# compile the model
regressor_model.compile(loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['mean_absolute_error'])

#  Show the model Summary
regressor_model.summary()

# fitting the model 
regressor_model.fit(train_data_seq, 
                    train_label, 
                    epochs = 50, 
                    validation_data = (test_data_seq, test_label),
                    verbose = 1)






