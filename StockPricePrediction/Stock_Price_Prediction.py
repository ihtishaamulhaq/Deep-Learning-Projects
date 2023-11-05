# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 12:04:50 2023

@author: Ihtishaam
"""

import pandas as pd
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