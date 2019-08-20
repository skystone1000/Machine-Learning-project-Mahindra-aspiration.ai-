#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:05:25 2019

@author: skystone
1) Dataframe  - https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
2) 
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os

# Change the working directory
print(os.getcwd())
os.chdir('/home/skystone/Documents/internship/')


# --------------------------  1.1

# Importing the dataset
dataset = pd.read_csv('M&M.csv')

# Taking the series Column
X = dataset.iloc[:, 1].values
tempX = pd.DataFrame(X)

# Delete all the rows int series column which does not contain "EQ"
delete = 0
for category in X:
    if category != 'EQ':
        dataset = dataset.drop(delete)        
    delete = delete + 1

dataset.head()
dataset.tail()
dataset.describe()
dataset.shape
dataset.ndim

# -------------------------- 1.2

# Calculate max min mean price for last 90 days (Price = CLosing price)    
y = dataset.iloc[-90:, 8].values  # -90 for last 90 days or rows of 8th (close price) column
tempy = pd.DataFrame(y)

dataset.iloc[-90:, 8].values.min()
dataset.iloc[-90:, 8].values.max()
dataset.iloc[-90:, 8].values.mean()


# --------------------------- 1.3
"""
To make use of Pandas functionality for dates, you need to ensure that the 
column is of type 'datetime64(ns)'. Changethe date column from 'object' 
type to 'datetime64(ns)' for future convenience.

ref - 
Converting object to datetime64[ns]
https://stackoverflow.com/questions/17705796/convert-to-datetime64-format-with-to-datetime
Adding a new column to existing dataframe pandas
https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
"""
dataset.dtypes  # we have Date column as object

newDates  = dataset.iloc[:,2]
newDates  = pd.to_datetime(newDates).values
newDates.dtype


temp = dataset.drop("Date",axis = 1) # Drop the Dates column
# There is no need of droping as assign() replaces the Date column with Dates

dataset = dataset.assign(Date = newDates)
dataset.dtypes

subOfDates = newDates.max() - newDates.min()
# By subtracting max and min of Date we get timedelta64 object ie diff in ns

# ----------------------------- 1.4
"""
Ref -   
Get year month or day from numpy datetime64
https://stackoverflow.com/questions/13648774/get-year-month-or-day-from-numpy-datetime64
operations on groupby() and iterating through them
https://www.tutorialspoint.com/python_pandas/python_pandas_groupby
Summarising, Aggregating, and Grouping data in Python Pandas
https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
"""

dateIndex = pd.DatetimeIndex(newDates)
Year = dateIndex.year
Month = dateIndex.month

"""
temp.index
temp.columns
temp['Months'] = Month
#temp = temp.insert(2,"Month",'kusdfij')
#temp = temp.insert(2,"Month",Year)
CreatedGroup = temp.groupby('Months')
CreatedGroup.first()
"""


# Add month and year column to dataset
dataset['Month'] = Month
dataset['Year'] = Year

monthWise = dataset.groupby('Month')

monthWise
monthWise.first()
monthWise.get_group(4)  # Get month 4 data 

for name,group in monthWise:
    print(name)
    print(group)
    print(group.mean())
    print(monthWise['Close Price'])
    print(monthWise['Close Price'].mean())
    print(monthWise['Close Price'] #monthWise[])

# ------------------------------ 1.5 
"""
Write a function to calculate the average price over the last N days of 
the stock price data where N is a user defined parameter.
"""


def calcAvg(N):
    avg = dataset.iloc[-N:,].mean()
    return avg
    
allAvg = calcAvg(15)

"""
Write a second function to calculate the profit/loss percentage over the 
last N days.
"""

def profitLossPercent(N):
    
    
    
    

    
"""
Calculate the average price AND the profit/loss percentages over the 
course of last - 1 week, 2 weeks, 1 month, 3 months, 6 months and 1 year.
"""

calcAvg(7)
calcAvg(14)
calcAvg(30)
calcAvg(90)
calcAvg(180)
calcAvg(365)
profitLossPercent(7)
profitLossPercent(14)
profitLossPercent(30)
profitLossPercent(90)
profitLossPercent(180)
profitLossPercent(365)


# --------------------------- 1.6

    