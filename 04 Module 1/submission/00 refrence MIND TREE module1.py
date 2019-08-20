
###################################
"""
Query 1.1
Import the csv file of the stock of your choosing using 'pd.read_csv()' function into a dataframe. Shares of a company can be offered in more than one category. The category of a stock is indicated in the ‘Series’ column. If the csv file has data on more than one category, the ‘Date’ column will have repeating values. To avoid repetitions in the date, remove all the rows where 'Series' column is NOT 'EQ'. Analyze and understand each column properly. You'd find the head(), tail() and describe() functions to be immensely useful for exploration. You're free to carry out any other exploration of your own.
"""

import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import datetime
from functools import partial


import os

# Change the working directory
print(os.getcwd())
os.chdir('/home/skystone/Documents/internship/')


data = pd.read_csv("M&M.csv")

data

filtered_data = data[data.Series == 'EQ']
filtered_data

filtered_data.head()

filtered_data.tail()

filtered_data.describe()


###################################
"""
Query 1.2
Calculate the maximum, minimum and mean price for the last 90 days. (price=Closing Price unless stated otherwise)
"""

filtered_data.tail(90)['Close Price'].max()

filtered_data.tail(90)['Close Price'].min()

filtered_data.tail(90)['Close Price'].mean()


###################################
"""
Query 1.3
Analyse the data types for each column of the dataframe. Pandas knows how to deal with dates in an intelligent manner. But to make use of Pandas functionality for dates, you need to ensure that the column is of type 'datetime64(ns)'. Change the date column from 'object' type to 'datetime64(ns)' for future convenience. See what happens if you subtract the minimum value of the date column from the maximum value.
"""

filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
filtered_data['Date'].dtype
#datetime64[ns] maps to either <M8[ns] or >M8[ns] depending on the endian-ness of the machine

filtered_data['Date'].max()-filtered_data['Date'].min()



##################################
"""
Query 1.4
In a separate array , calculate the monthwise VWAP (Volume Weighted Average Price ) of the stock. ( VWAP = sum(price*volume)/sum(volume) ) To know more about VWAP , visit - VWAP definition {Hint : Create a new dataframe column ‘Month’. The values for this column can be derived from the ‘Date” column by using appropriate pandas functions. Similarly, create a column ‘Year’ and initialize it. Then use the 'groupby()' function by month and year. Finally, calculate the vwap value for each month (i.e. for each group created).
"""

data['Month'] = pd.DatetimeIndex(data['Date']).month
data['Year'] = pd.DatetimeIndex(data['Date']).year
data['VWAP'] = (data['Close Price'] * data['Total Traded Quantity']).cumsum() / data['Total Traded Quantity'].fillna(0).cumsum()
data_vwap = data[['Month','Year','VWAP']]
group = data_vwap.groupby(['Month','Year'])
group.first()



##################################
"""
Query 1.5
Write a function to calculate the average price over the last N days of the stock price data where N is a user defined parameter. Write a second function to calculate the profit/loss percentage over the last N days. Calculate the average price AND the profit/loss percentages over the course of last - 1 week, 2 weeks, 1 month, 3 months, 6 months and 1 year. {Note : Profit/Loss percentage between N days is the percentage change between the closing prices of the 2 days }
"""

#here N refers to number of days
def avg_price(N):
    return (data['Average Price'].tail(N).sum())/N
print("Average prices for last N days are as follows:")
print("Last 1 week",avg_price(5))
print("Last 2 weeks",avg_price(10))
print("Last 1 month",avg_price(20))
print("Last 3 months",avg_price(60))
print("Last 6 months",avg_price(120))
print("Last 1 year",avg_price(240))
print("Profit/Loss % for N days are as follows:")
def prof_loss(N):
    difference = (data['Close Price'].tail(N).iloc[N-1] - data['Close Price'].tail(N).iloc[0])
    if difference < 0 :
        loss = -(difference)
        loss_percen = (loss/data['Close Price'].tail(N).iloc[N-1])*100
        return loss_percen
    if difference > 0 :
        profit = difference
        profit_percen = (profit/data['Close Price'].tail(N).iloc[N-1])*100
        return profit_percen
print("Loss/Profit percentage for last N days are as follows:")
print("Last 1 week",prof_loss(5))
print("Last 2 weeks",prof_loss(10))
print("Last 1 month",prof_loss(20))
print("Last 3 months",prof_loss(60))
print("Last 6 months",prof_loss(120))
print("Last 1 year",prof_loss(240))



#######################################
"""
Query 1.6
Add a column 'Day_Perc_Change' where the values are the daily change in percentages i.e. the percentage change between 2 consecutive day's closing prices. Instead of using the basic mathematical formula for computing the same, use 'pct_change()' function provided by Pandas for dataframes. You will note that the first entry of the column will have a ‘Nan’ value. Why does this happen? Either remove the first row, or set the entry to 0 before proceeding.
"""

data['Day_Perc_Change'] = data['Close Price'].pct_change().fillna(0)
data

#######################################
"""
Query 1.7¶
Add another column 'Trend' whose values are: 'Slight or No change' for 'Day_Perc_Change' in between -0.5 and 0.5 'Slight positive' for 'Day_Perc_Change' in between 0.5 and 1 'Slight negative' for 'Day_Perc_Change' in between -0.5 and -1 'Positive' for 'Day_Perc_Change' in between 1 and 3 'Negative' for 'Day_Perc_Change' in between -1 and -3 'Among top gainers' for 'Day_Perc_Change' in between 3 and 7 'Among top losers' for 'Day_Perc_Change' in between -3 and -7 'Bull run' for 'Day_Perc_Change' >7 'Bear drop' for 'Day_Perc_Change' <-7
"""

if ((data['Day_Perc_Change'] >= -0.5).all() and (data['Day_Perc_Change'] < 0.5).all()):
    data['Trend'] = 'Slight or No change'
if ((data['Day_Perc_Change'] >= 0.5) & (data['Day_Perc_Change'] < 1)).all():
    data['Trend'] = 'Slight positive'
if ((data['Day_Perc_Change'] <= -0.5) & (data['Day_Perc_Change'] > -1)).all():
    data['Trend'] = 'Slight negative'
if ((data['Day_Perc_Change'] >= 1) & (data['Day_Perc_Change'] < 3)).all():
    data['Trend'] = 'Positive' 
if ((data['Day_Perc_Change'] <= -1) & (data['Day_Perc_Change'] > -3)).all():
    data['Trend'] = 'Negative'
if ((data['Day_Perc_Change'] >= 3) & (data['Day_Perc_Change'] < 7)).all():
    data['Trend'] = 'Among top gainers'
if ((data['Day_Perc_Change'] <= -3) & (data['Day_Perc_Change'] > -7)).all():
    data['Trend'] = 'Among top losers'
if (data['Day_Perc_Change'] > 7).all():
    data['Trend'] = 'Bull run' 
if (data['Day_Perc_Change'] < -7).all():
    data['Trend'] = 'Bear drop' 
data
temp = data['Day_Perc_Change']



########################################
"""
Query 1.8
Find the average and median values of the column 'Total Traded Quantity' for each of the types of 'Trend'. {Hint : use 'groupby()' on the 'Trend' column and then calculate the average and median values of the column 'Total Traded Quantity'}
"""

data.groupby(data.Trend).mean()['Total Traded Quantity']
data.groupby(data.Trend).median()['Total Traded Quantity']




########################################
"""
Query 1.9
SAVE the dataframe with the additional columns computed as a csv file week2.csv. In Module 2, you are going to get familiar with matplotlib, the python module which is used to visualize data.
"""

data.to_csv('Module1Solutions.csv')







