# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Change the working directory
print(os.getcwd())
os.chdir('/home/skystone/Documents/internship/')


# --------------------------  1.1

dataset = pd.read_csv('M&M.csv')
X = dataset.iloc[:, 1].values

# Delete all the rows int series column which does not contain "EQ"
delete = 0
for category in X:
    if category != 'EQ':
        dataset = dataset.drop(delete)        
    delete = delete + 1

dataset.head()
dataset.tail()
dataset.describe()

# ---------------------------- 1.2

dataset.iloc[-90:, 8].values.min()    # 8th column Close Price
dataset.iloc[-90:, 8].values.max()
dataset.iloc[-90:, 8].values.mean()

# --------------------------- 1.3

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Date'].dtype

dataset['Date'].max() - dataset['Date'].min()

# --------------------------- 1.4

dataset['Month'] = pd.DatetimeIndex(dataset['Date']).month
dataset['Year'] = pd.DatetimeIndex(dataset['Date']).year

dataset['Total Traded Quantity'].fillna(0)
Nr = (dataset['Close Price'] * dataset['Total Traded Quantity']).cumsum() 
Dr = dataset['Total Traded Quantity'].cumsum()
dataset['VWAP'] = Nr / Dr

ymv_dataset = dataset[['Month','Year','VWAP']]
groupedObj = ymv_dataset.groupby(['Month','Year'])
groupedObj.first()



# ---------------------------- 1.5


def calcAvg(N):
    avg = dataset.iloc[-N:,].mean()
    return avg

def profitLossPercent(N):
    check = (dataset['Close Price'].tail(N).iloc[N-1] - dataset['Close Price'].tail(N).iloc[0])
    if check < 0 :
        loss = -(check)
        lossPer = (loss/dataset['Close Price'].tail(N).iloc[N-1])*100
        print(lossPer)
        return lossPer
    if check > 0 :
        profit = check
        profitPer = (profit/dataset['Close Price'].tail(N).iloc[N-1])*100
        print(profitPer)
        return profitPer

print("Avg of last 1 week, 2 weeks, 1 month, 3 months, 6 months and 1 year")
calcAvg(7)
calcAvg(14)
calcAvg(30)
calcAvg(90)
calcAvg(180)
calcAvg(365)
print("Profit or loss of 1 week, 2 weeks, 1 month, 3 months, 6 months and 1 year")
profitLossPercent(7)
profitLossPercent(14)
profitLossPercent(30)
profitLossPercent(90)
profitLossPercent(180)
profitLossPercent(365)
    


# ---------------------------- 1.6

dataset['Close Price'].fillna(0)
dataset['Day_Perc_Change'] = dataset['Close Price'].pct_change()
dataset


# ---------------------------- 1.7

dataset['Trend'] = "nan"


if ((dataset['Day_Perc_Change'] >= -0.5) & (dataset['Day_Perc_Change'] <= 0.5)).all():
    dataset['Trend'] = 'Slight or No change'
if ((dataset['Day_Perc_Change'] >= 0.5) & (dataset['Day_Perc_Change'] <= 1)).all():
    dataset['Trend'] = 'Slight positive'
if ((dataset['Day_Perc_Change'] <= -0.5) & (dataset['Day_Perc_Change'] >= -1)).all():
    dataset['Trend'] = 'Slight negative'
if ((dataset['Day_Perc_Change'] >= 1) & (dataset['Day_Perc_Change'] <= 3)).all():
    dataset['Trend'] = 'Positive' 
if ((dataset['Day_Perc_Change'] <= -1) & (dataset['Day_Perc_Change'] >= -3)).all():
    dataset['Trend'] = 'Negative'
if ((dataset['Day_Perc_Change'] >= 3) & (dataset['Day_Perc_Change'] <= 7)).all():
    dataset['Trend'] = 'Among top gainers'
if ((dataset['Day_Perc_Change'] <= -3) & (dataset['Day_Perc_Change'] >= -7)).all():
    dataset['Trend'] = 'Among top losers'
if (dataset['Day_Perc_Change'] > 7).all():
    dataset['Trend'] = 'Bull run' 
if (dataset['Day_Perc_Change'] < -7).all():
    dataset['Trend'] = 'Bear drop' 

dataset


# ----------------------------- 1.8




# ----------------------------- 1.9

dataset.to_csv('Module1Solutions.csv')












