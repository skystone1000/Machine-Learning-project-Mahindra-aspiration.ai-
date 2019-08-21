import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir('/home/skystone/Documents/internship')

# ------------------------------------ 2.1
"""
ref
    Python Pandas - Date Column to Column index
    https://stackoverflow.com/questions/15752422/python-pandas-date-column-to-column-index
    set_index
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html
    matplotlib
    https://towardsdatascience.com/matplotlib-tutorial-learn-basics-of-pythons-powerful-plotting-library-b5d1b8f67596
    Scrap the data
    https://towardsdatascience.com/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460
	
	Beautiful soup scraping
	https://www.dataquest.io/blog/web-scraping-tutorial-python/

"""


dataset = pd.read_csv('Module1Solutions.csv')
del dataset['Unnamed: 0']
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Date'].dtype

dataset.set_index('Date')

# Ploting Date Vs Close Price
plt.figure(figsize=(10,10))
plt.plot(dataset['Date'],dataset['Close Price'])
plt.title("General Outlook")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()
# From graph we can see that there is drastic changes at 2 points
# and when we look at data from csv those dates are
# 21-12-2017 drops below half in one day
# 21-09-2018 starts decreasing for 10 days


from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests 

url_21_12_2017 = 'https://www.moneycontrol.com/news/business/markets/d-street-buzz-psu-bank-auto-stocks-slip-led-by-bob-m-zee-ent-spikes-7-indiabulls-housing-tanks-4302961.html'

response = requests.get(url_21_12_2017) 
soup = BeautifulSoup(response.content,"html.parser")

#list(soup.children)
[type(item) for item in list(soup.children)]
html = list(soup.children)[2]



stock_news_title_container = soup.find(class_ = 'articleData clr')
stock_news_subhead_container = soup.find('div',class_ = 'brk_wraper clearfix')
stock_news_body_container = soup.find('article',class_ = 'Normal')

print("Money Control News")
print("News for 21-12-2017")
print('\n')
print("Title:")
print (soup.find(class_ = "artTitle").get_text())
print('\n')
print("Subject:")
print (soup.find(class_ = "subhead").get_text())
print('\n')
print("Report:")
print(soup.select("div p"))

# ------------------------------------------------------2.2
import matplotlib.pyplot as plt
plt.stem(dataset['Date'],dataset['Day_Perc_Change'])

# ---------------------------------------------------------2.3
plt.stem(dataset['Date'],dataset['Total Traded Quantity'])

plt.stem(dataset['Date'],dataset['Day_Perc_Change'])

plt.stem(dataset['Day_Perc_Change'],dataset['Total Traded Quantity'])

plt.stem(dataset['Total Traded Quantity'],dataset['Day_Perc_Change'])

plt.plot(dataset['Day_Perc_Change'],dataset['Total Traded Quantity'])

plt.plot(dataset['Total Traded Quantity'],dataset['Day_Perc_Change'])

# -------------------------------------------------------2.4

import matplotlib.pyplot as plt 
from collections import Counter

Trendsare = ['Postive','Negative','Breakout Bull','Breakout Bear','Among top losers','Among top gainers','Slight or No Change','Slight Positive','Slight Negative']
Trends_to_list = dataset['Trend'].tolist()
counts = Counter(Trends_to_list)
counts

# So we only have 1 trend so there will be only 1 color pie chart 
counter = [494]
labels = ['Slight or No change']
colors = ['r']
plt.pie(counter,labels=labels,colors=colors,startangle=90)
plt.show()

# we will find avg of each trend type
# but here we have only onr trend ie slight or no change
bg = dataset.groupby(['Trend'])['Total Traded Quantity']
bg.describe()

dataset.groupby(['Trend'])['Total Traded Quantity'].mean().plot.bar()

dataset.groupby(['Trend'])['Total Traded Quantity'].median().plot.bar()

#---------------------------------------------------------2.5

plt.hist(dataset['Day_Perc_Change'])
plt.show()

#-------------------------------------------------------2.6
import pandas as pd
airtel_stock = pd.read_csv('airtel_stock_data.csv')
cub_stock = pd.read_csv('cub_stock_data.csv')
itc_stock = pd.read_csv('itc_stock_data.csv')
tcs_stock = pd.read_csv('tcs_stock_data.csv')
wipro_stock = pd.read_csv('wipro_stock_data.csv')

airtel_stock_filt = airtel_stock[airtel_stock.Series == 'EQ']
cub_stock_filt = cub_stock[cub_stock.Series == 'EQ']
itc_stock_filt = itc_stock[itc_stock.Series == 'EQ']
tcs_stock_filt = tcs_stock[tcs_stock.Series == 'EQ']
wipro_stock_filt = wipro_stock[wipro_stock.Series == 'EQ']

dataFrame_ClosePrice = pd.DataFrame(columns = ['airtel','cub','itc','tcs','wipro'])
dataFrame_ClosePrice['airtel'] = airtel_stock_filt['Close Price']
dataFrame_ClosePrice['cub'] = cub_stock_filt['Close Price']
dataFrame_ClosePrice['itc'] = itc_stock_filt['Close Price']
dataFrame_ClosePrice['tcs'] = tcs_stock_filt['Close Price']
dataFrame_ClosePrice['wipro'] = wipro_stock_filt['Close Price']
dataFrame_ClosePrice.dropna()

#   pandas.Series.pct_change
#   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.pct_change.html
dataFrame_pctChange = dataFrame_ClosePrice.pct_change()
dataFrame_pctChange.dropna()

import seaborn as sns
sns.set(color_codes = True)
sns.pairplot(dataFrame_pctChange)

# ------------------------------------------------------ 2.7
import matplotlib.pyplot as plt

# rolling avg of pct change
rolling_average_airtel = []
rolling_average_airtel = dataFrame_pctChange['airtel'].rolling(7).mean()

# standard deviation
standard_deviation_airtel = rolling_average_airtel.fillna(0).std()
standard_deviation_airtel

# plot the values
dateList = pd.to_datetime(airtel_stock_filt['Date']).tolist() 
rollingAvgList = rolling_average_airtel.fillna(0).tolist()
plt.plot( dateList ,  rollingAvgList )
plt.show()

# ------------------------------------------------------ 2.8
nifty_data = pd.read_csv('Nifty50.csv')

nifty_ClosePrice = nifty_data['Close']
nifty_pct_change = nifty_ClosePrice.pct_change().fillna(0).rolling(7).mean().fillna(0)
nifty_date = pd.to_datetime(nifty_data['Date'])
nifty_date_list = nifty_date.tolist()

cub_Date = pd.to_datetime(cub_stock_filt['Date'])
cub_Date_list = cub_Date.tolist()
cub_ClosePrice = cub_stock_filt['Close Price']
cub_pct_change = cub_ClosePrice.pct_change().fillna(0).rolling(7).mean().fillna(0)
plt.figure(figsize=(20,10))
plt.plot(cub_Date_list,rolling_average_airtel.fillna(0).tolist())

plt.title("Volatility of NIFTY with respect to airtel and cub")
plt.plot(nifty_date,nifty_pct_change.tolist(),label = 'nifty')
plt.plot(dateList,rolling_average_airtel.fillna(0).tolist(),label = 'airtel')
plt.plot(cub_Date_list,cub_pct_change,label = 'cub')
plt.legend(loc='upper left')
plt.show()


# ------------------------------------------------------- 2.9


# we will use cub stocks for applying buy/sell signals
plt.figure(figsize=(20,10))
plt.plot(cub_Date_list,cub_pct_change,label = 'cub')
plt.legend(loc='upper left')
plt.show()

#making short and long signals
short_window = 21
long_window = 34

signals = pd.DataFrame(index=cub_stock_filt.index)
signals['signal'] = 0.0

#SMA of Short Window
signals['short_mavg'] = cub_stock_filt['Close Price'].rolling(window=short_window, min_periods=1,center=False).mean()

#SMA of Long Window
signals['long_mavg'] = cub_stock_filt['Close Price'].rolling(window=long_window,min_periods=1, center=False).mean()

#Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0,0.0)

#Generate trading orders
signals['positions'] = signals['signal'].diff()
print(signals)

# Initialize the plot figure
fig = plt.figure(figsize=(20,15))

#Add a subplot and label for y-axis
ax1 = fig.add_subplot(111, ylabel='Price')

#Plot the closing price
cub_stock_filt['Close Price'].plot(ax=ax1, color='black', lw=2.)

#plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

#Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^' , markersize=20,color='g')

#Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v' , markersize=20,color='r')

plt.show()



# -----------------------------------------------------------------2.10

import pandas as pd
import matplotlib.pyplot as plt

symbol = 'CUB'

# read csv file, use date as index and read close as a column
df = pd.read_csv('cub_stock_data.csv'.format(symbol), index_col='Date',
                 parse_dates=True, usecols=['Date', 'Close Price'],
                 na_values='nan')

# rename the column header with symbol name
df = df.rename(columns={'Close Price': symbol})
df.dropna(inplace=True)

# calculate Simple Moving Average with 14 days window
sma = df.rolling(window=14).mean()

# calculate the standar deviation
rstd = df.rolling(window=14).std()

upper_band = sma + 2 * rstd
upper_band = upper_band.rename(columns={symbol: 'upper'})
lower_band = sma - 2 * rstd
lower_band = lower_band.rename(columns={symbol: 'lower'})
df = df.join(upper_band).join(lower_band)
ax = df.plot(title='{} Price and BB'.format(symbol))
ax.fill_between(df.index, lower_band['lower'], upper_band['upper'], color='#ADCCFF', alpha='0.4')
ax.set_xlabel('Date')
ax.set_ylabel('SMA and BB')
ax.grid()
plt.show()






