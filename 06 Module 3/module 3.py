# ------------------------------------ Query 3.1

import pandas as pd

goldData = pd.read_csv('GOLD.csv')
goldData.set_index('Date',inplace=True)
goldWithoutNan = goldData.dropna()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = np.array(goldWithoutNan["Pred"])
x = np.array(goldWithoutNan["new"])
x = x.reshape(-1,1)
y = y.reshape(-1,1)

regression_model = LinearRegression()

regression_model.fit(x, y)

y_predicted = regression_model.predict(x)

rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)

print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('RMS error: ', rmse)
print('R2 score: ', r2)

plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y_predicted,color='g')
plt.show()




rawData = goldData[:]
rawDataNew = rawData['new']
rawDataNew = rawDataNew.values.reshape(-1,1)
naData = (regression_model.predict(rawDataNew))
naDataSeries = pd.Series(naData.ravel())
sata = naDataSeries.to_frame()
goldData['Pred'] = sata
goldData



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = np.array(goldData["new"])
x = np.array(goldData["Pred"])
x = x.reshape(-1,1)
y = y.reshape(-1,1)

regression_model = LinearRegression()
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)

rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)

print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('RMS error: ', rmse)
print('R2 score: ', r2)

plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y_predicted)
plt.show()



import matplotlib.pyplot as plt
plt.hist(goldData['Pred'])
plt.show()



import seaborn as sns
sns.distplot(goldData['Pred'])
plt.show()


# ------------------------------------ Query 3.2

tcs_data = pd.read_csv('tcs_stock_data.csv')
tcs_data['Date'] = pd.to_datetime(tcs_data['Date'])
tcs_data = tcs_data.sort_values('Date')
tcs_data.set_index('Date', inplace=True)
tcs_data


nifty_data = pd.read_csv('NIFTY50_Data.csv')
nifty_data['Date'] = pd.to_datetime(nifty_data['Date'])
nifty_data = nifty_data.sort_values('Date')
nifty_data.set_index('Date', inplace=True)
nifty_data


fil_tcs = tcs_data[405:]
fil_nifty = nifty_data[405:]
return_tcs = fil_tcs['Close Price'].pct_change()
return_nifty = fil_nifty['Close'].pct_change()

plt.figure(figsize=(20,10))
return_tcs.plot()
return_nifty.plot()
plt.ylabel("Daily Return of TCS and NIFTY")
plt.show()


fil_tcs['pct_change'] = fil_tcs['Close Price'].pct_change()
fil_nifty['pct_change'] = fil_nifty['Close'].pct_change()



x = fil_tcs['pct_change'].dropna()
y = fil_nifty['pct_change'].dropna()
import pandas as pd 
import statsmodels.api as sm
myModel = sm.OLS(y,x).fit()
myModel.summary()



import pandas as pd
import statsmodels.api as sm

'''
Download monthly prices of TCS and NIFTY 50 for Time period: 1-Jan-2014--12-Jan-2017
'''
tcs = pd.read_csv('TCS.NS.csv', parse_dates=True, index_col='Date',)
nifty50 = pd.read_csv('^NSEI.csv', parse_dates=True, index_col='Date')

# joining the closing prices of the two datasets 
monthly_prices = pd.concat([tcs['Close'], nifty50['Close']], axis=1)
monthly_prices.columns = ['TCS', 'NIFTY50']

# check the head of the dataframe
print(monthly_prices.head())

# calculate monthly returns
monthly_returns = monthly_prices.pct_change(1)
clean_monthly_returns = monthly_returns.dropna(axis=0)  # drop first missing row
print(clean_monthly_returns.head())




# split dependent and independent variable
X = clean_monthly_returns['TCS']
y = clean_monthly_returns['NIFTY50']

# Add a constant to the independent value
X1 = sm.add_constant(X)

# make regression model 
model = sm.OLS(y, X1)

# fit model and print results
results = model.fit()
print(results.summary())



# Daily beta value for the past 3 motnhs for the stock
#TCS is 0.1968 which is less than 1 and hence it is
# less volatile than the benchmark




# The monthly beta value for the stock TCS is 0.1327
# which is less than 1 and hence it is less volatile
# than the benchmark



