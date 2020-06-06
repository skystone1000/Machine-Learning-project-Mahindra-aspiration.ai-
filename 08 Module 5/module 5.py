# --------------------------------------------- 5.1

import pandas as pd
import matplotlib.pyplot as plt

itc_stock_data = pd.read_csv('itc_stock_data.csv')
itc_stock_data


import numpy as np
itc_stock_data['Daily Return'] = (itc_stock_data['Close Price']).pct_change() 
itc_stock_data['Daily Return'] = itc_stock_data['Daily Return'].replace([np.inf, -np.inf], np.nan)
itc_stock_data = itc_stock_data.dropna()
print("Mean Daily Return")
itc_stock_data['Daily Return'].mean()



itc_stock_data['Daily Standard Deviation'] = (itc_stock_data['Close Price']).pct_change() 
itc_stock_data['Daily Standard Deviation'] = itc_stock_data['Daily Standard Deviation'].replace([np.inf, -np.inf], np.nan)
itc_stock_data = itc_stock_data.dropna()
print("Daily Standard Deviation")
itc_stock_data['Daily Standard Deviation'].std()


annual_mean =  -0.0002877035862325215 * 252
print("Annual Mean: "+ str(annual_mean))


import math
annual_stdev = 0.01348201288694839 * math.sqrt(252)
print("Annual Standard Deviation: "+ str(annual_stdev))


# --------------------------------------------- 5.2

fortis_data = pd.read_csv('fortis_stock_data.csv', sep='\s*,\s*',header=0, encoding='ascii', engine='python')
fortis_data

ongc_data = pd.read_csv('ongc_stock_data.csv', sep='\s*,\s*',header=0, encoding='ascii', engine='python')
ongc_data

wipro_data = pd.read_csv('wipro_stock_data.csv')
wipro_data

itc_data = pd.read_csv('itc_stock_data.csv')
itc_data

airtel_data = pd.read_csv('airtel_stock_data.csv')
airtel_data

data = pd.DataFrame(airtel_data['Date'])
data['Fortis'] = pd.DataFrame(fortis_data['Close Price'])
data['Ongc'] = pd.DataFrame(ongc_data['Close Price'])
data['Wipro'] = pd.DataFrame(wipro_data['Close Price'])
data['Itc'] = pd.DataFrame(itc_data['Close Price'])
data['Airtel'] = pd.DataFrame(airtel_data['Close Price'])
print("Closing Prices of the 5 respective stocks")
data = data.drop(['Date'], axis = 1) 
data.dropna()


# ---------------------------------------------- 5.3

import numpy as np
returns = data.pct_change()
mean_daily_returns = returns.mean()
mean_daily_returns = mean_daily_returns.values.reshape(5,1)
cov_matrix = returns.cov()
weights = np.asarray([0.2,0.2,0.2,0.2,0.2]) #weights of repective stocks
portfolio_return = round(np.sum(mean_daily_returns * weights) * 252,2)
portfolio_std_dev = round(np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252),2)
print('Portfolio expected annualised return is {} and volatility is {}'.format(portfolio_return,portfolio_std_dev))



#Monte Carlo Simulation

returns = data.pct_change()
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

num_portfolios = 25000

#set up array to hold results
results = np.zeros((3,num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(5)
    #rebalance weights to sum to 1
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    

    #store results in results array
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2,i] = results[0,i] / results[1,i]    
#convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe'])

results_frame
#create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.colorbar()


#--------------------------------------------------- 5.4

stocks = ['Fortis','Ongc','Wipro','Itc','Airtel']

returns = data.pct_change()
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

num_portfolios = 25000

#array for results
#increased the size of the array for holding the weight values for each stock
results = np.zeros((4+len(stocks)-1,num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(5)
    #rebalance weights to sum to 1
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    

    #store results in results array
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
    results[2,i] = results[0,i] / results[1,i]  
     #iterate through the weight vector and add data to results array
    for j in range(len(weights)):
        results[j+3,i] = weights[j]
#convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',stocks[0],stocks[1],stocks[2],stocks[3],stocks[4]])

#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
#locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]


#create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=1000)
#plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=1000)