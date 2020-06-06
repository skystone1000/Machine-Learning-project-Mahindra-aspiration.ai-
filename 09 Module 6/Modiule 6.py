#---------------------------------------- 6.1

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans,vq
import pandas as pd
import pandas_datareader as dr
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

gr = pd.DataFrame()
manapuram = pd.read_csv('30 stocks/smallcaps/manapuram_stock_data.csv')
gr['Manapuram'] = manapuram['Close Price']
program = pd.read_csv('30 stocks/smallcaps/progam_stock_data.csv')
gr['Program'] = program['Close Price']
ramcos = pd.read_csv('30 stocks/smallcaps/ramcos_stock_data.csv')
gr['Ramcos'] = ramcos['Close Price']
sunteck = pd.read_csv('30 stocks/smallcaps/sunteck_stock_data.csv')
gr['Sunteck'] = sunteck['Close Price']
time_techno = pd.read_csv('30 stocks/smallcaps/time_techno_stock_data.csv')
gr['Time_Techno'] = time_techno['Close Price']
welcorp = pd.read_csv('30 stocks/smallcaps/welcorp_stock_data.csv')
gr['Welcorp'] = welcorp['Close Price']
yes_bank = pd.read_csv('30 stocks/smallcaps/yes_bank_stock_data.csv')
gr['Yes_Bank'] = yes_bank['Close Price']
airtel = pd.read_csv('30 stocks/largecaps/bharti_airtel_stock_data.csv')
gr['Airtel'] = airtel['Close Price']
bhel = pd.read_csv('30 stocks/largecaps/bhel_stock_data.csv')
gr['Bhel'] = bhel['Close Price']
cadila = pd.read_csv('30 stocks/largecaps/cadila_stock_data.csv')
gr['Cadila'] = cadila['Close Price']
dlf = pd.read_csv('30 stocks/largecaps/dlf_stock_data.csv')
gr['DLF'] = dlf['Close Price']
hdfc = pd.read_csv('30 stocks/largecaps/hdfc_stock_data.csv')
gr['Hdfc'] = hdfc['Close Price']
hdnzinc = pd.read_csv('30 stocks/largecaps/hdnzinc_stock_data.csv')
gr['hdnzinc'] = hdnzinc['Close Price']
IDEA = pd.read_csv('30 stocks/largecaps/IDEA_stock_data.csv')
gr['IDEA'] = IDEA['Close Price']
msm = pd.read_csv('30 stocks/largecaps/msm_stock_data.csv')
gr['Msm'] = msm['Close Price']
piramal = pd.read_csv('30 stocks/largecaps/piramal_stock_data.csv')
gr['Piramal'] = piramal['Close Price']
ultracemo = pd.read_csv('30 stocks/largecaps/ultracemo_stock_data.csv')
gr['ultracemo'] = ultracemo['Close Price']
boi = pd.read_csv('30 stocks/midcaps/boi_stock_data.csv')
gr['Boi'] = boi['Close Price']
dishtv = pd.read_csv('30 stocks/midcaps/dishtv_stock_data.csv')
gr['Dishtv'] = dishtv['Close Price']
escorts = pd.read_csv('30 stocks/midcaps/escorts_stock_data.csv')
gr['Escorts'] = escorts['Close Price']
godrej = pd.read_csv('30 stocks/midcaps/godrej_stock_data.csv')
gr['godrej'] = godrej['Close Price']
gujgas = pd.read_csv('30 stocks/midcaps/gujgas_stock_data.csv')
gr['Gujgas'] = gujgas['Close Price']
pageinds = pd.read_csv('30 stocks/midcaps/pageinds_stock_data.csv')
gr['Pageinds'] = pageinds['Close Price']
supreme = pd.read_csv('30 stocks/midcaps/supreme_stock_data.csv')
gr['Supreme'] = supreme['Close Price']
tatacomm = pd.read_csv('30 stocks/midcaps/tatacomm_stock_data.csv')
gr['Tatacomm'] = tatacomm['Close Price']
tatglobal = pd.read_csv('30 stocks/midcaps/tatglobal_stock_data.csv')
gr['Tatglobal'] = tatglobal['Close Price']
ubl = pd.read_csv('30 stocks/midcaps/ubl_stock_data.csv')
gr['Ubl'] = ubl['Close Price']
cafin = pd.read_csv('30 stocks/smallcaps/canfin_stock_data.csv')
gr['cafin'] = cafin['Close Price']
capacite = pd.read_csv('30 stocks/smallcaps/capacite_stock_data.csv')
gr['Capacite'] = capacite['Close Price']
db_corp = pd.read_csv('30 stocks/smallcaps/db_corp_stock_data.csv')
gr['db_corp'] = db_corp['Close Price']
gr['Date'] = bhel['Date']
gr['Date'] = pd.to_datetime(gr['Date'])
gr.set_index('Date',inplace = True)



#---------------------------------------- 6.2

#Calculate average annual percentage return and volatilities over a theoretical one year period
gr = gr.dropna()
returns = gr.pct_change().mean() * 252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = gr.pct_change().std() * sqrt(252)
returns


# ------------------------------------ 6.3

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
X = data
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
    

# -------------------------------------- 6.4

# computing K-Means with K = 3 (3 clusters)
centroids,_ = kmeans(X,3)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sg',markersize=3)



#identify the outlier
print(returns.idxmax())



#drop the relevant stock from our data
returns.drop('Piramal',inplace=True)
#recreate data to feed into the algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T



# computing K-Means with K = 3 (3 clusters)
centroids,_ = kmeans(data,3
)# assign each sample to a cluster
idx,_ = vq(data,centroids)
# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()




centroids,_ = kmeans(X,3)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sg',markersize=3)



#identify the outlier
print(returns.idxmax())


#drop the relevant stock from our data
returns.drop('Manapuram',inplace=True)
#recreate data to feed into the algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T



# computing K-Means with K = 3 (3 clusters)
centroids,_ = kmeans(data,3)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()



details = [(name,cluster) for name, cluster in zip(returns.index,idx)]
for detail in details:
    print(detail)