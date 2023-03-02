# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:16:40 2023

@author: dogukan1
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

df = yf.download(tickers=['SISE.IS'], period='3y')
df2 = df['Close'].fillna(method='ffill')

# Plotting Şişecam Stock Prices for 3 years.
df2.plot(figsize=(15,6))
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Price of Şişecam Stock')



#Parameters for ARIMA 

#First of all, We are going to make time serie stationary. When we made data stationary with diff,
#We are going to build acf and pacf graphs to decide the values of P and Q parameters.

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df2)

# Autocorrelation graph after first diff.
f = plt.figure(figsize=(15,6))
ax1 = f.add_subplot(121)
ax1.set_title('1st Order Differencing')
ax1.plot(df2.diff())
ax2 = f.add_subplot(122)
plot_acf(df2.diff().dropna(), ax = ax2)
plt.show()
 
#Looks pretty well. To prove that looking well, let's check the p-value of ADF test.

from statsmodels.tsa.stattools import adfuller

result = adfuller(df2.dropna())    #İşlem görmemiş değerlerin durgunluğu
print('p-value :', result[1])      #H0 can not be rejected. Not stationary


result2 = adfuller(df2.diff().dropna())   #With diff observations our time serie is stationary.
print('p-value :', result2[1])            #H0 can be rejected.


# At this part, we have decided to accept d parameter as 1.
# After calculating d parameter, Our goal is being able to evaluate p parameter.


f2 = plt.figure(figsize=(15,6))
ax3 = f2.add_subplot(121)
ax3.set_title('1st Order Differencing')
ax3.plot(df2.diff())

ax4 = f2.add_subplot(122)
plot_pacf(df2.diff().dropna(), ax = ax4)
plt.show()

#As we can see from output of PACF graph, the first lag is  significantly out of the limit
#so that we are going to take p parameter as 1.

#The conditions are same at ACF graph as well. Therefore, we are going to take q parameter
#as 1, too.


#Building ARIMA Model

from statsmodels.tsa.arima_model import ARIMA

arima_model = ARIMA(df2, order = (1,1,1))

model = arima_model.fit()
model.summary()

plt.figure(figsize = (15,6))
model.plot_predict(dynamic = False)

#Predictions for 150 days ahead with confidence interval
predictions, error, conf_interval = model.forecast(150)
plt.figure(figsize = (15,6))
plt.plot(predictions)
plt.fill_between(range(len(predictions)), conf_interval[:,0], conf_interval[:,1], alpha = 0.3)



#####ANOTHER TYPE OF GRAPHIC FOR EVALUATING THE PERFORMANCE OF ARIMA MODEL############
train = df2[:544]
test = df2[544:]
model2 = ARIMA(train, order=(1, 1, 1))  
fitted = model2.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(208, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals (Şişecam)')
plt.legend(loc='upper left', fontsize=8)
plt.show()
#############################################################################



#The model that we built performed well. Now we are going to use part of our data to make a prediction.
#In order to check how our model can perform close prices that already happened.

y_pred = pd.Series(model.forecast(198)[0], index = df2[554:].index)
y_true = df2[554:]

#Plotting the results
plt.figure(figsize = (15,6))
plt.plot(y_pred, label='Prediction', color='green')
plt.plot(y_true, label='Actual', color='purple')
plt.legend(title='Group')
plt.ylabel('Price', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.title('Şişecam Stock Price', fontsize=16)
plt.show()

# Model is completely bad for Şişecam Stock. Some of Time series forecasting models are not useful for Turkish Stocks.
#Not only for Stocks of Şişecam Company, ARIMA models were tried to use on Turkish Stock Markets. All of results did seem bad.

#Some metrics for our model;

mape = np.mean(np.abs(y_pred - y_true)/np.abs(y_true)) #Mean Absolute Percentage Error
mae = np.mean(np.abs(y_pred - y_true)) #Mean Absolute Error
mpe = np.mean((y_pred - y_true)/y_true)
rmse = np.mean((y_pred - y_true)**2)**.5 #RMSE
corr = np.corrcoef(y_pred,y_true)[0,1] #Correlation Coefficient

import pprint

pprint.pprint({'Corr' :corr,'MAPE':mape, 'MAE':mae, 'MPE':mpe,
               'RMSE':rmse})


#Our metrics are not looking good for a stock, which is changed between 40 and 50 Turkish Liras last year.






