from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/^GSPC.csv", index_col=0, parse_dates=True)
df.head(10)

# Making Time Series Stationary
# Log Scale Transformation
X = df["Adj Close"]
ts_log = np.log(X)
# plot log transformed time series
plt.plot(ts_log)
plt.savefig("plots_lena/Log Transformed Adjusted Close Price")
plt.show()

# Seasonal Decomposition
decomposition = seasonal_decompose(ts_log, freq=50)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("plots_lena/Seasonal Decomposition")
plt.show()

# there is strange trend in seasonality.
# It might be because of non-stationarity of the data plus a very long time span.

# Further Techniques to remove Seasonality and Trend
# Differncing - difference of the observation at a particular instant with that at the previous instant.
ts_log_diff = ts_log.T - ts_log.T.shift()
plt.plot(ts_log_diff)
plt.savefig("plots_lena/Log Transformed Residuals")
plt.show()

### Auto Regression and Moving Average (ARMA) model ###

# train / test split
# test - predictions of last 5 years of the Adj. Close price

# ts_data = ts_log # log transformed data
ts_data = ts_log_diff # log transformed residuals
test_length = 1265
train, test = ts_data[1:len(ts_data)-test_length], ts_data[len(ts_data)-test_length:]

# fit ARMA model
# AR order 2, MA order 0 for log transformed data
# AR order 4, MA order 2 for log residuals
arma = ARMA(train, order=(4, 2)).fit()
predictions = arma.predict(start=len(train), end=len(train)+len(test)-1)
predictions.index = test.index

# plot AR model fitted values
plt.figure(figsize = (15,10))
plt.plot(ts_log_diff, label = "Log transformed Residuals")
plt.plot(arma.fittedvalues, color='red', label = "ARMA model fitted Residuals")
plt.title("Auto Regression and Moving Average model ARMA(4,2) - log Fitted Residuals")
plt.legend()
plt.savefig("plots_lena/ARMA log Fitted Residuals")
plt.show()

# plot AR model prediction results
plt.figure(figsize = (15,10))
plt.plot(test, label = "Log transformed Test Residuals")
plt.plot(predictions, color='red', label = "ARMA model predictions")
plt.title("Auto Regression and Moving Average model ARMA(4,2) - log Residuals Predictions")
plt.legend()
plt.savefig("plots_lena/ARMA Residuals Predictions")
plt.show()