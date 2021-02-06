# Created by Lena
from statsmodels.tsa.ar_model import AR
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PATH = "plots/lena/"

df = pd.read_csv("data/^GSPC.csv", index_col=0, parse_dates=True)
df.head(10)

# checking for Autocorrelation
# Autocorrelation plots
autocorrelation_plot(df)
plt.savefig(PATH + "Autocorrelation plot")
plt.show()

# Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots
plot_acf(df["Adj Close"], lags=50) # lag 50 days
plt.savefig(PATH + "Autocorrelation Function plot")
plt.show()

plot_pacf(df["Adj Close"], lags=50)
plt.savefig(PATH + "Partial Autocorrelation Function plot")
plt.show()

# If the time series is stationary, the ACF/PACF plots will show a quick drop-off in correlation after a
# small amount of lag between points.
# This data is non-stationary as a high number of previous observations are correlated with future values.

### Autoregressive model ###

# train / test split
# test - predictions of last 5 years of the Adj. Close price
test_length = 1265
X = df["Adj Close"].values
train, test = X[1:len(X)-test_length], X[len(X)-test_length:]

# Fit AR model
model = AR(train)
model_fit = model.fit(maxlag=19, ic="bic")
# bic method determined best value for maxlag is 19

# make predictions for the last 5 years
train_predictions = model_fit.predict(start=19, end=len(train))
predictions = model_fit.predict(start=len(train)-1, end=len(train)+len(test)-1) # 22109 to 23374

# plot AR model train results
plt.figure(figsize = (15,10))
plt.plot(train, label = "Train Adjusted close price")
plt.plot(train_predictions, color='red', label = "AR(p) model predictions")
plt.title("Autoregressive model AR(p) - Train data")
plt.legend()
plt.savefig(PATH + "AR_Train")
plt.show()

# plot AR model prediction results
plt.figure(figsize = (15,10))
plt.plot(test, label = "Test Adjusted close price")
plt.plot(predictions, color='red', label = "AR(p) model predictions")
plt.title("Autoregressive model AR(p) - Test data")
plt.legend()
plt.savefig(PATH + "AR_Test")
plt.show()

# Train and test prediction results show that model is highly overfitted.
# The model can't generalise well and performs poorly on unseen test data - resulting in linear fit.


