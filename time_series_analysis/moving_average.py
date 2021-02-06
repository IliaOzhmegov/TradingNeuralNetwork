import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/^GSPC.csv", index_col=0, parse_dates=True)
df.head(10)

# plot the adj. close price
plt.figure(figsize = (15,10))
df['Adj Close'].plot()
plt.ylabel('Adjusted Close Price')
plt.savefig("plots_lena/Adjusted_Closing_price")
plt.show()

#  Compute a short moving average of 253 days (MA253) - the number of trading days in a year
#  and a long moving average of 1265 days (MA1265) (5 years) on the Adj. Close Price
# these can be adjusted depending on desired granularity
MAS = 253 # short MA 1 year
MAL = 1265 # long MA 5 years

df['MA Short'] = df['Adj Close'].rolling(MAS).mean()
df['MA Long'] = df['Adj Close'].rolling(MAL).mean()
df.dropna(inplace=True)
df.head()

# plot the Adj.Close price and both moving averages
plt.figure(figsize= (15,10))
df['Adj Close'].plot(label = "Adjusted Close Price")
df['MA Short'].plot(color = 'red', label = "MA 1 year")
df['MA Long'].plot(color = 'green', label = "MA 5 years")
plt.legend()
plt.savefig("plots_lena/Moving_Average_all_time")
plt.show()


# It also makes sense to use a subset of the data.
# for example the last 10 years, beginning from 2010-Jan-02.
dt_range = pd.date_range(start="2010-01-02", end="2021-01-22")
df_short = df[df.index.isin(dt_range)]
df_short.head()

# Now we can "zoom in" and see moving averages for shorter time periods.
MAS = 44 # short MA 44 days (2 month - 22 trading days on average in a month)
MAL = 253 # long MA 1 year
df_short['MA Short'] = df_short['Adj Close'].rolling(MAS).mean()
df_short['MA Long'] = df_short['Adj Close'].rolling(MAL).mean()

# plot the Adj.Close price and both moving averages for the last 10 years+
plt.figure(figsize= (15,10))
df_short['Adj Close'].plot(label = "Adjusted Close Price")
df_short['MA Short'].plot(color = 'red', label = "MA 2 month")
df_short['MA Long'].plot(color = 'green', label = "MA 1 year")
plt.legend()
plt.savefig("plots_lena/Moving_Average_10_years")
plt.show()
