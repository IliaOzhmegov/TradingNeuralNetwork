import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
from mplfinance.original_flavor import candlestick_ohlc

# df = pd.read_csv("data/SP500.csv", index_col=0, parse_dates=True)
# df.head()
# extracting Data for plotting
df = pd.read_csv("data/SP500.csv")
df = df[['Date', 'Open', 'High', 'Low', 'Close']]

# convert into datetime object
df['Date'] = pd.to_datetime(df['Date'])

# apply map function
df['Date'] = df['Date'].map(mpdates.date2num)

i = 0
n_predictors = 30
wdf = df.iloc[i:i + n_predictors, :].copy()
t0 = wdf.Date[0]
wdf["Date"] = np.arange(t0, t0+n_predictors)


alpha = 0.9 # 0.5-1
width = 0.9 # 0.3-0.9

# creating Subplots
fig, ax = plt.subplots()

# plotting the data
candlestick_ohlc(ax, wdf.values, width=width,
                 colorup='green', colordown='red',
                 alpha=alpha)

# allow grid
ax.grid(True)
# plt.gca().set_axis_off()
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
#             hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
fig.tight_layout()

fig.set_size_inches(12, 2)
plt.savefig("example2.png", format='png', bbox_inches='tight', dpi=100)


# second part
