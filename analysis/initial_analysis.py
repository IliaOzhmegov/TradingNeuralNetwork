# Created by Ilia
import pandas as pd
import mplfinance as fplt

df = pd.read_csv("data/^GSPC.csv", index_col=0, parse_dates=True)

path = "plots/initial_analysis/"

fplt.plot(df,
          type='candle',
          title='S&P 500',
          ylabel='Price ($)',
          style='yahoo')

fplt.plot(df,
          type='candle', style='yahoo',
          title='S&P500',
          ylabel='Price ($)',
          savefig=path + 'SP500_all.png')

dt_range = pd.date_range(start="1927-12-30", end="1962-01-03")
fplt.plot(df[df.index.isin(dt_range)],
          type='candle', style='yahoo',
          title='S&P500',
          ylabel='Price ($)',
          savefig=path + 'SP500_before_1962.png')

fplt.plot(df[df.index.isin(dt_range)],
          type='line', style='yahoo',
          title='S&P500',
          ylabel='Price ($)',
          savefig=path + 'SP500_before_1962_line.png')

# As you can see it does not make much sense to look at the candle
# sticks before 1962-Jan-02. Before that date data is no full.

dt_range = pd.date_range(start="1962-01-02", end="2021-01-22")
df = df[df.index.isin(dt_range)]

fplt.plot(df,
          type='candle',
          title='S&P 500',
          ylabel='Price ($)',
          style='yahoo',
          savefig =path + 'SP500_after_1962.png')

df.to_csv("data/SP500.csv", index=True)