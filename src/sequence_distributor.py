import pandas as pd
import mplfinance as fplt
from libs.common import Scaler

df = pd.read_csv("data/SP500.csv", index_col=0, parse_dates=True)


i = 0

n_predictors = 30
n_to_predict = 5

# window DATA frame
wdf = df.iloc[i:i+n_predictors,:]
max = wdf.High.max()
min = wdf.Low.min()

max
min

scaler = Scaler(min=min, max=max)
scaler.get_marigins()

wdf.apply(scaler.scale, axis=1)
scaler.scale(wdf.High)



wdf = df.iloc[i:i+n_predictors + n_to_predict,:]
fplt.plot(wdf,
          type='candle',
          title='S&P 500',
          ylabel='Price ($)',
          style='yahoo')

fplt.plot(wdf.apply(scaler.scale, axis=1),
          type='candle',
          title='S&P 500',
          ylabel='Price ($)',
          style='yahoo')
