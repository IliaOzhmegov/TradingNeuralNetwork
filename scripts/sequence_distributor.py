import pandas as pd
from libs.common import Scaler

df = pd.read_csv("data/SP500.csv", index_col=0, parse_dates=True)

n = df.shape[0]
n_predictors = 30
n_to_predict = 5


for i in range(n - n_predictors - n_to_predict):
    # window DATA frame
    wdf = df.iloc[i:i+n_predictors,:]
    max = wdf.High.max()
    min = wdf.Low.min()

    scaler = Scaler(min=min, max=max)

    wdf = df.iloc[i:i+n_predictors + n_to_predict,:]
    wdf.apply(scaler.scale, axis=1).to_csv("data/sequences/" + str(i) + ".csv", index=True)

    if i % 1000 ==0: print("Done:", i)
