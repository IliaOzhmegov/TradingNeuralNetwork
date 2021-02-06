import pandas as pd
from libs.common import Scaler

df = pd.read_csv("data/SP500.csv", index_col=0, parse_dates=True)
df.head()
