# Convention 

Folders and their meaning:

* Folder `analysis` contains your python or jupyter notebook where put analysis 
for a certain topic. Do not forget to put your name at the beginning.
* Folder `data` contains text data for training a model or for an analysis.
* Folder `images` contains images for a report.
* Folder `libs` contains framework that you would need more than once and you
would like to make it into a nice wrapper (e.g. a window scaler).
* Folder `papers` contains articles that you found useful and decided to save 
as access via a link is not possible.
* Folder `plots` contains your plots after analysis.
* Folder `scripts` contains ETL or certain movements for a data transformation.

Common rules:

* DO NOT PUSH your changes directly into MASTER branch: work in your local 
branch, then push your local branch, request a pull-request and after that 
let the gatekeeper (Ilia) know that you want to do a merge.
* Put your name(s) at the beginning of the python jupyter file(s).
* If you are not sure, ask. It is faster to prevent an issue than to solve it.


# Convolutional Networks for Stock Trading

We will try to use convolutional networks to predict movements in stock prices
from a picture of a time series of past price fluctuations, with the ultimate 
goal of using them to buy and sell shares of stock in order to make a profit.

## Useful resources 

* [Convolutional Networks for Stock Trading, Stanford](http://cs231n.stanford.edu/reports/2015/pdfs/ashwin_final_paper.pdf)
* [A quantitative trading method using deep convolution neural network](https://iopscience.iop.org/article/10.1088/1757-899X/490/4/042018/pdf)
* [Predicting the Trend of Stock Market Index Using the Hybrid Neural Network Based on Multiple Time Scale Feature Learning](papers/applsci-10-03961.pdf)
* [Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach)
* [Stock Market Buy/Sell/Hold prediction Using convolutional Neural Network](https://github.com/nayash/stock_cnn_blog_pub)
* [Kaggle: Predicting Stock Buy/Sell signal using CNN](https://www.kaggle.com/darkknight91/predicting-stock-buy-sell-signal-using-cnn/#data)
* [yahoo finance](https://finance.yahoo.com/quote/BTCUSD%3DX/history?p=BTCUSD%3DX)
* [Gentle Introduction to Models for Sequence Prediction with RNNs] (https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/)

# Intro

We want to build a DNN that can predict the following trend by candle plot.

![candleplot](images/candleplot.png)

Every candle represents a certain time interval 1 day, 1 hour or 1 minute. Now 
let me briefly explain to you what does mean every bar that usually 
called a "candle".

![bar_explanation](images/bar_explanation.jpeg)

Weeeell, we have red and blue or usually green candles on the picture. And upper
and lower tails shows us the highest and lowest prices respectively. Whereas, 
upper and downer sides of the bar show us opening price and closing not respectively.
If the bar is blue or green it means the opening price is lower than closing. 
If the bar is red vice versa.

# Our main goal

To have fun and finally to fill gaps from the last term in Deep Learning 
(LSTM, autoencoders). Also, to build a NN that allows to predict to following 
trend by the history right before it.

![goalplot](images/goalplot.png)

What's sequence of candle we should expect after 2PM?

# Tasks to reach our goal

GOAL: It is a well spread to suffer. Get used to it!

0. Read "Useful resources"  | NOT DONE
1. Collect Data: (Everybody) (yahoo.finance) | DONE
2. Conduct an initial analysis. | DONE
2. Train A CNN to extract data from a picture of a candle.
    1. Draw Candle plots for a fixed range (maybe a month) and save them as jpeg/png. | NOT DONE yet
    2. Slice those candle plots into a single candle plot. | NOT DONE yet
    3. Create labels out of those candle pictures. | DONE
    4. Create CNN regression model for a candle.
3. Apply Classic Approaches for Time series data:
    1. The autoregressive model AR(p). | DONE
    2. The moving average MA(q) Model. | DONE
    3. The ARMA(p,q) Model.
    4. Maybe show partial autocorrelation function.
4. Apply a NN:
    1. Apply a time delay neural networks TDNN.
    2. Apply a simple recurrent neural network RNN.
    3. Apply a LSTM.
5. Compare result.
    


# Initial Analysis 

Candles for the whole period.

![all_candels](plots/initial_analysis/SP500_all.png)

There is something odd between 1948 and 1967.

![all_candels_before_1962](plots/initial_analysis/SP500_before_1962.png)

Red dots are barely seen. Let's look at those as on a line.

![all_candels_before_1962](plots/initial_analysis/SP500_before_1962_line.png)

Looks okay, but perhaps there was something wrong with data originally. So we 
will ignore these data until `1962-01-03`. So we have the next data for every
day (14 867 observations). 

![all_candels_before_1962](plots/initial_analysis/SP500_after_1962.png)

