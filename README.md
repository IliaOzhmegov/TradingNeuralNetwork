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

# Introduction

Stock Market is very inpredictibale and throughout it's history there have been many attempts to try to predict it's movement.

There are many useful computational intelligence techniques for financial trading systems, including traditional Time Series analysis as well as newer more advanced methods such as Neural Networks. In our research we will look into both: the classic Time Series analysis, such as Moving Average, Autoregressive and ARMA models, as well as Neural Nets, such as Time Delay NN, RNN, LSTM and CNN.

CNN for trading or for predicting the stock market price movement, is a very new and recent approach that is still being researched and developed. So our attempt to try to build the CNN for trading predictions is purely for research and educational purposes.

For our reaseach we decided to use S&P500 index historical data.

Many traders predict the market movement by analizing so salled candle plots. We want to build a DNN that can predicts the following trend by candle plot image. 



![candleplot](images/candleplot.png)



**How does candle plot work?**

Every candle represents a certain time interval 1 day, 1 hour or 1 minute. Now  let us briefly explain to you what does mean every bar that usually is called a "candle" (because it resebles the candle).

![bar_explanation](images/bar_explanation.jpeg)



We have red and blue or usually green candles as on the picture. Red color traditionaly represents a downward movement, e.i. if the openning price was higher that a close price.  Naturally green color (or blue) of the candle means that the price went up and that our close price is higher that the open price. And upper and lower tails shows us the highest and lowest prices respectively. Whereas,  upper and downer sides of the bar show us opening price and closing not respectively.


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
    1. Draw Candle plots for a fixed range (maybe a month) and save them as jpeg/png. | DONE 
    2. Slice those candle plots into a single candle plot. | DONE 
    3. Create labels out of those candle pictures. | DONE
    4. Create CNN regression model for a candle. | in process
3. Apply Classic Approaches for Time series data:
    1. The autoregressive model AR(p). | DONE
    2. The moving average MA(q) Model. | DONE
    3. The ARMA(p,q) Model. | DONE
    4. Maybe show partial autocorrelation function. | DONE
4. Apply a NN:
    1. Apply a time delay neural networks TDNN.
    2. Apply a simple recurrent neural network RNN.
    3. Apply a LSTM.
6. Compare results.





**Why S&P 500?**

S&P500 is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States. For example, the 10 largest companies in the index, in order of weighting, are Apple Inc., Microsoft, Amazon.com, Facebook, Tesla, Inc., Alphabet Inc., Berkshire Hathaway, Johnson & Johnson, and JPMorgan Chase & Co., respectively.

It is one of the most commonly followed equity indices. The reason why is it so popular is because it reflects the overall state of the economy really well. 

It is also well established trading index with relatively non-volitile performance. The average annual total return and compound annual growth rate of the index, including dividends, since inception in 1926 has been approximately 9.8%, or 6% after inflation; however, there were several years where the index declined over 30%. The index has posted annual increases 70% of the time. However, the index has only made new highs on 5% of trading days, meaning that on 95% of trading days, the index has closed below its all-time high.

For these reasons, we have decided to focus our research on S&P500 index instead of Bitcoin, which is relatively new actor on the market with very high volaitlity.




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





# Classic Approaches to Time Series data 



We explore three main Time Series Analysis methods:

1. Moving Average model MA(q).
2. Autoregressive model AR(p).
3. The  Autoregressive and Moving Average model ARMA(p,q).



**Moving Average model MA(q)**

We start with one of the oldest and simplest time series approaches is the moving average of the stock price to proxy the recent trend of the price. Let us looks at the historical Adjusted Close Price of our S&P500 index for the entire period.

![historical_adj_close_price](plots/lena/Adjusted_Closing_price.png)

Now we apply MA(q) approach. The idea of the Moving Average mathod is that we use a q-day moving average of our price of our index, then a significant portion of the daily price noise will have been "averaged-out", so we smooth out short-term fluctuations. Thus, we can can observe more closely the longer-term behaviour of the asset, without the noise.

We compute a short moving average of one 1 year (MA253 - days) - the number of trading days in a year, and a long moving average of 5 years (MA1265 - days) of the Adj. Close Price.

![historical_MA](plots/lena/Moving_Average_all_time.png)

We can see that the Adjusted Close Price has been nicely smoothed out. 

Now let us explore the more recent data. We now look at the the last 10 years of the data and our short term MA is 44 days (2 month - 22 trading days on average in a month) and long term MA is 1 year MA(253).

![MA_10_years](plots/lena/Moving_Average_10_years.png)

The general trend is very clear.



**Autoregressive model AR(p)**

In an Autoregressive model, we forecast the Adjusted Close price using a linear combination of past values of the variable. The term autoregression indicates that it is a regression of the variable against itself - AR(p) model, an autoregressive model of order p.

We first check our data for autocorrelation.

![Autocorrelation_plot](plots/lena/Autocorrelation plot.png)



We can observe a strong positive autocorrelation for the first 5000 lags.

Now let us explore Autocorrelation Function ACF (left), and Partial Autocorrelation Function PACF (right).



<img src="plots/lena/Autocorrelation Function plot.png" alt="ACF_plot" style="zoom:60%;" /><img src="plots/lena/Partial Autocorrelation Function plot.png" alt="ACF_plot" style="zoom:60%;" />



If the time series is stationary, the ACF / PACF plots will show a quick drop-off in correlation after a small amount of lag between points. Our data is non-stationary as a high number of previous observations are correlated with future values.

Next, we fit AR model and compare the fitted train values and the test predictions for the last 5 years.

![train_AR](plots/lena/AR_Train.png)

![test_AR](plots/lena/AR_Test.png)



Train and test prediction results show that model is highly overfitted - it can't generalise well and performs poorly on unseen test data - resulting in linear fit.



**Auto Regression and Moving Average model ARMA(p,q)**

ARMA models combine autoregressive and moving average models and used to forecast a time series. The notation ARMA(p, q) refers to the model with p autoregressive terms and q moving-average terms.

The method is suitable for univariate time series without trend and seasonal components, this is why we first try to make our time series stationary. This is what our data and residuals look like after Log scale transformation.

<img src="plots/lena/Log Transformed Adjusted Close Price.png" alt="ACF_plot" style="zoom:60%;" /><img src="plots/lena/Log Transformed Residuals.png" alt="ACF_plot" style="zoom:60%;" />



We can also do a seasonal decomposition to detect seasonality in out data.



![AR_test](plots/lena/Seasonal Decomposition.png)



There is strange trend in seasonality - it might be because of non-stationarity of the data plus a very long time span.

Next, we apply the ARMA(2,0) model to the log transfomed Adjusted Close Price (AR order 2, MA order 0). We compare fitted values of the train data (left) and test predictions for the last 5 years (right).

<img src="plots/lena/ARMA log Fitted Values.png" alt="ACF_plot" style="zoom:26.5%;" /><img src="plots/lena/ARMA Predictions.png" alt="ACF_plot" style="zoom:26.5%;" />



Let us also fit ARMA(4,2) model (AR order 2, MA order 0) to the log residuals and see if this has any different effect. The train fitted residuals are on the left and test residual predictions (last 5 years) are on the right.

<img src="plots/lena/ARMA log Fitted Residuals.png" alt="ACF_plot" style="zoom:26.5%;" /><img src="plots/lena/ARMA Residuals Predictions.png" alt="ACF_plot" style="zoom:26.5%;" />



Again, as with AR model, ARMA model is highly overfitted and gives very poor predictions.



**Conclusion**

Classic Statistical approaches to Time Series data are really good to analize the structure of the data. But they generally  give poor prediction results, especially the case of non-stationary data. Even after we try to induce the stationarity to the data - the models still don't perform well. One of the possible applications of these methods are for short time windows. Classic statistical methods might be more suitable for short term spans.

