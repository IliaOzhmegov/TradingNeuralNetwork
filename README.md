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
* [Gentle Introduction to Models for Sequence Prediction with RNNs](https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/)
* [Crypto Fear & Greed Index - Bitcoin Sentiment, Emotion Analysis](https://alternative.me/crypto/fear-and-greed-index/)

## Brief overview of the resources 

##### Convolutional Networks for Stock Trading, Stanford

The place where everything started. While we had an idea on trying to visually predict the movement of stock prices, we decided to do a bit of research and this paper was among the first we had found. The paper gives an overview of applying the CNN on the graph of time series for past prices (as opposed to our research where we use candles instead) in order to make future price predictions.  The purpose of predicting the stock prices is simply to make profit. If the model predicts the movement correctly then that will result in profit for investor/trader making decisions based on that. The paper also mentions a very important aspect related to ML and AI in general - whether a presence of such tool in the market will have any dramatic impact on the market itself. All in all, the project resulted in some little success. The author suggests that for high frequencies of data it is better to use a classification approach as the price fluctuations within small periods are very minimal. 

##### A quantitative trading method using deep convolution neural network

The Deep convolution neural network has been a great success in field of image processing,but rarely applied in market portfolios. Theh paper oveviews the potential application of such networks into analyzing time series stock data and making predictions based on that. Some of suggestions from this paper were used to choose the models. 

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

To have fun and finally to fill gaps from the last term in Deep Learning  (LSTM, TDNN, RNN). Also, to build a NN that allows to predict to following  trend by the history right before it.



![goalplot](images/goalplot.png)

What's sequence of candle we should expect after 2PM?

We would like to biuld a tool for making a predictions from the candle plot, where user can select an area as shown below, and receive a prediction about the movement of the price of the S&P500 index.



![goalplot11](images/area_selection.png)


​    

# Financial & Trading Background 

As you have guessed so far the project is mainly focused on financial domain. We think it would be a good idea to provide you with some guidance into the complicated world of trading and finance in order to give a better overview and justification of some of our choices. 

## Volatility 

Simply speaking volatility is [*a measurement of price change*](https://www.wallstreetmojo.com/volatility-formula/) of a certain asset or stock. Volatility is also associated with the risk of a certain asset.  

To put it into context: 

*  High volatility - suggests that the asset is subject to sharp price fluctuations. Consequently that would  mean that investing into such an asset will be associated with higher risks as there can be a negative spike in the price. On the other side such an asset is attractive to investors/traders as a positive spike may result in a large profit. 
*  Low volatility - suggests that the asset is subject to very little or almost no price fluctuations. Investment into such assets is associated with lower risks. Usually low volatility is relevant for well-established or old markets.

At this point you probably would ask yourself whether CSPC and SP500 indexes are highly volatile? 

The answer is - mostly, or event better -  relatively no. 

Why this is important? Well, lowly volatile assets are usually part of a well-established market which is not subject to dramatic changes. This is good from data perspective, as the models are usually better predicting  such kind of data rather the one which is subject to constant strong fluctuations. So data-wise the choice was clear. 

In the light of recent events, you probably may ask yourself two questions:

* Why not BTC/ETH or any cryptocurrency?
* What would happen if the situation like with GameStop or Silver occurs? 

##### Why not crypto?

All cryptocurrencies are the part of developing market which is subject to constant dramatic changes (look at BTC & ETH price fluctuations). This would give us a hard time training our models. Additionally, from statistical point of view all cryptocurrencies are highly volatile.

![BTC's volatility](images/BTC_volatility_daily.jpeg)

While the plot above clearly represents BTC's high volatility. There is also another aspect which makes analyzing and predicting prices of crypto a bit hard - we do not know what exactly influences the prices. When we speak about company's stock (like Apple - AAPL or Goggle - GOOG stocks) you can rely on the company's revenues and financial reports/audits to make certain assumptions about the future price change and historical factors which influenced the change. The situation with crypto is not so obvious. There is a very sophisitcated analysis done to find the influencing factors (incl. buy and sell orders). A nice example of such an analysis - [Crypto Fear & Greed Index - Bitcoin Sentiment, Emotion Analysis](https://alternative.me/crypto/fear-and-greed-index/)

##### What if situation like with GameStop happens?

![GameStop_Prices_Raising](images/gamestop-stock-rise.jpg)

If the sitaution like that happens, probably our model and most exisitng models there won't be able to forsee that. Since such a behaviour is caused by factors which were not introduced before, the models won't be prepared for that. Probably considering the number of so called buy/sell orders could help to predict that but there is also the small delay between the number of buy orders increasing (as it was with GameStop) and price increasing as well. 


## Bear vs Bull 

Or also Bear vs Bullish market - what that actually means and why is that important? 

![Bear_Market_VS_Bull_Market](images/Bear_vs_Bull_Cartoon.jpg)

Simply bull market is an optimistic market which faces a positive trand in the assets' prices and bear market is the market facing a decrease in prices due to depressive enviroment or to the fact that the investors do not believe in the market. 

![Bear_Market_VS_Bull_Market_Projected_Onto_stock_Prices](images/Bear-Bull_Illusttrated.png)

On the plot above you can see that one stock can face "bull" and "bear" periods. It is important to know whether the asset is a part of bullish or bear market in general (histoical view). The first one will let us know that despite all price fluctuations there is positive trend in price (and positive sharp spike called **bullrun** may occur). The opposite applies if an  asset is mostly a part of bear market. 


# Tasks to reach our goal

GOAL: It is a well spread to suffer. Get used to it!

0. Read "Useful resources"  | DONE
1. Collect Data: (Everybody) (yahoo.finance) | DONE
2. Conduct an initial analysis. | DONE
3. Train A CNN to extract data from a picture of a candle.
   1. Draw Candle plots for a fixed range (maybe a month) and save them as jpeg/png. | DONE 
   2. Slice those candle plots into a single candle plot. | DONE 
   3. Create labels out of those candle pictures. | DONE
   4. Create CNN regression model for a candle. | DONE
4. Apply Classic Approaches for Time series data:
   1. The autoregressive model AR(p). | DONE
   2. The moving average MA(q) Model. | DONE
   3. The ARMA(p,q) Model. | DONE
   4. Maybe show partial autocorrelation function. | DONE
5. Apply a NN:
   1. Apply a time delay neural networks TDNN. | DONE
   2. Apply a simple recurrent neural network RNN. | DONE
   3. Apply a LSTM. | DONE
   4. Apply a CNN | DONE
6. Compare results.



# Data 



We have used  S&P500 index historical data for our research. We have collected the raw data from https://finance.yahoo.com/.  This is what S&P500 index looks like on the website.



![yahoo](images/yahoo.png)



**Why S&P 500?**

S&P500 is a stock market index that measures the stock performance of 500 large companies listed on stock exchanges in the United States. For example, the 10 largest companies in the index, in order of weighting, are Apple Inc., Microsoft, Amazon.com, Facebook, Tesla, Inc., Alphabet Inc., Berkshire Hathaway, Johnson & Johnson, and JPMorgan Chase & Co., respectively.

It is one of the most commonly followed equity indices. The reason why is it so popular is because it reflects the overall state of the economy really well. 

It is also well established trading index with relatively non-volitile performance. The average annual total return and compound annual growth rate of the index, including dividends, since inception in 1926 has been approximately 9.8%, or 6% after inflation; however, there were several years where the index declined over 30%. The index has posted annual increases 70% of the time. However, the index has only made new highs on 5% of trading days, meaning that on 95% of trading days, the index has closed below its all-time high.

Initially we intended to use Bitcoin data for our research. But for reasons above, we have decided to focus our research on S&P500 index instead of Bitcoin, which is relatively new actor on the market with very high volaitlity.



# Data Description



Data preparation has been a significant part of our project as we deal with images of the financial time series data - it required a lot of preprocessing for various tasks. The complete pipeline is described below in this chapter.

The original data is located in the *"^GSPC*" csv file. Name "^GSPC" is the listed symbol of the S&P500 on the NYSE, Cboe BZX Exchange, NASDAQ exchanges. Data consists of 163,638 observations and 7 variables, including: Date, Open, High, Low, Close, Adjusted Close prices and volume. Here is the sample.

![yahoo](images/gspc.png)



The time span of our data is from December 30, 1927 to January 22, 2021 - almost hundred years of data. 



**Sequences**

We have created a shortened version of the raw data - the *"SP500"* csv file. Since more reliable data is from aproximatelly 1962, we have discarted the data before January 2, 1962. 

Next, as part of the data preparation, we have create the sequences of these data. They were created by building a custom sequence distributor (see *sequence_distributor.py*) to slice data into the small windows of 30 days each, which will be used as predictors, and the slices with the span of 5 days - the target values that need to be predicted. The predictor and target data were scaled separetelly using different approaches. For example, the 30-day predictor sequences were scaled by taking the **minimum** and **maximum** values as 0.04(45) and 0.95(45) respectively. The choice of such values can be explained by scale factor of plot function So overall we have created 14,832 sequences. These sequences were used as an inputs for our Time Series Neural Networks: TDNN, LSTM, RNN.



**Pipeline** 

For analyzing the raw visual data,  we had to create a certain pipeline to put it into the proper format and then make forecasts based on that. First of all, simply inputting the whole sequence of candle plots and asking the models to make a forecast - (continue the picture) is not the best approach for our specific task. We need to detach the individual candles in the first place, remove the background noise (for instance the grid) and so on. The illustration below demonstrates how we do it. 

![Pipeline](images/pipeline_ps.png)



1. We take the  images with candle plots and then pass them trough our **selector**. The purpose of the **selector** is to find and detach each individual candle into separate images. 
2. As an output we get separated candles. 
3. Afterwards, we pass each individual candle trough **interpreter**. The purpose of interpreter is to convert  individual candles into more convenient numeric representation which we have already mentioned. 
4.  Afterwards we get the **HLOC** (High, Low, Open, and Close) values of our candles. The variable **__t__** represents the sequence of the candle. In other words the order. The index **__k__** - is the total number of candles at the input. It is very important to keep the sequence of the candles although we separate them as each next candle is dependent on previous one. 
5. As we get our data converted into convenient **HLOC** format we start the forecasting process using our ANN models by inputting those sequences into them. As an output we get the **HLOC** and **t<sub>k+1</sub>** values of the predicted candles. These predicted and input values together can be used by our candle plot generator to obtain the whole candle plot with forecasted candles. 



**Candles.**

Other part of our data preparation process we have created an images of the candles (see *candle_builder.py*). This was a tedious process, where we had to create candle plots and "cut-out" each candle separately. We did so by drawing the 30-day slices of the data, then slicing each of them into individual plots. These individual plots contain a sindle candle with preserved spatial properties (including the center). The size of each candle box is 34 (width) x 200 (hight) pixels. 

For model robustness, we have introduced a parameter *lambda λ* that controls the transparency of the candle color, but also width of the each plots and slight swing to left and right.  In addition, we have added some background noise. Here are the samples of these candles.



<img src="plots/candles/2_0.png" alt="ACF_plot" style="zoom:100%;" />      <img src="plots/candles/3_26.png" alt="ACF_plot" style="zoom:100%;" />       <img src="plots/candles/3_20.png" alt="ACF_plot" style="zoom:100%;" />        <img src="plots/candles/0_3.png" alt="ACF_plot" style="zoom:100%;" />       <img src="plots/candles/0_1.png" alt="ACF_plot" style="zoom:100%;" />



The names of the images of the individual candles are not random. They folow a pre-defined convention where first number of the image name means it belongs to the according sequence file, and the second number means the corresponding row number of that particular sequence. For example, if we are looking at the candle with name "7_1.png", it means that it belongs to the 7th sequence, 2nd row in that sequence file.

These images of the candles were used as an input for CNN. More on the CNN methods later in the report.



**Selector.**

However, further we are planning to change the naive approach above to a more robust one, which will calculate the x of "the center of mass" of every candle and then slice 17 pixels margin in both directions from the center. The center of mass can be calculated with OpenCV library.

Then our selector would pass on this new data (individual images of the candles) to the interpretator.



# CNN Regressor 



**Train Interpretator.**

We have created a so called train "interpretator" (see *"train_interpretator.py"*) that trains CNN regressor model and outputs a price predictions for our **HLOC** values (High, Low, Open, Close). The architecture of this CNN model was inspired by AlexNet. Below is the performance result of our model, which gives quite good predictions.

![train_AR](plots/training_history_of_interpretator.png)

*"check_iterpretator.py"* - checks how our interpretator works. 








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

Classic Statistical approaches to Time Series data are really good to analize the structure of the data. 
But they generally  give poor prediction results, especially the case of non-stationary data. Even 
after we try to induce the stationarity to the data - the models still don't perform well. One of the 
possible applications of these methods are for short time windows. Classic statistical methods might be 
more suitable for short term spans.


# Non-classical approaches - ANN 

## Introduction

As mentioned above classical methods are good when it comes to analyzing stationary timeseries data. 
However the data we use as mostly any stock exchange data is non-stationary. This makes Neural Network 
particular attractive for us as there is no proof that they are bad at working with non-stationary data. 
Our idea was to use four different ANNs architectures and find the one that performs the best. 

## Model Choice

We have decided to go with for NN architectures: 

* RNNs - Recurrent Neural Networks 
  * That was an obvious choice as **RNNs** can be used to model sequence of data (i.e. time series) so that 
  each sample can be assumed  to be dependent on previous ones.
  * It can be also easily integrated with other layers like CNNs to extend the effective pixel neighborhood. 
* TDNNs - Time-Delay Neural Networks 
  * The strength of the **TDNN** comes from its ability to examine objects shifted in **time** forward and 
  backward to define an object detectable as the **time** is altered.  If an object can be recognized in this 
  manner, an application can plan on that object to be found in the future and perform an optimal action.
* LSTMs - Long Short Term Memory NNs
  * The memory property of such networks helps them to keep the time related dependencies of sequence data. 
  * (**LSTM** is able to solve many **time series** tasks unsolvable by feed-forward networks using fixed 
  size **time** windows.
* CNN  - Convolutional Neural Networks
  * One of the advantages of **CNNs**  is that it automatically detects the important features, requires 
  fewer hyperparameters, and less human supervision.
  * Generally, performance-wise (computational time) **CNNs** considered to be a bit faster than **RNNs**. 

## Model Settings

**Data Split** 

The data was split in the following way : 

* The first 70% of the whole dataset for the **training** set. (0-70)
* The next 30% of the dataset for the **validation** set. (70-90)
* The rest 10% for the **testing** set. (90-100)

It should be mentioned that the periods between 70% and 90% are the bearish periods, meaning the market prices 
had stagnated and started to decrease. As mentioned above for any model it is quite hard to predict **"bull"** 
or **"bear"** runs, thus it adds additional challenge for our model. The testing set catches the period of 
**"bull"** run which also introduces additional sophistications for our models. A `WindowGenerator` was created 
to define the shift and the number of variables which will be used for the training, validation, and testing. 

**Results** 

We have decided to use two approaches:

* Simple - where we try to predict one variable  - **_"Close"_** 
* Complex - where we try to predict the whole sequence (all columns) as in multivariate time-series analysis. 

## Simple Models Results

The first thing we notice immediately for all models is that the models' training stagnate quite quickly. In 
other words there is no significant improvement after a couple or more epochs. The `EarlyStopping`  stops 
the model training after the stagnation occurs - usually at the 10th epochs. Here we are trying to predict the next day. 

![Tiny Window](plots/modelling/simiple_models/tiny_narrow_window.png)

### Simple RNN 

![RNN History](plots/modelling/simiple_models/RNN_history.png)

The simple RNN has showed quite a great results with an MSE equal to 0.06. As mentioned above the stagnation in 
training occurred quite early - at the 2<sup>nd</sup> epoch.

Illustrated below  we can see that the model has quite correctly predicted the stock price. At the first graph 
the match is very precise. 

![RNN Prediction](plots/modelling/simiple_models/prediction/RNN_prediction.png)

### Simple TDNN

![TDNN History](plots/modelling/simiple_models/TDNN_history.png)

The simple TDNN has showed worse results than RNN with the best MSE equal 0.014. For training set the stagnation 
as opposed to RNN occurred relatively late at around 14<sup>th</sup> epoch. Contrary to RNN the MSE for validation 
set differentiates from the training set. There is less improvement with additional epochs.

![TDNN Prediction](plots/modelling/simiple_models/prediction/TDNN_prediction.png)

Here we can see again that despite being not completely precise, the match is still quite good enough with a 
mismatch of maximum 0.1 on a normalized prices.

### Simple LSTM

![LSTM History](plots/modelling/simiple_models/LSTM_history.png)

As in RNN the stagnation in training occurs almost immediately, meaning there is no significant improvement 
in the performance after the  2<sup>nd</sup> epoch. The MSE curvatures both for training and validation 
correlate with each other. 

![LSTM Prediction](plots/modelling/simiple_models/prediction/LSTM_prediction.png)

Prediction-wise we can see that the model provides noticeably good results. 

### Simple CNN 

![CNN History](plots/modelling/simiple_models/CNN_history.png)

We were quite excited about applying the CNN onto our data as usually it is rarely applied to analyze such kind 
of data - rather for image analysis. At the end of the day images are also a set of numbers, so we expected some 
nice results from CNN as well. From the plot above we can clearly see that results  are  good - it can predict 
the tendency correctly, - but not the best actually opposite. Again the same tendency occurs and model quickly 
reaches saturation in the training process after the 3<sup>rd</sup> epoch. 

From the plot below we observe a large distance between the predicted and true price. Compared to previous models 
the difference is quite noticeable. 

![CNN Prediction](plots/modelling/simiple_models/prediction/CNN_prediction.png)

### Final overview of simple models

![Simple Overview](plots/modelling/simiple_models/performance.png)

First of all, there is a little difference performance-wise between the validation and test sets which good 
considering the significant difference between the two. 

Overall, for simple models the TDNN has showed the best performance. RNNs and LSTMs showed quite similar performance 
and CNN was the worst. 

## Complex Models Results

The idea of complex models was to use and predict all available variables. The models were also changed a bit for 
that purpose. Additional layers were introduced. Immediately we can notice that our models converged quick quickly
- usually at the first 5 epochs. One more difference compared to simple models is that we try to predict five days 
instead of one

![Standard Window](plots/modelling/complex_models/standard_window.png)

### Complex RNN

![Complex RNN History](plots/modelling/complex_models/RNN_history.png)

The model converged almost immediately without any significant improvement after the 2<sup>nd</sup> epoch. We should
also keep in mind that our RNN faced additional improvements like new layers and introduced dropout. That is one more 
reason why this model is called Complex RNN. 

![RNN Prediction](plots/modelling/complex_models/prediction/RNN_close_prediction.png)

From the plot above we can see that most of the times the prediction was spot on with very little distance between 
predicted and true value. 

![RNN Prediction-Comparison](plots/predictions/RNN_prediction.png)

### Complex LSTM 

![Complex LSTM History](plots/modelling/complex_models/LSTM_history.png)

As the Complex RNN the LSTM model converged quick fast - there was no significant improvement after the first-second 
epoch. Compared to simple LSTM the model demonstrated a bit better performance with MSE  = 0.04.

![Complex LSTM Prediction](plots/modelling/complex_models/prediction/LSTM_close_prediction.png)

We can see that the prediction is not as good as it was with Complex RNN, sometimes the gap between the prediction 
and true value is quite large. Interestingly enough the model continue the plot in a straight way and thus doesn't 
follow the curvature of true values.

![RNN Prediction-Comparison](plots/predictions/LSTM_prediction.png)

### Complex TDNN

![Complex TDNN History](plots/modelling/complex_models/TDNN_history.png)

Performance-wise complex TDNN performed worse than simple TDNN (which was the best among simple models). Overall 
the performance is compatible with the Complex LSTM. The model also converges quickly in terms of epochs.

![Complex TDNN Prediction](plots/modelling/complex_models/prediction/TDNN_close_prediction.png)

Sometimes the model provides a spot on prediction if we look at the third plot. It seems that it follows the general 
trend of the curve but with certain shifts. 

![RNN Prediction-Comparison](plots/predictions/TDNN_prediction.png)

### Complex CNN

![Complex CNN History](plots/modelling/complex_models/CNN_history.png)

Here the CNN has not surprised us with a great performance. However, that could be expected as we are not analyzing 
the images to be precise, so probably that is why simpler models like RNN provide a better performance. As all the 
models before the model converges quite quickly. It doesn't become better with more iterations. 

![Complex CNN Prediction](plots/modelling/complex_models/prediction/CNN_close_prediction.png) 

It seems like Complex CNN provides spot on results when it comes to predicting a straight trend. When the trend is 
slightly curved the model provides quite bad results. In other words the model is not quite good at predicting the 
shifts but still manages to some extent predict the trends. 

![RNN Prediction-Comparison](plots/predictions/CNN_prediction.png)

### Final overview of complex models

![Complex CNN Prediction](plots/modelling/complex_models/performance.png)

The complex models overall provided a better performance than simple models. However, that is not the case for TDNN 
which was the best in this regard. Nonetheless, we are mostly interested to compare the complex models with one 
another. CNN and LSTM has lost to TDNN and RNN with a very negligible difference. In the result RNN provided the 
best results with slightly better results than TDNN. 


### Comparing different ANN models

Up to this point we can see that ANNs can predict the stock prices with noticeable results. They have definitely proved 
that they worth to be used for such kind of analysis. Still business wise the performance is not good enough to connect 
those models to any trading models and use them as automated/robot traders. However, as we noticed with RNNs - there is 
always space for improvement (adding dropout or additional layers for example).


# Conclusion 



This was very interesting and complex project - we looked at the task from the various angels and applied many different models. Hovewer the project ended up being of a larger scale than we originally presumed. But this is exactly what have made this project even more interesting and we intent to continue working on it after the end of the Lerning from Images Course.

The idea of this project is to build a Convolutional Neural Network that would predict the S&P500 index market movement and price. In addition to this, we haded to compare other Time Series approaches, including classic Time Series models as well as more contemporary and recent methods such as Time Delay Neural Net, Reccurent NN, and Long Short-Term Memory NN.

Overall our project can be generalized into the following acomplished tasks:

- Financial background.
- Data collection.
- Data Processing, including creation of the candle plot and slicing them into individual single candle plots.
- Applying the classic approaches to the tme series data.
- Developing **simple** and **complex** versions of the TDNN, RNN, LSTM and CNN.
- Deriving the results.

The traditional Time Series approaches, such as Moving Average, Autoregressive and ARMA methods, resulted in low performance with poor predictions.

Time Series Neural Networks proved to be a lot more effective and much better in making predictions. 

We have developed two versions of each our our Neural Network: 

- the **simple** models that predict only one variable with outcome value at t+1.
- and **complex** models that make simultaneous multivariate multistep up to t+5 predictions.

Among the **simple** models, RNN and LSTM had almost similar performances. To our big surprise - our TDNN model had the best performance and beat other Neural Nets, giving the lowest Mean Absolute Error.

For **complex** models, RNN ended up being the best model with lowest Mean Absolute Error. TDNN model was second best - not too far off the RNN performance. And LSTM didn't perform as good with this task.

Our CNN model had the lowest performance among the Neural Net models, however it still has beat our expectations as we were not sure if this type of model would be able to make predictions on the stock market data. And to our pleasant surprise CNN model was able to make some accurate predictions even though not as accurate as Time Series Neural Nets.

