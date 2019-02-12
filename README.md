# PricePredictor
Predict Stock Price Direction using Tensorflow and historic stock prices.

_This project downloads the list of stock symbols that comprise the S&P 500. It then downloads daily price information for each stock, enriches the data to make it easier to train a Tensorflow model, trains the model, and, from there predicts whether a stock's price is likely to go UP, DOWN, or stay FLAT the next day._

# download.py
_Obtain a list of S&P 500 component stocks._
This program looks to see if there is a local cache of S&P 500 stocks. If there is a cache file and it is less than three days old, the program uses that cache.
If there is no cache file or if thre is one but it is more than 3 days old, a new one is downloaded from ```alphavantage.co```. This is a site that lets you download stock prices for free, but you are limited to 500 requests per day at a rate of no more than 5 per minute. This program respects those limits.

# enrich.py
_Process the downloaded stock data into new datasets for charting and training the mode._
