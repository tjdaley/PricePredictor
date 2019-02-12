# PricePredictor
Predict Stock Price Direction using Tensorflow and historic stock prices.

_This project downloads the list of stock symbols that comprise the S&P 500. It then downloads daily price information for each stock, enriches the data to make it easier to train a Tensorflow model, trains the model, and, from there predicts whether a stock's price is likely to go UP, DOWN, or stay FLAT the next day._

# download.py
_Obtain a list of S&P 500 component stocks._

This program looks to see if there is a local cache of S&P 500 stocks. If there is a cache file and it is less than three days old, the program uses that cache.

If there is no cache file or if thre is one but it is more than 3 days old, a new one is downloaded from ```alphavantage.co```. This is a site that lets you download stock prices for free, but you are limited to 500 requests per day at a rate of no more than 5 per minute. This program respects those limits.

The data are downloaded into CSV files, one for each symbol and generally look like this:

```csv
MSFT_price_data.csv

timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient
2019-02-11,106.2000,106.5800,104.9650,105.2500,105.2500,18529025,0.0000,1.0000
2019-02-08,104.3900,105.7800,104.2603,105.6700,105.6700,21461093,0.0000,1.0000
2019-02-07,105.1850,105.5900,104.2900,105.2700,105.2700,29760697,0.0000,1.0000
2019-02-06,107.0000,107.0000,105.5300,106.0300,106.0300,20609759,0.0000,1.0000

```

# enrich.py
_Process the downloaded stock data into new datasets for charting and training the mode._

**coming soon**
