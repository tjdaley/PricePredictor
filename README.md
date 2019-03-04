# PricePredictor
Predict Stock Price Direction using Tensorflow and historic stock prices.

_This project downloads the list of stock symbols that comprise the S&P 500. It then downloads daily price information for each stock, enriches the data to make it easier to train a Tensorflow model, trains the model, and, from there predicts whether a stock's price is likely to go UP, DOWN, or stay FLAT the next day._

# download.py

_Obtain a list of S&P 500 component stocks and their recent daily price data._

This program looks to see if there is a local cache of S&P 500 stocks. If there is a cache file and it is less than three days old, the program uses that cache.

If there is no cache file or if thre is one but it is more than 3 days old, a new one is downloaded from ```alphavantage.co```. This is a site that lets you download stock prices for free, but you are limited to 500 requests per day at a rate of no more than 5 per minute. This program respects those limits.

The data are downloaded into CSV files, one for each symbol and generally look like this:

```
MSFT_price_data.csv
```

```csv
timestamp,open,high,low,close,adjusted_close,volume,dividend_amount,split_coefficient
2019-02-11,106.2000,106.5800,104.9650,105.2500,105.2500,18529025,0.0000,1.0000
2019-02-08,104.3900,105.7800,104.2603,105.6700,105.6700,21461093,0.0000,1.0000
2019-02-07,105.1850,105.5900,104.2900,105.2700,105.2700,29760697,0.0000,1.0000
2019-02-06,107.0000,107.0000,105.5300,106.0300,106.0300,20609759,0.0000,1.0000

```

## Command Line Options

```
usage: python3 download.py [-h] [--symbol-count SYMBOL_COUNT]
                   [--truncate-to TRUNCATE_TO] [--status] [--append]
```

| argument | description |
|----------|-------------|
| **-h, --help** | Prints a help message and exits. |
| **--append** | If specified, newly downloaded data will be appended to existing data. Default is to overwrite existing data. |
| **--status** | If specified, a simple progress bar will be displayed on the terminal screen. |
| **--symbol X** | If specified, price data only for the symbol *X* will be downloaded. Use this in testing or to repair a ruined file. |
| **--symbol-count N** | If specified, limits the number of stock symbols to N. Normally you wouldn't specify this parameter unless you were testing. |
| **--truncate-to N** | If specified, the final data file will be truncated to this many rows. Must be combined with --append or it will be ignored. |


# enrich.py
_Process the downloaded stock data into new datasets for charting and training the mode._

This program reads the raw pricing data downloaded into the `*price_data.csv` files and enriches for training and prediction.

Among the things that it does is normalizes the prices so that feature values from a lower-priced stock are comparable to feature
values from a higher-priced stock. Also, intraday and interday relationships are normalized so that "big" tails are of the same
relative size no matter what the stock price is. Trading volume is converted into a percentage of historical average daily volume
(this probably needs to be changed to a percentage of a long-term moving average).

It does other things, but they are futile to document because this is where most of the experimentation happens. This is where
features are selected, created, normalized, etc. In other words, this is the brains of the system.

## Command Line Options

```
usage: python3 enrich.py [-h] [--symbol-count SYMBOL_COUNT] [--symbol SYMBOL]
                 [--retry] [--pivot-days PIVOT_DAYS] [--day-count DAY_COUNT]
                 [--status]
```

| argument | description |
|----------|-------------|
| **-h, --help** | Prints a help message and exits. |
| **--day-count N** | If specified, only the first N rows of price data will be enriched. If omitted, the entire price history file is enriched, which is what you want when you are about to train the model. If the model is already trained and you're just adding data to the enriched file for predictions, then specify a value for N that is equal to the number of newly added days picked up from the download process PLUS the number of pivot-days. So if you downloaded on day of price data per symbol and appended that data to the raw price files and assuming a pivot-days value of 6, you would specify 7 as the value of N. It is OK to specify a larger value for N than you actually need (other than it will slow the processing), but you'll wreck yourself if you specify a number that is too small. |
| **--pivot-days N** | Specifies the number of days of data that will be pivoted into a single row of data. Default is 6. |
| **--retry** | If specified, will wait forever for a missing symbol file. Only use this flag if you are running the download.py program from a different terminal session at the same time. In that case, as data are downloaded, it will be enriched. The download process is slower than the enrichment process so this keeps the two in sync. |
| **--status** | If specified, a simple progress bar will be displayed on the terminal screen. |
| **--symbol X** | If specified, price data only for the symbol *X* will be enriched. |
| **--symbol-count N** | If specified, limits the number of stock symbols to N. Normally you wouldn't specify this parameter unless syou were testing. |

# train.py

_Train the neural network using enriched stock price data._

This program loads enriched stock price data, trains a neural network, and saves the trained model to disk to used for preductions.

## Command Line Options

```
usage: python3 train.py [-h] [--bsize BSIZE] [--days DAYS] [--dropout DROPOUT]
                [--epochs EPOCHS] [--estimate] [--load-only]
                [--model-name MODEL_NAME] [--seed SEED] [--splits SPLITS]
                [--status] [--symbol SYMBOL] [--verbose]
```

### Training Hyperparameters

| argument | description |
|----------|-------------|
| **--bsize N** | Sets the training batch size. Default is 5. |
| **--dropout N** | Sets the dropout rate to N. Default is 0.3. Using a value greater than zero helps prevent the model from overfitting the training data. |
| **--epochs N** | Number of training epochs per split of the training data. Default is 200. |
| **--seed N** | Specifies the random number seed. Use the same number so that your results are comparable. Default is 7 |
| **--splits N** | Number of splits for the training data. Must be at least 2. Default is 10. |
| **--verbose** | Not a hyperparameter per se but used in the training process to control the amount of output produced by the training process. |

### Operational Parameters

| argument | description |
|----------|-------------|
| **-h, --help** | Prints a help message and exits. |
| **--days N** | Number of days of data to load from each enriched symbol file. If specified, loads the most recent N days' of data from each file. If omittied, all days are loaded and used for training. I set N=500, which is about 2 years of data. |
| **--model-name X** | If specified, provides a name for the model (X). Default is "prxpred". The only effect of this parameter is the names of the disk files to which the model and its weights are serialized. |
| **--status** | If specified, a simple progress bar will be displayed on the terminal screen. |


### Testing Parameters

| argument | description |
|----------|-------------|
| **--estimate** | If specified, the programm will "quickly" estimate the model's accuracy, but will not train it or save it. |
| **--load-only** | If specified, training data will be loaded and summarized, then the process will exit. No training or estimating will take place. |
| **--symbol X** | If specified, price data only for the symbol *X* will be used for training. |

## TODO

1. Add a --random-sample flag to be used in combination with --days. For now, when --days is specififed, the top N days are used. The idea of --random-sample would be for N days to be randomly selected.

# preduct.py

_Coming Soon_