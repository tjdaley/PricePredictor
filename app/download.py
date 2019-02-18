"""
download.py - Download stock price data

invoke like this:

LINUX:

$ export ALPHAVANTAGE_KEY=BR549 && python3 download.py

WINDOWS:

$ env:ALPHAVANTAGE_KEY = 'BR549'; python download.py

Copyright (c) 2019 by Thomas J. Daley. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

import csv
import time
import urllib
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from resources.apikeys import APIKEYS
from lib.symbol_list import SymbolLister
from lib.raw_prices import RawPrices
from lib.progress_bar import ProgressBar
from lib.logger import Logger

MAX_SYMBOLS = 500
MAX_REQUESTS_PER_MINUTE = 5
SLEEP_TIME = 60 / MAX_REQUESTS_PER_MINUTE
LOGGER = Logger.get_logger()

def get_quotes(symbols:list):
    """
    Get quotes for each symbol.
    """
    url = APIKEYS["alphavantage"]["time_series_daily_adjusted_url"] \
          .format(apikey = APIKEYS["alphavantage"]["apikey"])
    raw_prices = RawPrices()
    progress_bar = ProgressBar(len(symbols))
    iteration_count = 0

    for symbol in symbols:

        try:
            iteration_count += 1
            progress_bar.update(iteration_count, symbol)
            my_url = url.format(symbol=symbol)
            LOGGER.debug(my_url)
            break
            response = urllib.request.urlopen(my_url)
            csv_data = response.read()

            # If the first 1,000 characters does not contain something that looks like an error message,
            # then save the data.
            if csv_data[:1000].find(b"Error Message") == -1 and csv_data[:1000].find(b'"Note":') == -1:
                raw_prices.save(symbol, csv_data)
                time.sleep(SLEEP_TIME)
        except urllib.error.HTTPError:
            LOGGER.error("HTTP Error retrieving data for %s", symbol)
        except urllib.error.URLError:
            LOGGER.error("URL Error retrieving data for %s from %s", symbol, my_url)
        except Exception as e:
            LOGGER.error("Error retrieving data for %s from %s", symbol, my_url)
            LOGGER.exception(e)

def main():
    """
    Main processing function.
    """
    symbol_lister = SymbolLister(MAX_SYMBOLS)
    if symbol_lister.symbol_limit > 0:
        print("Will only download the first {} symbols.".format(symbol_lister.symbol_limit))
    symbols = symbol_lister.get_symbols()
    get_quotes(symbols)

if __name__ == "__main__":
    main()