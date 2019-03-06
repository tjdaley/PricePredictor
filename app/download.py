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
__version__ = "0.0.2"

import argparse
import csv
import time
import urllib
import json

import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

from resources.apikeys import APIKEYS
from lib.symbol_list import SymbolLister
from lib.raw_prices import RawPrices
from lib.progress_bar import ProgressBar
from lib.logger import Logger

MAX_REQUESTS_PER_MINUTE = 5
SLEEP_TIME = 60 / MAX_REQUESTS_PER_MINUTE
LOGGER = Logger.get_logger("prxpred.download")

def get_options():
    """
    Process command line arguments.
    """
    parser = argparse.ArgumentParser(description="Enrich price data for training and prediction.")
    parser.add_argument("--append", action="store_true", default=False,
                        help="If specified, will append new data to any extent enriched data file for this symbol")
    parser.add_argument("--status", action="store_true", default=False,
                        help="If specified, will display a simple status bar on the terminal windows.")
    parser.add_argument("--symbol", action="store",
                        help="Use this to specify a single symbol that will be downloaded.")
    parser.add_argument("--symbol-count", action="store", default=500, help="Number of symbols to process. Default is 500.")
    parser.add_argument("--truncate-to", action="store", type=int, default=0,
                        help="If greater than zero, the final CSV file will be truncated to this many rows. Must be combined with --append.")
    args = parser.parse_args()
    return args


def get_quotes(symbols:list, args):
    """
    Get quotes for each symbol.
    """
    url = APIKEYS["alphavantage"]["time_series_daily_adjusted_url"]
    apikey = APIKEYS["alphavantage"]["apikey"]
    raw_prices = RawPrices(append=args.append, truncate_to=args.truncate_to)

    if args.status:
        progress_bar = ProgressBar(len(symbols))

    iteration_count = 0

    for symbol in symbols:

        try:
            iteration_count += 1

            if args.status:
                progress_bar.update(iteration_count, symbol.ljust(5, ' '))

            my_url = url.format(apikey=apikey, symbol=symbol)
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
    args = get_options()
    symbol_lister = SymbolLister(args.symbol_count)
    if symbol_lister.symbol_limit > 0:
        LOGGER.info("Will only download the first %s symbols.", symbol_lister.symbol_limit)

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = symbol_lister.get_symbols()

    get_quotes(symbols, args)

if __name__ == "__main__":
    main()