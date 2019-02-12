"""
enrich.py - Enrich the raw prices in preparation for ML training.

Copyright (c) 2019 by Thomas J. Daley. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from lib.symbol_list import SymbolLister
from lib.raw_prices import RawPrices
from lib.logger import Logger
from lib.progress_bar import ProgressBar

if __name__ == "__main__":
    logger = Logger.get_logger()
    logger.info("Starting")
    symbol_count = 500
    symbol_lister = SymbolLister(symbol_count)
    symbols = symbol_lister.get_symbols()
    logger.info("Loaded %s symbols", symbol_count)
    progress_bar = ProgressBar(len(symbols))
    iteration_count = 0

    for symbol in symbols:
        retry = True
        iteration_count += 1
        progress_bar.update(iteration_count, symbol)

        while retry:
            retry = False
            try:
                # Read CSV data and create a Pandas dataframe
                in_file = RawPrices.price_file_name(symbol)
                df = pd.read_csv(in_file, parse_dates=["timestamp"])

                # Delete noise columns (features that won't help)
                del df["split_coefficient"]
                #del df["close"]
                #df.rename(index=str, columns={"adjusted_close": "close"})

                # See how today's trading volume compares to average volume
                df["mean_volume"] = df['volume'].mean()
                df["vol_pct_mean"] = df["volume"] / df["mean_volume"]
                del df["mean_volume"]
                del df["volume"]

                # Combine prior 4 days into this day to give us a week of trading data in each row.
                # I don't understand this at all. I copied it from:
                # https://stackoverflow.com/questions/47450259/merge-row-with-next-row-in-dataframe-pandas
                # /tjd/
                n = 6
                new_df = pd.concat([df.add_suffix(1)] +
                        [df[x+1:].reset_index(drop=True).add_suffix(x+2)
                        for x in range(n)], axis=1)

                # Insert the trend data.
                for day in range(1, n):
                    # Some shortcut indices - Just makes the code a little easier to read
                    _open = "open{}".format(day)
                    _high = "high{}".format(day)
                    _low = "low{}".format(day)
                    _close = "close{}".format(day)
                    _top_tail = "top_tail{}".format(day)
                    _bot_tail = "bot_tail{}".format(day)
                    _color = "color{}".format(day)

                    # open_dir
                    #
                    # Compare today's opening to yesterday's close
                    # Opened up if > 1, Opened down if < 1, Opened flat if == 1
                    new_df["open_dir{}".format(day)] = new_df[_open] / new_df["close{}".format(day+1)]

                    # open_trend
                    #
                    # Compare today's opening to yesterday's open
                    # Opened higher than yesterday's open if > 1, Opened lower than yesterday's open if < 1, Same open if == 1
                    new_df["open_trend{}".format(day)] = new_df[_open] / new_df["open{}".format(day+1)]

                    # high_trend
                    #
                    # Compare today's high to yesterday's high
                    # Today's high was higher if > 1, Today's high was lower if < 1, Same open if == 1
                    new_df["high_trend{}".format(day)] = new_df[_high] / new_df["high{}".format(day+1)]

                    # low_trend
                    #
                    # Compare today's low to yesterday's low
                    # Higher low if > 1, lower low if < 1, Same low if == 1
                    new_df["low_trend{}".format(day)] = new_df[_low] / new_df["low{}".format(day+1)]

                    # close_trend
                    #
                    # Compare today's closing to yesterday's close
                    # Closed higher than yesterday's close if > 1, Closed lower than yesterday's close if < 1, Same open if == 1
                    new_df["close_trend{}".format(day)] = new_df[_open] / new_df["close{}".format(day+1)]

                    # body_type
                    #
                    # Compare today's opening to today's closing
                    # Closed higher than it opened if > 1, Closed lower than it opened if < 1, Closed and opened at same price if == 1
                    new_df["body_type{}".format(day)] = new_df[_close] / new_df[_open]

                    # top_tail
                    #
                    # Compare today's high to the high end of the open--close range.
                    # If the high was greater than greater of (open, close), we have a tail of some magnitude, otherwise 0.
                    new_df[_top_tail] = 0
                    new_df.loc[new_df[_close] > new_df[_open], _top_tail] = new_df[_high] / new_df[_close]
                    new_df.loc[new_df[_close] <= new_df[_open], _top_tail] = new_df[_high] / new_df[_open]

                    # bot_tail
                    #
                    # Compare today's low to the low end of the open--close range.
                    # If the low was lower than the lesser of (open, close), we have a tail of some magnitude, otherwise 0.
                    new_df[_bot_tail] = 0
                    new_df.loc[new_df[_close] > new_df[_open], _bot_tail] = new_df[_open] / new_df[_low]
                    new_df.loc[new_df[_close] <= new_df[_open], _bot_tail] = new_df[_close] / new_df[_low]

                new_df["label"] = np.where(new_df["open1"].gt(new_df["close2"]), "UP",
                                            np.where(new_df["open1"].lt(new_df["close2"]), "DOWN", "FLAT"))

                # Remove columns we don't need.
                drop_cols = []
                for day in range(1, n):
                    # Some shortcut indices - Just makes the code a little easier to read
                    drop_cols.append("open{}".format(day))
                    drop_cols.append("high{}".format(day))
                    drop_cols.append("low{}".format(day))
                    drop_cols.append("close{}".format(day))
                    drop_cols.append("adjusted_close{}".format(day))
                    drop_cols.append("timestamp{}".format(day+1)) # We'll keep the first timestamp column.

                chart_cols = ["timestamp1", "open1", "high1", "low1", "close1"]
                #chart_cols.extend(drop_cols)
                chart_df = new_df.filter(chart_cols, axis=1)
                df = new_df.drop(columns=drop_cols)

                # Add the symbols column.
                df["symbol"] = symbol
                chart_df["symbol"] = symbol

                # Save enriched dataframe to a CSV file
                df.to_csv(RawPrices.enriched_price_file_name(symbol))
                chart_df.to_csv(RawPrices.chart_file_name(symbol))
            except FileNotFoundError:
                progress_bar.update(iteration_count, symbol+" (w)")
                time.sleep(6)
                retry = True
            except Exception as e:
                logger.error("Error processing %s in %s.", symbol, in_file)
                logger.exception(e)
                retry = False
