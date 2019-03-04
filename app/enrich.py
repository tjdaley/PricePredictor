"""
enrich.py - Enrich the raw prices in preparation for ML training.

Copyright (c) 2019 by Thomas J. Daley. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from lib.symbol_list import SymbolLister
from lib.raw_prices import RawPrices
from lib.logger import Logger
from lib.progress_bar import ProgressBar

def get_options():
    """
    Process command line arguments.
    """
    parser = argparse.ArgumentParser(description="Enrich price data for training and prediction.")
    parser.add_argument("--symbol-count", action="store", default=500, type=int, help="Number of symbols to process")
    parser.add_argument("--symbol", action="store", help="One symbol to process rather than a list.")
    parser.add_argument("--retry", action="store_true", default=False,
                        help="If specified, this program will wait indifinitely for a missing symbol's price file to arrive.")
    parser.add_argument("--pivot-days", action="store", type=int, default=6,
                        help="Number of days to pivot into a single row for trend analysis. Default is 6")
    parser.add_argument("--day-count", action="store", type=int,
                        help="""
                        Number of days to enrich. If omitted, the entire history is enriched, which is what if you
                        are training the model. If specified, only the top N rows will be enriched, which may be what
                        you want when you need to enrich only newly added data. In that case, specify a day-count that
                        is equal to the number of newly added rows (per symbol file) PLUS the number of pivot-days.
                        """)
    parser.add_argument("--status", action="store_true", default=False,
                        help="If specified, will display a simple status bar on the terminal windows.")
    args = parser.parse_args()
    return args

def remove_zeros(symbol, df, dfz, logger):

    # Move the rows having zeros to a separate dataframe.
    # Keep track of the number of records added to that dataframe.
    df_temp = dfz.head(0) # Copy columns so that axies are aligned for the append operations.
    df_temp = df_temp.append(df[df.open == 0])
    df_temp = df_temp.append(df[df.high == 0])
    df_temp = df_temp.append(df[df.low == 0])
    df_temp = df_temp.append(df[df.close == 0])
    df_zero_vol = df[df.volume == 0]

    # Count the number of zero rows that we found.
    zero_price_count = df_temp.shape[0]
    zero_vol_count = df_zero_vol.shape[0]

    # Add a symbol to the temp frame so we'll have it in our final zeros.csv file.
    df_temp["symbol"] = symbol

    # Append the zero price rows from our temp dataframe into our master ZEROS dataframe
    dfz = dfz.append(df_temp)

    # If we didn't add any rows to the zero's dataframe, we had clean data and we're done.
    if not zero_price_count and not zero_vol_count:
        return df, dfz

    # Fix zero volumes by averaging adjacent days.
    
    logger.warn("%s Will fix %s zero-volume records.", symbol, df_zero_vol.shape[0])
    for row in df_zero_vol.itertuples():
        divisor = 2
        if row.Index-1 in df.index:
            vol_above = df.loc[row.Index-1]["volume"]
        else:
            vol_above = 0
            divisor -= 1

        if row.Index+1 in df.index:
            vol_below = df.loc[row.Index+1]["volume"]
        else:
            vol_below = 0
            divisor -= 1
        
        if divisor == 0:
            break # we can't fix this.

        new_volume = int((vol_above + vol_below) / divisor)
        df.loc[row.Index, "volume"] = new_volume

    # With zeros fixed, if the prices were clean, return the dataframes.
    if not zero_price_count:
        return df, dfz

    # So we found some zero prices.
    # Remove them from the main dataframe.
    df = df[df.open != 0]
    df = df[df.high != 0]
    df = df[df.low != 0]
    df = df[df.close != 0]
    df = df[df.volume != 0]

    logger.warn("%s Removed %s rows with ZERO values.", symbol, zero_price_count)
    return df, dfz

if __name__ == "__main__":
    logger = Logger.get_logger("prxpred.enrich")
    logger.info("Starting")
    args = get_options()

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbol_lister = SymbolLister(args.symbol_count)
        symbols = symbol_lister.get_symbols()
        logger.info("Loaded %s symbols", len(symbols))

    progress_bar = ProgressBar(len(symbols))
    symbol_count = 0
    df_zeros = pd.DataFrame()

    for symbol in symbols:
        retry = True
        symbol_count += 1

        if args.status:
            progress_bar.update(symbol_count, symbol.ljust(5, ' '))

        while retry:
            retry = False
            try:
                # Read CSV data and create a Pandas dataframe
                in_file = RawPrices.price_file_name(symbol)
                df = pd.read_csv(in_file, parse_dates=["timestamp"])

                # Limit the number of rows we process
                if args.day_count:
                    df = df.head(args.day_count)

                # Remove rows with ZEROS
                df, df_zeros = remove_zeros(symbol, df, df_zeros, logger)

                # See how today's trading volume compares to average volume
                df["mean_volume"] = df['volume'].mean()
                df["vol_pct_mean"] = df["volume"] / df["mean_volume"]

                # Delete noise columns (features that won't help)
                df.drop(columns=["split_coefficient", "mean_volume", "volume"])

                # Combine prior N days into this day to give us a week of trading data in each row.
                # I copied this from:
                # https://stackoverflow.com/questions/47450259/merge-row-with-next-row-in-dataframe-pandas
                # /tjd/
                n = args.pivot_days
                new_df = pd.concat([df.add_suffix(1)] +
                        [df[x+1:].reset_index(drop=True).add_suffix(x+2)
                        for x in range(n)], axis=1)

                # Insert the trend data.
                for day in range(1, n+1):
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

                # The data labels are literal. If the day closed up at all, it's labeled up, etc.
                # TODO: To generate stronger signals, consider making the FLAT range a little broader than
                #       absolute zero, i.e. days that are only very slightly up or down, treat them as FLAT
                #       and see if the UP/DOWN signals are stronger as a result.
                new_df["label"] = np.where(new_df["open1"].gt(new_df["close2"]), "UP",
                                            np.where(new_df["open1"].lt(new_df["close2"]), "DOWN", "FLAT"))

                # Remove columns we don't need.
                drop_cols = []
                for day in range(1, n+2):
                    # Some shortcut indices - Just makes the code a little easier to read
                    drop_cols.append("open{}".format(day))
                    drop_cols.append("high{}".format(day))
                    drop_cols.append("low{}".format(day))
                    drop_cols.append("close{}".format(day))
                    drop_cols.append("adjusted_close{}".format(day))
                    if day > 1:
                        drop_cols.append("timestamp{}".format(day)) # We'll keep the first timestamp column.

                # Drop all the unnamed columns
                col_names = new_df.columns
                for col_name in col_names:
                    if "UNNAMED" in col_name.upper():
                        drop_cols.append(col_name)

                chart_cols = ["timestamp1", "open1", "high1", "low1", "close1"]
                #chart_cols.extend(drop_cols)
                chart_df = new_df.filter(chart_cols, axis=1)
                df = new_df.drop(columns=drop_cols)

                # Drop the last n+1 rows because they won't have enough data rolled into them to be
                # useful AND they'll cause the training program to blow up when normalizing feature values.
                df.drop(df.tail(n+1).index, inplace=True)
                chart_df.drop(chart_df.tail(n+1).index, inplace=True)

                # Drop rows having NaN in any column. These will probably be the last n records
                # where there is not enough prior history to fill in all the columns.
                df.dropna()
                chart_df.dropna()

                # Add the symbols column.
                df["symbol"] = symbol
                chart_df["symbol"] = symbol

                # Save enriched dataframe to a CSV file
                df.to_csv(RawPrices.enriched_price_file_name(symbol))
                chart_df.to_csv(RawPrices.chart_file_name(symbol))
            except FileNotFoundError:
                if args.status:
                    progress_bar.update(symbol_count, symbol+" (w)")
                if args.retry:
                    time.sleep(6)
                retry = args.retry
            except ValueError as e:
                # Probably a price file that has only error messages in it.
                logger.error("Error processing %s in %s: %s", symbol, in_file, e)
                retry = False
            except Exception as e:
                logger.error("Error processing %s in %s.", symbol, in_file)
                logger.exception(e)
                retry = False

    # Save our ZERO rows for analysis
    df_zeros.to_csv("zeros.csv")
