"""
raw_prices.py - Save and load raw price data.

Copyright (c) 2019 by Thomas J. Daley, J.D. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

import sys

if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import pandas as pd

PRICE_FILE = "../data/{}_price_data.csv"
PRICE_FILE_ENRICHED = "../data/{}_enriched_data.csv"
CHART_FILE = "../data/{}_chart.csv"

class RawPrices(object):
    """
    Save and load raw price data.
    """

    def __init__(self, append:bool =False, truncate_to:int =0):
        """
        Class Initializer.

        Raises:
            ValueError if truncate_to is set to a non-zero value but append is False

        Args:
            append (bool): If true, the save() method will append to an existing file, if any.
            truncate_to (int): If greater than zero, the final csv file will be truncated to this many
                               rows. Only works when append is True.
        """

        if (not append) and (truncate_to):
            raise ValueError("Cannot specify a non-zero value for truncate_to unless append is True")
        self.price_file  = PRICE_FILE
        self.append_data = append
        self.truncate_to = truncate_to

    @staticmethod
    def price_file_name(symbol:str)-> str:
        """
        Return the name of the price file for a given symbol.
        """
        return PRICE_FILE.format(symbol)

    @staticmethod
    def enriched_price_file_name(symbol:str)-> str:
        """
        Return the name of the enriched price file for a given symbol.
        """
        return PRICE_FILE_ENRICHED.format(symbol)

    @staticmethod
    def chart_file_name(symbol:str)-> str:
        """
        Return the name of the chart file for a given symol.
        """
        return CHART_FILE.format(symbol)

    def append(self, symbol:str, price_data:str)-> bool:
        """
        Append price data to an existing file, if any, removing duplicates.

        Args:
            symbol (str): Stock symbol we are processing.
            price_data (str): Price data in CSV format.

        Returns:
            (bool): True if successful, otherwise False
        """

        file_name = RawPrices.price_file_name(symbol)
        csv_stream = StringIO(price_data.decode('utf-8'))
        new_df = pd.read_csv(csv_stream)

        try:
            old_df = pd.read_csv(file_name, index_col=0)
            combined_df = pd.concat([new_df, old_df], sort=False).drop_duplicates().reset_index(drop=True)
            if self.truncate_to > 0:
                combine_df = combine_df.head(self.truncate_to)
        except FileNotFoundError:
            # Nothing to append to--first time this symbol is being downloaded
            combined_df = new_df

        combined_df.to_csv(file_name)

        return True

    def save(self, symbol:str, price_data:str)-> bool:
        """
        Save price data for one stock to a local file.
        """
        if self.append_data:
            return self.append(symbol, price_data)

        try:
            file_path = RawPrices.price_file_name(symbol)
            with open(file_path, "wb") as out_file:
                out_file.write(price_data)
            return True
        except Exception as e:
            print("Unable to save data to {}".format(file_path), e)

        return False

    def load(self, symbol:str)-> bytes:
        """
        Read price data for one stock from a local file.
        """
        try:
            file_path = RawPrices.price_file_name(symbol)
            with open(file_path, "rb") as in_file:
                price_data = in_file.read()
            return price_data
        except Exception as e:
            print("Unable to save data to {}".format(file_path), e)

        return None
