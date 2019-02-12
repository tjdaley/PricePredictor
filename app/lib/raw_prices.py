"""
raw_prices.py - Save and load raw price data.

Copyright (c) 2019 by Thomas J. Daley, J.D. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

PRICE_FILE = "../data/{}_price_data.csv"
PRICE_FILE_ENRICHED = "../data/{}_enriched_price_data.csv"
CHART_FILE = "../data/{}_chart.csv"

class RawPrices(object):
    """
    Save and load raw price data.
    """

    def __init__(self):
        """
        Class Initializer.
        """
        self.price_file  = PRICE_FILE

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

    def save(self, symbol:str, price_data:str)-> bool:
        """
        Save price data for one stock to a local file.
        """
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
