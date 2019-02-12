"""
symbol_list.py - Load the list of symbols we are going to process.

Copyright (c) 2019 by Thomas J. Daley. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

import json
import os
import time
import urllib

# We'll use cached data that is less than this number of seconds old.
MAX_CACHE_AGE = 3*24*60*60

class SymbolLister(object):
    """
    Handles the loading of a list of symbols that we will process.
    """
    def __init__(self, symbol_limit:int =0):
        """
        Class initializer.
        """
        self.symbol_file = "../data/symbols.json"
        self.url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.json"
        self.symbol_limit = symbol_limit # max number of symbols to process

    def __save_symbol_data(self, symbol_data:object)-> bool:
        """
        Save downloaded symbol data to a local file in case we can't get it on our next run.
        """
        with open(self.symbol_file, "w") as out_file:
            out_file.write(json.dumps(symbol_data))

    def __load_symbol_data(self)-> object:
        """
        Load symbols data from a recently downloaded file.
        """
        symbols = []

        try:
            with open(self.symbol_file, "r") as in_file:
                symbol_data = json.loads(in_file.read())
            symbols = [symbol["Symbol"] for symbol in symbol_data]
            print("Loaded symbols from local file:", self.symbol_file)
        except Exception as e:
            print(e)
            print("No symbol data loaded.")

        return symbols

    def get_symbols(self)-> list:
        """
        Get a list of S&P 500 stock symbols in json format. The number of symbols returned is
        controlled by the symbol_limit.
        """

        symbols = []

        # First, see if we have a cached version. If so and it's recent enough, say 72 hours, use it.
        try:
            file_time = os.stat(self.symbol_file).st_mtime
            file_age = time.time() - file_time
            if file_age < MAX_CACHE_AGE:
                print("Using cached symbol data.")
                symbols = self.__load_symbol_data()
        except Exception as e:
            print(e.with_traceback)
            return symbols

        # If cache is too old, then load from the network (unless we can't then use the ancient cached data).
        if not symbols:
            try:
                response = urllib.request.urlopen(self.url)
                symbol_data = json.loads(response.read())
                self.__save_symbol_data(symbol_data)
                symbols = [symbol["Symbol"] for symbol in symbol_data]
                print("Loaded symbols from URL:", self.url)
            except urllib.error.HTTPError as e:
                print("HTTP Error:", e)
                symbols = self.__load_symbol_data()
            except urllib.error.URLError as e:
                print("URL Error:", self.url)
                symbols = self.__load_symbol_data()
            except Exception as e:
                print("Unexpected error:", e)
                symbols = self.__load_symbol_data()

        # In dev, we might want to process fewer than all our symbols. If so,
        # set the symbol_limit to some number greater than zero.
        if self.symbol_limit > 0:
            return symbols[:self.symbol_limit]
        
        # Otherwise, return the entire list of symbols
        return symbols
