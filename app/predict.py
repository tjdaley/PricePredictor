"""
predict.py - Predict whether a stock will be up, down, or flat tomorrow.

Copyright (c) 2019 by Thomas J. Daley, J.D. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

import argparse
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from train import Trainer

from lib.logger import Logger
from lib.progress_bar import ProgressBar
from lib.symbol_list import SymbolLister
from lib.raw_prices import RawPrices

class Predictor(object):
    """
    Encapsulates our prediction logic.
    """
    def __init__(self, options, model):
        """
        Class initializer.

        Args:
            options (args): Command line arguments.
            model (tf.keras.models.Sequential): Restored model to used for making predictions
        """
        self.options = options
        self.model = model
        self.logger = Logger.get_logger("prxpred.predict")

    def load_dataset(self):
        """
        Load dataset to base predictions from.
        """

        # If we were given a symbol, then predictg that one symbol.
        # Otherwise, get the full list of symbols to process.
        if self.options.symbol:
            symbols = [self.options.symbol]
        else:
            symbol_lister = SymbolLister(self.options.symbol_count)
            symbols = symbol_lister.get_symbols()

        # Prepare our progress bar, if any.
        if self.options.status:
            progress_bar = ProgressBar(len(symbols))

        # Iterate through the symbol list to create a dataset of symbols to make predictions for.
        symbol_count = 0
        dataframe = pd.DataFrame()

        for symbol in symbols:
            # Update progress bar, if any
            if self.options.status:
                symbol_count += 1
                progress_bar.update(symbol_count, suffix=symbol.ljust(5, ' '))

            # Load symbol data for this symbol
            file_name = RawPrices.enriched_price_file_name(symbol)
            try:
                tmp_df = pd.read_csv(file_name, index_col=0, parse_dates=["timestamp1"])

                if self.options.trade_date:
                    start_date = self.options.trade_date
                    end_date = start_date # for now.
                    mask = (tmp_df['timestamp1'] >= start_date) & (tmp_df['timestamp1'] <= end_date)
                    tmp_df = tmp_df.loc[mask]
                else:
                    tmp_df = tmp_df.head(1)

                dataframe = dataframe.append(tmp_df, sort=False)
            except FutureWarning as e:
                self.logger.error("Warning while processing %s: %s", symbol, e)
                self.logger.exception(e)
            except FileNotFoundError as e:
                self.logger.error("Failed to load %s: %s",symbol, e)

        # Get the MAX of the timestamp field and warn about dates that are older than the max.
        # They probably indicate a failed download OR a bad symbol (such BF.B).
        dataframe.to_csv("predict_input.csv")
        dataframe.reset_index()
        max_date = dataframe.max()["timestamp1"]
        mask = (dataframe['timestamp1'] < max_date)
        df_old = dataframe.loc[mask]
        old_count = df_old.shape[0]
        if old_count > 0:
            self.logger.warn("Found %s symbols with price data older than %s.", old_count, max_date)
            self.logger.warn(df_old["symbol"])
        return dataframe[dataframe.timestamp1 == max_date]

    def scale_data(self, dataframe):
        """
        Scale the data.
        TODO: Should we be doing this in enrich.py?
        """
        dataset = dataframe.values
        cols = dataframe.shape[1]

        # Independent data (features). Skip first two columns (index and date) and last two
        # columns (category and symbol).
        X = dataset[:,3:cols-2].astype(float)
        scaler = MinMaxScaler()

        try:
            scaled_X = scaler.fit_transform(X)
        except ValueError as e:
            scaled_X = None
            self.logger.warn("Error scaling the X axis: %s", e)

        return scaled_X

    def predict(self, dataframe, scaled_X):
        """
        Make predictions.
        """
        predictions = self.model.predict_on_batch(scaled_X)
        return predictions


def get_options()->dict:
    """
    Get command line options.
    """
    parser = argparse.ArgumentParser(description="Predict tomorrow's stock prices")
    parser.add_argument("--model-name", action="store", default="prxpred",
                        help="Specifies the name of the model. Used in serializing/deserializing the model.")
    parser.add_argument("--trade-date", action="store",
                        help="Specify a trade date in YYY-MM-DD form to predict rows having that specific date.")
    parser.add_argument("--status", action="store_true", default=False,
                        help="If specified, will show a status bar as the data sets are loaded.")
    parser.add_argument("--symbol", action="store", default=None, type=str, help="Symbol to process. Omit to process all enriched data.")
    parser.add_argument("--symbol-count", action="store", default=0, type=int,
                        help="Number of symbols to process. Default is to process all of them.")
    args = parser.parse_args()

    return args

def main():
    """
    Main processing logic.
    """

    args = get_options()
    trainer = Trainer(args)
    model = trainer.restore()

    if not model:
        trainer.logger.info("Exiting . . .")
        exit(3)

    predictor = Predictor(args, model)
    df = predictor.load_dataset()
    predictor.logger.debug(df.head(1))
    X, Y = trainer.prepare_dataset(df)
    predictions = predictor.predict(df, X)

    index = 0
    labels = sorted(["UP", "DOWN", "UNKNOWN"]) #sorted(["STRONG_UP", "STRONG_DOWN", "WEAK_UP", "WEAK_DOWN", "UNKNOWN"])
    correct = 0
    for prediction in predictions:
        symbol = df.iloc[index]["symbol"]
        truth = df.iloc[index]["label"]
        prediction_index = np.argmax(prediction)
        predictor.logger.debug("%-5s Truth=%-5s Pred=%-5s (%6.3f)", symbol, truth, labels[prediction_index], prediction[prediction_index]*100)
        if truth == labels[prediction_index]:
            correct += 1
        index +=1

    predictor.logger.debug("Correct: %s / %s = %6.3f%%", correct, index, correct/index*100)

if __name__ == "__main__":
    main()