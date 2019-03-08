"""
train.py - Train the Tensorflow model

Copyright (c) 2019 by Thomas J. Daley, J.D. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

import argparse
import sys
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#from sklearn.pipeline import Pipeline

from lib.logger import Logger
from lib.progress_bar import ProgressBar
from lib.symbol_list import SymbolLister
from lib.raw_prices import RawPrices

class Trainer(object):
    """
    Encapsulates the process for training our prediction model.
    """
    def __init__(self, options):
        """
        Class initializer.
        """
        self.logger = Logger.get_logger("prxpred.train")
        self.options = options
        self.model_path = "./{}_model.json".format(options.model_name)
        self.model_weights_path = "./{}_weights.h5".format(options.model_name)
        self.initialize_random()

    def initialize_random(self, seed:int =7)->None:
        """
        Initialize the random number generator to a constant value
        so that our results re reproducable.

        Args:
            seed (int): Value to use to seed the random number generator.
                        Default = 7.
        Returns:
            None
        """
        np.random.seed(seed)

    def load_dataset(self):
        """
        Load a dataset for a stock symbol or all stock symbols, depending on command line.

        Returns:
            X,Y
        """

        # If we were given a symbol, then use that one symbol as our training data.
        # Otherwise, get the full list of symbols to process.
        if self.options.symbol:
            symbols = [self.options.symbol]
        else:
            symbol_lister = SymbolLister()
            symbols = symbol_lister.get_symbols()

        # Progress bar for data loading.
        if self.options.status:
            progress_bar = ProgressBar(len(symbols))

        # Blank dataframe to append all price data into.
        dataframe = pd.DataFrame()

        # Load data for all symbols into a single dataframe.
        # The data have already been converted to percentages, more or less, so they are
        # all of the same scale. (Rf enrich.py)
        self.logger.debug("Beginning to load data for %s symbols.", len(symbols))
        sym_count = 0
        for symbol in symbols:
            sym_count += 1

            if self.options.status:
                progress_bar.update(sym_count, suffix=symbol.ljust(5, ' '))

            file_name = RawPrices.enriched_price_file_name(symbol)
            try:
                tmp_df = pd.read_csv(file_name, index_col=0)
                if self.options.days > 0:
                    tmp_df = tmp_df.head(self.options.days)
                
                dataframe = dataframe.append(tmp_df, sort=False)
            except FutureWarning as e:
                self.logger.error("Warning while processing %s: %s", symbol, e)
                self.logger.exception(e)
            except FileNotFoundError as e:
                self.logger.error("Failed to load %s: %s",symbol, e)

        self.logger.debug("Data load complete.")
        return dataframe

    def prepare_dataset(self, dataframe):
        # Let's see if we have some NaN values.
        self.logger.debug("Testing for null values.")
        df_test = dataframe[dataframe.isnull().any(axis=1)]
        if not df_test.empty:
            self.logger.warn("We have some null values and cannot continue. Check 'nulls.csv'.")
            df_test.to_csv("nulls.csv")
            return None, None

        # Continue with good data
        dataset = dataframe.values
        cols = dataframe.shape[1]

        # Dependent data (category) is in the second to last column.
        self.logger.debug("Extracting category labels.")
        Y = dataset[:,cols-2:cols-1]
        Y = np.ravel(Y)

        # Independent data (features). Skip first two columns (index and date) and last two
        # columns (category and symbol).
        self.logger.debug("Extracting and scaling feature values.")
        X = dataset[:,3:cols-2].astype(float)
        scaler = MinMaxScaler()

        try:
            scaled_X = scaler.fit_transform(X, Y)
        except ValueError as e:
            scaled_X = None
            self.logger.warn("Error scaling the X axis: %s", e)

        return scaled_X, Y

    def encode_category_labels(self, Y):
        """
        One-hot encode the category labels.
        """
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        categorical_Y = np_utils.to_categorical(encoded_Y)

        self.logger.debug("Categories: %s", Y)
        self.logger.debug("Encoded categories: %s", categorical_Y)
        return categorical_Y

    def train(self, X, Y):
        """
        Train the model and print descriptive statistics.

        Args:
            X ([float]): Features
            Y ([[]]): Labels / Categories

        Returns:
            (tf.keras.models.Sequential): Trained model
        """
        self.logger.debug("Model training starting.")
        model = baseline_model()
        model.fit(
            X,
            Y,
            batch_size=self.options.bsize,
            epochs=self.options.epochs,
            validation_split=self.options.validation_split,
            verbose=self.options.verbose
            )
        scores = model.evaluate(X, Y, verbose=0)
        self.logger.info("%s: %.2f%%", model.metrics_names[1], scores[1]*100)
        self.logger.debug("Model training complete.")
        return model

    def save(self, model)-> bool:
        """
        Save model configuration and weights to disk.

        Args:
            model (tf.keras.models.Sequential): Compiled/Trained model.

        Returns:
            (bool): True if successful, otherwise False.
        """
        try:
            model_json = model.to_json()
            with open(self.model_path, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(self.model_weights_path)
            self.logger.info("Saved model to disk as %s and %s", self.model_path, self.model_weights_path)
            return True
        except Exception as e:
            self.logger.error("Error saving model: %s", e)
            self.logger.exception(e)

        return False

    def restore(self):
        """
        Restore the model and its weights from disk.

        Args:
            None.

        Returns:
            (tf.keras.models.Sequential): Model with weights loaded into it or None if load fails.
        """
        try:
            json_file = open(self.model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            self.logger.debug("Loaded model configuration from %s.", self.model_path)

            # load weights into new model
            loaded_model.load_weights(self.model_weights_path)
            self.logger.debug("Loaded weights from %s.", self.model_weights_path)
            loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.logger.info("Restored and compiled %s model.", self.options.model_name)
            return loaded_model
        except Exception as e:
            self.logger.error("Error restoring model: %s", e)
            self.logger.exception(e)
        return None

# These parameters are defined as globals for the esimation function.
X_DIM = 0
CATEGORY_COUNT = 0
DROPOUT_RATE = 0.3
def baseline_model(): #(x_dim, category_count):
    """
    Define our baseline model.

    Args:
        None

    Returns:
        (tf.keras.models.Sequential): Fresh, configured model ready to train.
    """
    model = Sequential()
    model.add(Dense(X_DIM, input_shape=(X_DIM,), activation='relu'))
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(int((X_DIM+CATEGORY_COUNT)/2)))
    if DROPOUT_RATE > 0:
        model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(CATEGORY_COUNT, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_options()->dict:
    """
    Get command line options.
    """
    batch_size_default = 32
    epochs_default = 200
    splits_default = 10

    parser = argparse.ArgumentParser(description="Train PricePredictor model")
    parser.add_argument("--bsize", action="store", default=batch_size_default, type=int,
                        help="Batch size. Default is {}".format(batch_size_default))
    parser.add_argument("--days", action="store", default=0, type=int,
                        help="Number of days of data for each symbol. Omit to process all days.")
    parser.add_argument("--dropout", action="store", default=0.3, type=float,
                        help="Dropout rate, which helps the model from over-fitting. Default=0.3.")
    parser.add_argument("--epochs", action="store", default=epochs_default, type=int,
                        help="Number of training epochs per split. Default is {}".format(epochs_default))
    parser.add_argument("--estimate", action="store_true",
                        help="If specified, will 'quickly' estimate the models accuracy but will not actually train it or save it.")
    parser.add_argument("--load-only", action="store_true",
                        help="If specified, training data will be loaded, summarized, then the process will exit. No training will be done.")
    parser.add_argument("--model-name", action="store", default="prxpred",
                        help="Specifies the name of the model. Used in serializing/deserializing the model.")
    parser.add_argument("--seed", action="store", default=7, type=int,
                        help="Random number seed. Default is 7.")
    parser.add_argument("--splits", action="store", default=splits_default, type=int,
                        help="Number of splits for the training data. Must be at least 2. Default is {}.".format(splits_default))
    parser.add_argument("--status", action="store_true", default=False,
                        help="If specified, will show a status bar as the data sets are loaded.")
    parser.add_argument("--symbol", action="store", default=None, type=str,
                        help="Symbol to process. Omit to process all enriched data.")
    parser.add_argument("--validation-split", action="store", default=0.25, type=float,
                        help="Float beteen 0 and 1 expressing the fraction of training data reserved for validation of the model. Default is 0.25.")
    parser.add_argument("--verbose", action="store", default=0,
                        help="Verbosity control. 0=Silent (default) 1=Progress Bar 2=One line per epoch")
    args = parser.parse_args()

    if args.splits < 2:
        print("--splits must be at least 2, but you specified {}. I am using {}.".format(args.splits, splits_default))
        args.splits = 10

    if args.verbose:
        args.verbose = 1

    return args

def main():
    """
    Main processing logic.
    """

    # These globals are used by baseline_model()
    global X_DIM
    global CATEGORY_COUNT
    global DROPOUT_RATE

    args = get_options()

    trainer = Trainer(args)
    df = trainer.load_dataset()
    X, Y = trainer.prepare_dataset(df)

    if X is None or Y is None:
        trainer.logger.info("Exiting . . .")
        exit(3)

    encoded_Y = trainer.encode_category_labels(Y)

    X_DIM = len(X[1])
    CATEGORY_COUNT = len(encoded_Y[0])
    DROPOUT_RATE = args.dropout

    trainer.logger.debug("Features: %s Categories: %s", X_DIM, CATEGORY_COUNT)

    # If we are just loading data (for testing), it's time to exit.
    if args.load_only:
        sys.exit(0)

    # Estimate how well our model will work. Invoked when user includes --estimate on the command line.
    if args.estimate:
        try:
            estimator = KerasClassifier(build_fn=baseline_model, epochs=args.epochs, batch_size=args.bsize, verbose=args.verbose)
            kfold = KFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
            results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
            trainer.logger.info("Baseline: (%s) %.2f%% (%.2f%%)", args.model_name, results.mean()*100, results.std()*100)
        except KeyboardInterrupt:
            sys.exit(2)
        sys.exit(0)

    # Train and save the model
    model = trainer.train(X, encoded_Y)
    trainer.save(model)

    # Restore and test the saved model
    model = trainer.restore()
    score = model.evaluate(X, encoded_Y, verbose=0)
    trainer.logger.info("(%s) %s: %.2f%%", args.model_name, model.metrics_names[1], score[1]*100)

if __name__ == "__main__":
    main()