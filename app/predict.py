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

from lib.logger import Logger
from lib.progress_bar import ProgressBar
from lib.symbol_list import SymbolLister
from lib.raw_prices import RawPrices

def get_options()->dict:
    """
    Get command line options.
    """
    batch_size_default = 5
    epochs_default = 200
    splits_default = 10

    parser = argparse.ArgumentParser(description="Predict tomorrow's stock prices")
    parser.add_argument("--estimate", action="store_true", help="If specified, will 'quickly' estimate the models accuracy but will not actually train it or save it.")
    parser.add_argument("--seed", action="store", default=7, type=int, help="Random number seed. Default is 7.")
    parser.add_argument("--days", action="store", default=0, type=int, help="Number of days of data for each symbol. Omit to process all days.")
    parser.add_argument("--splits", action="store", default=splits_default, type=int,
                        help="Number of splits for the training data. Must be at least 2. Default is {}.".format(splits_default))
    parser.add_argument("--epochs", action="store", default=epochs_default, type=int,
                        help="Number of training epochs per split. Default is {}".format(epochs_default))
    parser.add_argument("--bsize", action="store", default=batch_size_default, type=int,
                        help="Batch size. Default is {}".format(batch_size_default))
    parser.add_argument("--dropout", action="store", default=0.3, type=float,
                        help="Dropout rate, which helps the model from over-fitting. Default=0.3.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="If specified, will produce verbose output.")
    parser.add_argument("--status", action="store_true", default=False,
                        help="If specified, will show a status bar as the data sets are loaded.")
    parser.add_argument("--symbol", action="store", default=None, type=str, help="Symbol to process. Omit to process all enriched data.")
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
    X, Y = trainer.load_dataset()

    if X is None or Y is None:
        trainer.logger.info("Exiting . . .")
        exit(3)

    encoded_Y = trainer.encode_category_labels(Y)

    X_DIM = len(X[1])
    CATEGORY_COUNT = len(encoded_Y[0])
    DROPOUT_RATE = args.dropout

    trainer.logger.debug("Features: %s Categories: %s", X_DIM, CATEGORY_COUNT)

    # Estimate how well our model will work when user includes --estimate on the command line.
    if args.estimate:
        try:
            estimator = KerasClassifier(build_fn=baseline_model, epochs=args.epochs, batch_size=args.bsize, verbose=args.verbose)
            kfold = KFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
            results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
            trainer.logger.info("Baseline: %.2f%% (%.2f%%)", results.mean()*100, results.std()*100)
        except KeyboardInterrupt:
            sys.exit(2)
        exit(0)

    # Train and save the model
    model = trainer.train(X, encoded_Y)
    trainer.save(model)

    # Restore and test the saved model
    model = trainer.restore()
    score = model.evaluate(X, encoded_Y, verbose=0)
    trainer.logger.info("%s: %.2f%%", model.metrics_names[1], score[1]*100)

if __name__ == "__main__":
    main()