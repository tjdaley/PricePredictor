"""
train.py - Train the Tensorflow model

Copyright (c) 2019 by Thomas J. Daley, J.D. All Rights Reserved.
"""
__author__ = "Thomas J. Daley, J.D."
__version__ = "0.0.1"

import numpy as np
import pandas as pd
from Tensorflow.keras.models import Sequential

from lib.logger import Logger
from lib.progress_bar import ProgressBar
from lib.symbol_list import SymbolLister
from lib.raw_prices import RawPrices


