import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import to_datetime
import requests as req
import json
import re
import csv
import quandl
from backtesting.backtester.BackTest.backtest import Strategy, Portfolio

class BollingerBandStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""
    def __init__(self, symbol, bars, MOV, std):
        self.symbol = symbol
        self.bars = bars
        self.short_window = int(MOV)
        self.std = int(std)
    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0
        # Create the set of short and long simple moving averages over the
        # respective periods
        signals['close'] = self.bars['Close']
        signals['middlBand'] = pd.rolling_mean(self.bars['Close'], self.short_window, min_periods=1)
        signals['standardDiviation'] = self.bars['Close'].rolling(window=self.std).std()
        signals['upperBand'] = signals['middlBand'] + (signals['standardDiviation']*2)
        signals['lowerBand'] = signals['middlBand'] - (signals['standardDiviation']*2)
        signals['positions'] = 0.0
        for row in range(len(signals)):
            if (signals['close'].iloc[row] > signals['upperBand'].iloc[row]) and (
                    signals['close'].iloc[row - 1] < signals['upperBand'].iloc[row - 1]):
                signals['positions'][row] = -1

            if (signals['close'].iloc[row] < signals['lowerBand'].iloc[row]) and (
                    signals['close'].iloc[row - 1] > signals['lowerBand'].iloc[row - 1]):
                signals['positions'][row] = 1

        # signals['positions'] = signals['signal']
        return signals