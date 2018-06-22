import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as req
import json
import re
import csv
import quandl
from backtesting.backtester.BackTest.backtest import Strategy, Portfolio
class MACDStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""
    def __init__(self, symbol, bars, short_window, long_window ,signalLine):
        self.symbol = symbol
        self.bars = bars
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.signalLine = int(signalLine)
    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0
        signals['close'] = self.bars['Close']
        signals['short_ema'] = pd.ewma(self.bars['Close'], span=self.short_window)
        signals['long_ema'] = pd.ewma(self.bars['Close'], span=self.long_window)
        signals['MACD']= signals['short_ema']- signals['long_ema']
        signals['signalLine'] = pd.ewma(signals['MACD'],span=self.signalLine)
        signals['signal']= np.where(signals['MACD'] > signals['signalLine'], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        print(signals)
        return signals

