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
class StochasticStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""
    def __init__(self, symbol, bars,K_period,D_period, higherLine, lowerLine):
        self.symbol = symbol
        self.bars = bars
        self.K_period = int(K_period)
        self.D_period = int(D_period)
        self.higherLine = int(higherLine)
        self.lowerLine = int(lowerLine)
    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0
        signals['close'] = self.bars['Close']
        signals['L14'] = self.bars['Low'].rolling(window=self.lowerLine).min()
        signals['H14'] = self.bars['High'].rolling(window=self.higherLine).max()
        signals['K'] = self.K_period * ((self.bars['Close'] - signals['L14']) / (signals['H14'] - signals['L14']))
        signals['D'] = signals['K'].rolling(window=self.D_period).mean()
        signals['signal'] = np.where(signals['K'] > signals['D'], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        # signals['Sell Entry'] = ((signals['K'] < signals['D']) & (signals['K'].shift(1) > signals['D'].shift(1))) & (signals['D'] > 80)
        # signals['Buy Entry'] = ((signals['K'] > si                                                                                                                                                         xgnals['D']) & (signals['K'].shift(1) < signals['D'].shift(1))) & (signals['D'] < 20)
        # # Create empty "Position" column
        # signals['positions'] = np.nan
        # # Set position to -1 for sell signals
        # signals.loc[signals['Sell Entry'], 'positions'] = -1
        # # Set position to -1 for buy signals
        # signals.loc[signals['Buy Entry'], 'positions'] = 1
        # # Set starting position to flat (i.e. 0)
        # signals['positions'].iloc[0] = 0
        # # Forward fill the osition column to show holding of positions through time
        # signals['positions'] = signals['positions'].fillna(method='ffill')
        # # print(signals)
        return signals



