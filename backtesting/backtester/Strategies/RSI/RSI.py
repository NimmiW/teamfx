import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests as req
import json
import re
import csv
import quandl
from backtesting.backtester.BackTest.backtest import Strategy, Portfolio
class RSIStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""
    def __init__(self, symbol, bars, rup, rdown):
        self.symbol = symbol
        self.bars = bars
        self.rup = int(rup)
        self.rdown = int(rdown)
    def generate_signals(self):
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0
        signals['close'] = self.bars['Close']
        delta = signals['close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        signals['rUp'] = up.ewm(com=self.rup - 1, adjust=False).mean()
        signals['rDown'] = down.ewm(com=self.rdown - 1, adjust=False).mean().abs()
        signals['RSI'] = 100 - 100 / (1 + signals['rUp'] / signals['rDown'])
        count =0
        for x in signals['RSI']:
            if x >= 70 and signals['RSI'][count-1]<70:
                signals['signal'][count] = -1.0
            else:
                if x <=30 and signals['RSI'][count-1]>30:
                  signals['signal'][count] = 1.0
            count = count + 1
        signals['positions'] = signals['signal']







        print(signals)
        return signals