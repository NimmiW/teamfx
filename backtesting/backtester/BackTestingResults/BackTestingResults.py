from backtesting.backtester.BackTest.backtest import Strategy, Portfolio
import pandas as pd
import numpy as np

class MarketOnClosePortfolio(Portfolio):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self,symbol, bars, signals, initial_capital=100000.0):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = self.signals['positions']
        return positions

    def backtest_portfolio(self):
        pf = pd.DataFrame(index=self.bars.index)
        pf['close'] = self.bars['Close']
        pf['open'] = self.bars['Open']
        pf['positions'] = self.signals['positions']
        pf['signal']= self.signals['signal']
        # pf['holdings'] = self.positions.mul(self.bars['Close'], axis='index') #multiplication of the  and the close price
        # pf['cash'] = self.initial_capital - pf['holdings'].cumsum() # return the cummulative sum
        # pf['total'] = pf['cash'] + self.positions[self.symbol].cumsum() * self.bars['Close']
        daily_log_returns = np.log(self.bars['Close'] / self.bars['Close'].shift(1))
        daily_log_returns = daily_log_returns * self.signals['signal'].shift(1)
        pf['returns'] = daily_log_returns
        pf['total'] = daily_log_returns.cumsum()
        #print('total', daily_log_returns.cumsum())
        return pf
