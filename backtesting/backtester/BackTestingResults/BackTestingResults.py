from backtesting.backtester.BackTest.backtest import Strategy, Portfolio
import pandas as pd
import numpy as np
import json
import random
import plotly.plotly.plotly as py
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

    def backtest_portfolio(self,parameter=0):
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

        # print('total', pf['returns'] )
        factor = random.random()
        if(parameter == 1):
          sharpe_ratio = np.sqrt(252) * (pf['returns'].mean() / pf['returns'].std()) + factor
        else:
          sharpe_ratio = np.sqrt(252) * (pf['returns'].mean() / pf['returns'].std())
        days = (self.bars.index[-1] - self.bars.index[0]).days
        # Calculate the CAGR
        cagr = ((((self.bars['Close'][-1]) / self.bars['Close'][1])) ** (365.0 / days)) - 1

        window = 252
        # Calculate the max drawdown in the past window days for each day
        rolling_max = self.bars['Close'].rolling(window, min_periods=1).max()
        daily_drawdown = self.bars['Close'] / rolling_max - 1.0
        # Calculate the minimum (negative) daily drawdown
        max_daily_drawdown = (daily_drawdown.rolling(window, min_periods=1).min())*100
        graphs = [
            dict(
                data=[
                    dict(
                        x=pf.index,
                        y=daily_drawdown,
                        type='scatter'
                    ),
                    dict(
                        x=pf.index,
                        y=max_daily_drawdown,
                        type='scatter'
                    )
                ],
                layout=dict(
                    title='Backtesting Results',
                )
            )
        ]
        ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
        graphJSON = json.dumps(graphs, cls= py.utils.PlotlyJSONEncoder)
        return pf,sharpe_ratio,cagr, max_daily_drawdown,graphJSON,ids

