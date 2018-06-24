import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import to_datetime
import requests as req
import json
import re
import csv
import quandl
import backtesting.backtester.fuzzySystem.membership as fuzz
import backtesting.backtester.fuzzySystem.control as ctrl
from backtesting.backtester.BackTest.backtest import Strategy, Portfolio
class FuzzyStochasticStrategy(Strategy):
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
        signals['close'] = self.bars['Close']
        signals['signal'] = 0.0
        signals['L14'] = self.bars['Low'].rolling(window=self.lowerLine).min()
        signals['H14'] = self.bars['High'].rolling(window=self.higherLine).max()
        signals['K'] = 100 * ((self.bars['Close'] - signals['L14']) / (signals['H14'] - signals['L14']))
        signals['D'] = signals['K'].rolling(window=self.D_period).mean()
        normalizedInput = ctrl.Antecedent(np.arange(-11.51187, 3.93323, 0.00001), 'normalizedInput')
        fuzzyOutput = ctrl.Consequent(np.arange(-1, 1, 0.00001), 'fuzzyOutput')
        signals['fuzzyInput'] = (signals['K'] - signals['D']/signals['K'])
        signals['Input'] = signals['K'] - signals['D']
        fuzzyThreshold = 0.25
        normalizedInput = ctrl.Antecedent(np.arange(-1 * fuzzyThreshold, fuzzyThreshold, 0.00001), 'normalizedInput')
        fuzzyOutput = ctrl.Consequent(np.arange(-1, 1, 0.001), 'fuzzyOutput')

        normalizedInput['high'] = fuzz.trimf(normalizedInput.universe, [0, 0, fuzzyThreshold])
        normalizedInput['low'] = fuzz.trimf(normalizedInput.universe, [-1 * fuzzyThreshold, 0, 0])
        normalizedInput['medium1'] = fuzz.trimf(normalizedInput.universe, [-1 * fuzzyThreshold, -1 * fuzzyThreshold, 0])
        normalizedInput['medium2'] = fuzz.trimf(normalizedInput.universe, [0, fuzzyThreshold, fuzzyThreshold])

        fuzzyOutput['low'] = fuzz.trimf(fuzzyOutput.universe, [-1, -0.075, -0.025])
        fuzzyOutput['high'] = fuzz.trimf(fuzzyOutput.universe, [0.025, 0.075, 1])
        fuzzyOutput['medium'] = fuzz.trimf(fuzzyOutput.universe, [-0.025, 0, 0.025])

        rule1 = ctrl.Rule(normalizedInput['low'], fuzzyOutput['low'])
        rule2 = ctrl.Rule(normalizedInput['medium1'], fuzzyOutput['medium'])
        rule3 = ctrl.Rule(normalizedInput['high'], fuzzyOutput['high'])
        rule4 = ctrl.Rule(normalizedInput['medium2'], fuzzyOutput['medium'])

        movingAverage_ctrl = ctrl.ControlSystem([rule1, rule3, rule2, rule4])

        movingAverageCrossOver = ctrl.ControlSystemSimulation(movingAverage_ctrl)
        i = 0;
        for x in signals['fuzzyInput']:
            movingAverageCrossOver.input['normalizedInput'] = round(x, 5)
            movingAverageCrossOver.compute()
            print("fuzzyOutput", movingAverageCrossOver.output['fuzzyOutput'])
            if movingAverageCrossOver.output['fuzzyOutput'] >= -1 and movingAverageCrossOver.output[
                'fuzzyOutput'] <= -0.025:
                signals['signal'][i] = 1
            else:
                if movingAverageCrossOver.output['fuzzyOutput'] > -0.025 and movingAverageCrossOver.output[
                    'fuzzyOutput'] < 0.025:
                    signals['signal'][i] = 0
                else:
                    if movingAverageCrossOver.output['fuzzyOutput'] >= 0.025 and movingAverageCrossOver.output[
                        'fuzzyOutput'] <= 1:
                        signals['signal'][i] = -1
        i = i + 1
        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()
        i = 0
        for x in signals['positions']:
            if (signals.positions[i] == -1 and signals.signal[i + 1] == -1):
                signals.positions[i] = 1
            else:
                if (signals.positions[i] == 1 and signals.signal[i + 1] == 1):
                    signals.positions[i] = -1
            i = i + 1

        i = 0
        for x in range(len(signals['Input']) - 1):
            if (signals['Input'][x] > -1.83898 and signals['Input'][x] < 0):
                    signals['positions'][x] = -1
            if (signals['Input'][x] < 1.897873 and signals['Input'][x] > 0):
                signals['positions'][x] = 1


        return signals


