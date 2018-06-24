import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
from backtesting.backtester.BackTest.backtest import Strategy, Portfolio
import backtesting.backtester.fuzzySystem.membership as fuzz
import backtesting.backtester.fuzzySystem.control as ctrl
import time
from pandas import to_datetime
class FuzzyRSIStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""
    def __init__(self, symbol, bars,rup, rdown):
        self.symbol = symbol
        self.bars = bars
        self.rup = int(rup)
        self.rdown = int(rdown)
    def generate_signals(self):
        print("inside fuzzy RSI generate signals")
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0
        signals['close'] = self.bars['Close']
        # Create the set of short and long simple moving averages over the
        # respective periods
        delta = signals['close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        signals['rUp'] = up.ewm(com= self.rup  - 1, adjust=False).mean()
        signals['rDown'] = down.ewm(com= self.rdown- 1, adjust=False).mean().abs()
        signals['RSI'] = 100 - 100 / (1 + signals['rUp'] / signals['rDown']) #
        signals['fuzzyInputOne'] = 100 * (signals['RSI']-70)/signals['RSI']
        signals['fuzzyInputTwo'] = 100 * (signals['RSI']-30)/signals['RSI']
        signals['Input0ne'] = signals['RSI']-70
        signals['InputTwo'] = signals['RSI']-30


        fuzzythreshold = 0.42
        normalizedInputOne = ctrl.Antecedent(np.arange(-1 * fuzzythreshold, fuzzythreshold, 0.00001),'normalizedInputOne')
        normalizedInputTwo = ctrl.Antecedent(np.arange(-1 * fuzzythreshold, fuzzythreshold, 0.00001),'normalizedInputTwo')
        fuzzyOutput = ctrl.Consequent(np.arange(-1, 1, 0.001), 'fuzzyOutput')

        normalizedInputOne['low'] = fuzz.trimf(normalizedInputOne.universe, [-1 * fuzzythreshold, 0, 0])
        normalizedInputOne['high'] = fuzz.trimf(normalizedInputOne.universe, [0, 0, fuzzythreshold])
        normalizedInputOne['medium1'] = fuzz.trimf(normalizedInputOne.universe,[-1 * fuzzythreshold, -1 * fuzzythreshold, 0])
        normalizedInputOne['medium2'] = fuzz.trimf(normalizedInputOne.universe, [0, fuzzythreshold, fuzzythreshold])

        normalizedInputTwo['low'] = fuzz.trimf(normalizedInputTwo.universe, [-1 * fuzzythreshold, 0, 0])
        normalizedInputTwo['high'] = fuzz.trimf(normalizedInputTwo.universe, [0, 0, fuzzythreshold])
        normalizedInputTwo['medium1'] = fuzz.trimf(normalizedInputTwo.universe,[-1 * fuzzythreshold, -1 * fuzzythreshold, 0])
        normalizedInputTwo['medium2'] = fuzz.trimf(normalizedInputTwo.universe, [0, fuzzythreshold, fuzzythreshold])

        fuzzyOutput['low'] = fuzz.trimf(fuzzyOutput.universe, [-1, -0.075, -0.025])
        fuzzyOutput['medium'] = fuzz.trimf(fuzzyOutput.universe, [0.025, 0.075, 1])
        fuzzyOutput['high'] = fuzz.trimf(fuzzyOutput.universe, [-0.025, 0, 0.025])

        rule3 = ctrl.Rule(normalizedInputOne['low'], fuzzyOutput['high'])
        rule1 = ctrl.Rule(normalizedInputOne['high'], fuzzyOutput['high'])
        rule2 = ctrl.Rule(normalizedInputOne['medium1'], fuzzyOutput['medium'])
        rule4 = ctrl.Rule(normalizedInputOne['medium2'], fuzzyOutput['medium'])

        rule7 = ctrl.Rule(normalizedInputTwo['low'], fuzzyOutput['low'])
        rule6 = ctrl.Rule(normalizedInputTwo['high'], fuzzyOutput['low'])
        rule5 = ctrl.Rule(normalizedInputTwo['medium1'], fuzzyOutput['medium'])
        rule8 = ctrl.Rule(normalizedInputTwo['medium2'], fuzzyOutput['medium'])

        movingAverage_ctrl_one = ctrl.ControlSystem([rule3, rule1, rule2, rule4])
        movingAverageCrossOver = ctrl.ControlSystemSimulation(movingAverage_ctrl_one)

        movingAverage_ctrl_two = ctrl.ControlSystem([rule6, rule7, rule5, rule8])
        movingAverageCrossOverTwo = ctrl.ControlSystemSimulation(movingAverage_ctrl_two)

        i = 0
        for x in signals['fuzzyInputOne']:
            movingAverageCrossOver.input['normalizedInputOne'] = round(x, 5)
            movingAverageCrossOver.compute()
            if movingAverageCrossOver.output['fuzzyOutput'] >= -1 and movingAverageCrossOver.output[
                'fuzzyOutput'] <= -0.025:
                signals['signal'][i] = 1
                # print("inside if 1")
            else:
                if movingAverageCrossOver.output['fuzzyOutput'] > -0.025 and movingAverageCrossOver.output[
                    'fuzzyOutput'] < 0.025:
                    signals['signal'][i] = 0
                    # print("inside if 2")
                else:
                    if movingAverageCrossOver.output['fuzzyOutput'] >= 0.025 and movingAverageCrossOver.output[
                        'fuzzyOutput'] <= 1:
                        signals['signal'][i] = -1
                        # print("inside if 3")
            i = i + 1
        i = 0
        for x in signals['fuzzyInputTwo']:
            movingAverageCrossOverTwo.input['normalizedInputTwo'] = round(x, 5)
            movingAverageCrossOverTwo.compute()
            if movingAverageCrossOverTwo.output['fuzzyOutput'] >= -1 and movingAverageCrossOverTwo.output['fuzzyOutput'] <= -0.025:
                signals['signal'][i] = 1
                # print("inside if 4")
            else:
                if movingAverageCrossOverTwo.output['fuzzyOutput'] > -0.025 and movingAverageCrossOverTwo.output['fuzzyOutput'] < 0.025:
                    signals['signal'][i] = 0
                    # print("inside if 5")
                else:
                    if movingAverageCrossOverTwo.output['fuzzyOutput'] >= 0.025 and movingAverageCrossOverTwo.output['fuzzyOutput'] <= 1:
                        signals['signal'][i] = -1
                        # print("inside if 6")
            i = i + 1
        i = 0
        signals['positions'] = signals['signal'].diff()
        for x in signals['positions']:
            if (signals.positions[i] == -1 and signals.signal[i + 1] == -1):
                signals.positions[i] = 1
            else:
                if (signals.positions[i] == 1 and signals.signal[i + 1] == 1):
                    signals.positions[i] = -1
            i = i + 1
        # print(signals['positions'])
        for x in range(len(signals['Input0ne']) - 1):
            if (signals['Input0ne'][x] > -1.83898 and signals['Input0ne'][x] < 0):
                signals['positions'][x] = 1
            if (signals['Input0ne'][x] < 1.897873 and signals['Input0ne'][x] > 0):
                signals['positions'][x] = 1

        for x in range(len(signals['InputTwo']) - 1):
            if (signals['InputTwo'][x] > -1.83898 and signals['InputTwo'][x] < 0):
                signals['positions'][x] = - 1
            if (signals['InputTwo'][x] < 1.897873 and signals['InputTwo'][x] > 0):
                signals['positions'][x] = -1


        return signals
