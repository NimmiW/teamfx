import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtesting.backtester.BackTest.backtest import Strategy, Portfolio
import backtesting.backtester.fuzzySystem.membership as fuzz
import backtesting.backtester.fuzzySystem.control as ctrl
from pandas import to_datetime
class FuzzyMovingAverageCrossStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""
    def __init__(self, symbol, bars, short_window, long_window):
        self.symbol = symbol
        self.bars = bars
        self.short_window = int(short_window)
        self.long_window = int(long_window)
    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['close'] = self.bars['Close']
        signals['signal'] = 0.0
        # Create the set of short and long simple moving averages over the
        # respective periods
        signals['short_mavg'] = pd.rolling_mean(self.bars['Close'], self.short_window, min_periods=1)
        signals['long_mavg'] = pd.rolling_mean(self.bars['Close'], self.long_window, min_periods=1)
        signals['fuzzyInput'] = 100*((signals['short_mavg']- signals['long_mavg'])/signals['short_mavg'])
        # print(signals['fuzzyInput'])

        # fuzzyThreshold = 0.0125
        fuzzyThreshold = 0.125
        normalizedInput= ctrl.Antecedent(np.arange(-1*fuzzyThreshold, fuzzyThreshold, 0.00001), 'normalizedInput')
        fuzzyOutput = ctrl.Consequent(np.arange(-1,1,0.001), 'fuzzyOutput')

        normalizedInput['high'] = fuzz.trimf(normalizedInput.universe, [0,0,fuzzyThreshold])
        normalizedInput['low'] = fuzz.trimf(normalizedInput.universe, [-1*fuzzyThreshold, 0,0] )
        normalizedInput['medium1'] = fuzz.trimf(normalizedInput.universe, [-1*fuzzyThreshold,-1*fuzzyThreshold,0])
        normalizedInput['medium2'] = fuzz.trimf(normalizedInput.universe, [0,fuzzyThreshold,fuzzyThreshold])

        fuzzyOutput['low'] = fuzz.trimf(fuzzyOutput.universe, [-1,-0.075 ,-0.025])
        fuzzyOutput['high'] = fuzz.trimf(fuzzyOutput.universe, [0.025,0.075,1])
        fuzzyOutput['medium'] = fuzz.trimf(fuzzyOutput.universe,[-0.025, 0,0.025])

        rule1 = ctrl.Rule(normalizedInput['low'] , fuzzyOutput['low'])
        rule2 = ctrl.Rule(normalizedInput['medium1'] , fuzzyOutput['medium'])
        rule3 = ctrl.Rule(normalizedInput['high'], fuzzyOutput['high'])
        rule4 = ctrl.Rule(normalizedInput['medium2'] , fuzzyOutput['medium'])

        movingAverage_ctrl = ctrl.ControlSystem([rule1, rule3, rule2, rule4])

        movingAverageCrossOver = ctrl.ControlSystemSimulation(movingAverage_ctrl)
        i =0;
        for x in signals['fuzzyInput']:
             movingAverageCrossOver.input['normalizedInput'] = round(x,5)
             movingAverageCrossOver.compute()

             if movingAverageCrossOver.output['fuzzyOutput']>= -1 and movingAverageCrossOver.output['fuzzyOutput']<= -0.025 :
               signals['signal'][i]= 1
             else:
               if movingAverageCrossOver.output['fuzzyOutput']> -0.025 and movingAverageCrossOver.output['fuzzyOutput']<0.025:
                signals['signal'][i]= 0
               else:
                if movingAverageCrossOver.output['fuzzyOutput']>= 0.025 and movingAverageCrossOver.output['fuzzyOutput']<= 1:
                 signals['signal'][i]=-1
             i = i+1
        # Take the difference of the signals in order to generate actual trading orders
        signals['positions']=signals['signal'].diff()
        # print(signals['positions'])
        i = 0
        for x in range(len(signals['positions'])-1):
            if(signals.positions[i]== -1 and signals.signal[i+1]==-1):
                signals.positions[i] = 1
            else:
                if(signals.positions[i]==1 and signals.signal[i+1]==1):
                 signals.positions[i]= -1
            i = i+1

        # print(signals)
        return signals
