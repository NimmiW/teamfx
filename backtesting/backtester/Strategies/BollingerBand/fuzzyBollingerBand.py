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
class FuzzyBollingerBandStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""
    def __init__(self, symbol, bars, MOV,std):
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
        signals['fuzzyInputUpperBand'] = ((signals['close'] - signals['upperBand'])/signals['close'] )
        signals['fuzzyInputlowerBand'] = ((signals['close'] - signals['lowerBand'])/signals['close'] )
        # print('fuzzy Input')
        # print(signals['fuzzyInputUpperBand'])
        # print(signals['fuzzyInputlowerBand'])
        fuzzythreshold = 0.02
        fuzzythresholdOne = 0.03
        normalizedInputOne = ctrl.Antecedent(np.arange(-1*fuzzythreshold, fuzzythreshold, 0.0000001), 'normalizedInputOne')
        normalizedInputTwo = ctrl.Antecedent(np.arange(-1*fuzzythreshold, fuzzythreshold, 0.0000001), 'normalizedInputTwo')
        fuzzyOutput = ctrl.Consequent(np.arange(-1,1,0.001), 'fuzzyOutput')

        normalizedInputOne['high'] = fuzz.trimf(normalizedInputOne.universe, [-1*fuzzythreshold, 0,0])
        normalizedInputOne['high'] = fuzz.trimf(normalizedInputOne.universe, [0,0,fuzzythreshold])
        normalizedInputOne['medium1'] = fuzz.trimf(normalizedInputOne.universe, [-1*fuzzythreshold,-1*fuzzythreshold,0])
        normalizedInputOne['medium2'] = fuzz.trimf(normalizedInputOne.universe, [0,fuzzythreshold,fuzzythreshold])

        normalizedInputTwo['low'] = fuzz.trimf(normalizedInputTwo.universe, [-1*fuzzythresholdOne, 0,0])
        normalizedInputTwo['low'] = fuzz.trimf(normalizedInputTwo.universe, [0,0,fuzzythresholdOne])
        normalizedInputTwo['medium1'] = fuzz.trimf(normalizedInputTwo.universe, [-1*fuzzythresholdOne,-1*fuzzythresholdOne,0])
        normalizedInputTwo['medium2'] = fuzz.trimf(normalizedInputTwo.universe, [0,fuzzythresholdOne,fuzzythresholdOne])


        fuzzyOutput['low'] = fuzz.trimf(fuzzyOutput.universe, [-1,-0.075 ,-0.025])
        fuzzyOutput['medium'] = fuzz.trimf(fuzzyOutput.universe, [0.025,0.075,1])
        fuzzyOutput['high'] = fuzz.trimf(fuzzyOutput.universe, [-0.025, 0,0.025])

        rule3 = ctrl.Rule(normalizedInputOne['high'], fuzzyOutput['high'])
        # rule1 = ctrl.Rule(normalizedInputOne['high'], fuzzyOutput['high'])
        rule2 = ctrl.Rule(normalizedInputOne['medium1'], fuzzyOutput['medium'])
        rule4 = ctrl.Rule(normalizedInputOne['medium2'], fuzzyOutput['medium'])


        rule7 = ctrl.Rule(normalizedInputTwo['low'], fuzzyOutput['low'])
        # rule6 = ctrl.Rule(normalizedInputTwo['low'], fuzzyOutput['low'])
        rule5 = ctrl.Rule(normalizedInputTwo['medium1'], fuzzyOutput['medium'])
        rule8 = ctrl.Rule(normalizedInputTwo['medium2'], fuzzyOutput['medium'])


        movingAverage_ctrl_one = ctrl.ControlSystem([rule3, rule2, rule4])
        movingAverageCrossOver = ctrl.ControlSystemSimulation(movingAverage_ctrl_one)

        movingAverage_ctrl_two = ctrl.ControlSystem([rule7, rule5, rule8])
        movingAverageCrossOverTwo = ctrl.ControlSystemSimulation(movingAverage_ctrl_two)
        print('fuzzyInputlowerBand')
        print(signals['fuzzyInputlowerBand'])


        for x in range(len(signals['fuzzyInputUpperBand'])-1):
            if(signals['fuzzyInputUpperBand'][x]>-0.00023 and signals['fuzzyInputUpperBand'][x]<0.00023):
                if (signals['close'][x] >signals['middlBand'][x]):
                  signals['positions'][x] = -1
                  print(signals['fuzzyInputUpperBand'][x])

        for x in range(len(signals['fuzzyInputlowerBand']) - 1):
            if (signals['fuzzyInputlowerBand'][x] >-0.0023 and signals['fuzzyInputlowerBand'][x] <0.00023):
                if (signals['close'][x] < signals['middlBand'][x]):
                 signals['positions'][x] = 1
                 print(signals['fuzzyInputlowerBand'][x])

        i = 0
        # for x in signals['close']:
        #   print("close" , x)
        #   print("middle" , signals['middlBand'][i])
        #
        #   if (x> signals['middlBand'][i]):
        #     print('Inside if Zero')
        #     movingAverageCrossOver.input['normalizedInputOne'] = round(x, 5)
        #     movingAverageCrossOver.compute()
        #     if movingAverageCrossOver.output['fuzzyOutput'] >= -1 and movingAverageCrossOver.output['fuzzyOutput'] <= -0.025:
        #         signals['signal'][i] = -1
        #         print("Inside if one")
        #
        #     else:
        #         if movingAverageCrossOver.output['fuzzyOutput'] > -0.025 and movingAverageCrossOver.output['fuzzyOutput'] < 0.025:
        #             signals['signal'][i] = 0
        #             print("Inside if two")
        #
        #         else:
        #             if movingAverageCrossOver.output['fuzzyOutput'] >= 0.000025 and movingAverageCrossOver.output['fuzzyOutput'] <= 1:
        #                 signals['signal'][i] = -1
        #                 print("Inside if three")
        #   else:
        #       signals['signal'][i] = 0.0
        #   i=i+1
        #
        # i = 0
        # for x in signals['close']:
        #   if (x< signals['middlBand'][i]):
        #     movingAverageCrossOverTwo.input['normalizedInputTwo'] = round(x, 5)
        #     movingAverageCrossOverTwo.compute()
        #     if movingAverageCrossOverTwo.output['fuzzyOutput'] >= -1 and movingAverageCrossOverTwo.output['fuzzyOutput'] <= -0.0025:
        #         signals['signal'][i] = 1
        #         print("Inside if four")
        #
        #     else:
        #         if movingAverageCrossOverTwo.output['fuzzyOutput'] > -0.025 and movingAverageCrossOverTwo.output['fuzzyOutput'] < 0.025:
        #             signals['signal'][i] = 0
        #             print("Inside if five")
        #         else:
        #             if movingAverageCrossOverTwo.output['fuzzyOutput'] >= 0.000025 and movingAverageCrossOverTwo.output['fuzzyOutput'] <= 1:
        #                 signals['signal'][i] = 1
        #                 print("Inside if six")
        #     i = i + 1
        # signals['positions'] = signals['signal']
        #
        # for x in range(len(signals.index)-1):
        #     if(signals.positions[x]== -1 and signals.signal[x+1]==-1):
        #         signals.positions[x] = 1
        #     else:
        #         if(signals.positions[x]==1 and signals.signal[x+1]==1):
        #          signals.positions[x]= -1
        # i =0
        # for x in signals['close']:
        #     if (x < signals['middlBand'][i]):
        #         if(signals['fuzzyInputUpperBand'][i] <= 0.00001 and signals['fuzzyInputUpperBand'][i] >= -0.00001 ):
        #             signals['signal'][i]= 1
        #         else :
        #             if(signals['fuzzyInputlowerBand'][i] <= 0.00001 and signals['fuzzyInputlowerBand'][i] >= 0.00001):
        #                 signals['signal'][i] = -1

        signals['signal'] = signals['positions']
        print("signal")
        print(signals)
        return signals
