import numpy as np
import pandas as pd
from backtesting.backtester.BackTest.backtest import Strategy, Portfolio
import backtesting.backtester.fuzzySystem.membership as fuzz
import backtesting.backtester.fuzzySystem.control as ctrl
from subprocess import Popen
class FuzzyMACDStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""
    def __init__(self, symbol, bars, short_window, long_window, signalLine):
        self.symbol = symbol
        self.bars = bars
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.signalLine = int(signalLine)
    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0
        signals['positions'] = 0.0
        signals['close'] = self.bars['Close']
        # Create the set of short and long simple moving averages over the
        # respective periods
        signals['short_ema'] = pd.ewma(self.bars['Close'], span=self.short_window)
        signals['long_ema'] = pd.ewma(self.bars['Close'], span=self.long_window)
        signals['MACD']= signals['short_ema']- signals['long_ema']
        signals['signalLine'] = pd.ewma(signals['MACD'],span= self.signalLine)
        signals['fuzzyInput'] =  (signals['signalLine'] - signals['MACD']/signals['signalLine']  )
        signals['Input'] = signals['signalLine'] - signals['MACD']/signals['signalLine']
        print(signals['fuzzyInput'])
        # p = Popen('fuzzyInput.csv', shell=True)
        # with open('fuzzyInput.csv', 'w') as f:
        #     print(signals['fuzzyInput'], file=f)

        fuzzythresholdOne =  0.806191
        fuzzythreshold = 0.87218

        normalizedInput= ctrl.Antecedent(np.arange(-1*fuzzythreshold, fuzzythresholdOne, 0.00001), 'normalizedInput')
        fuzzyOutput = ctrl.Consequent(np.arange(-1,1,0.001), 'fuzzyOutput')
        normalizedInput['high'] = fuzz.trimf(normalizedInput.universe, [0,0,fuzzythresholdOne])
        normalizedInput['low'] = fuzz.trimf(normalizedInput.universe, [-1*fuzzythreshold, 0,0] )
        normalizedInput['medium1'] = fuzz.trimf(normalizedInput.universe, [-1*fuzzythreshold,-1*fuzzythreshold,0])
        normalizedInput['medium2'] = fuzz.trimf(normalizedInput.universe, [0,fuzzythresholdOne,fuzzythresholdOne])

        fuzzyOutput['low'] = fuzz.trimf(fuzzyOutput.universe, [-1,-0.075 ,-0.025])
        fuzzyOutput['high'] = fuzz.trimf(fuzzyOutput.universe, [0.025,0.075,1])
        fuzzyOutput['medium'] = fuzz.trimf(fuzzyOutput.universe,[-0.025, 0,0.025])

        rule1 = ctrl.Rule(normalizedInput['low'] , fuzzyOutput['low'])
        rule2 = ctrl.Rule(normalizedInput['medium1'] , fuzzyOutput['medium'])
        rule3 = ctrl.Rule(normalizedInput['high'], fuzzyOutput['high'])
        rule4 = ctrl.Rule(normalizedInput['medium2'] , fuzzyOutput['medium'])

        MACD_ctrl = ctrl.ControlSystem([rule1, rule3, rule2, rule4])

        MACD = ctrl.ControlSystemSimulation(MACD_ctrl)
        i =0
        for x in signals['fuzzyInput']:
             MACD.input['normalizedInput'] = round(x,5)
             MACD.compute()
             # print("fuzzyOutput", MACD.output['fuzzyOutput'])
             if MACD.output['fuzzyOutput']>= -1 and MACD.output['fuzzyOutput']<= -0.025   :

               signals['signal'][i]= 1

             else:
               if MACD.output['fuzzyOutput']> -0.025 and MACD.output['fuzzyOutput']<0.025:
                signals['signal'][i]= 0

               else:
                if MACD.output['fuzzyOutput']>= 0.025 and MACD.output['fuzzyOutput']<= 1 :
                 signals['signal'][i]=-1

             i = i+1
        # Take the difference of the signals in order to generate actual trading orders
        signals['positions']=signals['signal'].diff()
        i = 0
        for x in range(len(signals['positions'])-1):
            if (signals.positions[i] == -1 and signals.signal[i + 1] == -1):
                signals.positions[i] = 1
            else:
                if (signals.positions[i] == 1 and signals.signal[i + 1] == 1):
                    signals.positions[i] = -1
            i = i + 1


        # for x in range(len(signals['Input']) - 1):
        #     if (signals['Input'][x] > -0.03898 and signals['Input'][x] < 0):
        #         signals['positions'][x] = -1
        #     if (signals['Input'][x] < 0.047873 and signals['Input'][x] > 0):
        #         signals['positions'][x] = 1

        for x in range(len(signals['Input']) - 1):
            if (signals['Input'][x] > -0.03898 and signals['Input'][x] < 0):
                signals['positions'][x] = -1
            if (signals['Input'][x] < 0.047873 and signals['Input'][x] > 0):
                signals['positions'][x] = 1
