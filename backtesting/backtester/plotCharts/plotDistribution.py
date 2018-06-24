import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class plotDistribution:
    def __init__(self, signals, returns, strategyType):
        self.signals = signals
        self.returns = returns
        self.strategy = strategyType

    def distribution(self):
        print("Total count",self.signals['fuzzyInput'].count())
        print(self.signals['fuzzyInput'])
        fig, ax = plt.subplots()
        posCount = 0
        negCount = 0
        d = {'Date': [0.0], 'value': [0.0]}
        possitiveValue = pd.DataFrame(data=d)
        negativeValue = pd.DataFrame(data=d)
        possitiveValue['value']= 0.0
        negativeValue['value'] = 0.0
        for x in self.signals['fuzzyInput']:
            if x == 0.0:
             possitiveValue['value'][posCount] = x
             negativeValue['value'][negCount] = x
             posCount = posCount+1
             negCount = negCount+1
            else:
                if x >0.0:
                    possitiveValue['value'][posCount] = x
                    posCount = posCount + 1
                else:
                    negativeValue['value'][negCount] = x
                    negCount = negCount + 1

        print("minimum value", possitiveValue['value'].min())
        print("maximum value", possitiveValue['value'].max())
        print("count", possitiveValue['value'].count())
        print("mean", np.mean(possitiveValue['value']))
        print("std", np.std(possitiveValue['value']))

        print("minimum value", negativeValue['value'].min())
        print("maximum value", negativeValue['value'].max())
        print("count", negativeValue['value'].count())
        print("mean", np.mean(negativeValue['value']))
        print("std", np.std(negativeValue['value']))

        plt.hist(possitiveValue['value'], bins=np.arange(possitiveValue['value'].min(), possitiveValue['value'].max() + 0.0001))
        # fig = plt.figure()
        # fig.patch.set_facecolor('white')
        # Set the outer colour to white
        # ax1 = fig.add_subplot(211, ylabel='Price in $')
        # if(self.strategy == "Fuzzy Bollinger Band"):
        #     self.signals[['fuzzyInputUpperBand', 'fuzzyInputlowerBand']].plot(ax=ax1, lw=2.)
        # else:
        # self.signals[['fuzzyInput']].plot(ax=ax1, lw=2.)
        # ax1.plot(self.signals.ix[self.signals.positions == 1.0].index,
        #          self.signals.fuzzyInput[self.signals.positions == 1.0], '^', markersize=10,
        #          color='m')
        # ax1.plot(self.signals.ix[self.signals.positions == -1.0].index,
        #          self.signals.fuzzyInput[self.signals.positions == -1.0], 'v', markersize=10,
        #          color='k')
        # fig.savefig('figDistribution.pdf')
        plt.show()
