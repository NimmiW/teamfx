# import matplotlib.pyplot as plt
# import plotly.plotly as py
# import plotly.graph_objs as go
from flask import request
import plotly.plotly.plotly as py
import json
import numpy as np

class PlotChart:
  def __init__(self,signals,returns, strategyType):
      self.signals = signals
      self.returns = returns
      self.strategy = strategyType
  def plotCharts(self):
      graphs = []
      if(self.strategy == "Bollinger Band" or self.strategy == "Fuzzy Bollinger Band"):
          buysignals = np.where(self.signals.positions == 1.0, self.signals.middlBand, '')
          sellsignals = np.where(self.signals.positions == -1.0, self.signals.middlBand, '')
          df = self.signals
          graphs = [
              dict(
                  data=[
                      dict(
                          x=df.index,
                          y=df.middlBand,
                          type='scatter'
                      ),
                      dict(
                          x=df.index,
                          y=df.upperBand,
                          type='scatter'
                      ),
                      dict(
                          x=df.index,
                          y=df.lowerBand,
                          type='scatter'
                      ),
                      dict(
                          x=df.index,
                          y=buysignals,
                          mode="markers",
                          type='plot'
                      ),
                      dict(
                          x=df.index,
                          y=sellsignals,
                          mode="markers",
                          type='plot'
                      )
                  ],
                  layout=dict(
                      title='Backtesting Results',
                  )
              )
          ]
      if(self.strategy == "Moving Average" or self.strategy =="Fuzzy Moving Average"):
          buysignals = np.where(self.signals.positions == 1.0, self.signals.short_mavg, '')
          sellsignals = np.where(self.signals.positions == -1.0, self.signals.short_mavg, '')
          df = self.signals
          graphs = [
              dict(
                  data=[
                      dict(
                          x=df.index,
                          y=df.short_mavg,
                          type='scatter'
                      ),
                      dict(
                          x=df.index,
                          y=df.long_mavg,
                          type='scatter'
                      ),
                      dict(
                          x=df.index,
                          y=buysignals,
                          mode="markers",
                          type='plot'
                      ),
                      dict(
                          x=df.index,
                          y=sellsignals,
                          mode="markers",
                          type='plot'
                      )
                  ],
                  layout=dict(
                      title='Backtesting Results',
                  )
              )
          ]

      if(self.strategy == "MACD" or self.strategy =="Fuzzy MACD" ):
          buysignals = np.where(self.signals.positions == 1.0, self.signals.signalLine, '')
          sellsignals = np.where(self.signals.positions == -1.0, self.signals.signalLine, '')
          df = self.signals
          graphs = [
          dict(
              data=[
                  dict(
                      x=df.index,
                      y=df.signalLine,
                      type='scatter'
                  ),
                  dict(
                      x=df.index,
                      y=df.MACD,
                      type='scatter'
                  ),
                  dict(
                      x=df.index,
                      y=buysignals,
                      mode="markers",
                      type='plot'
                  ),
                  dict(
                      x=df.index,
                      y=sellsignals,
                      mode="markers",
                      type='plot'
                  )
              ],
              layout=dict(
                  title='Backtesting Results',
              )
          )
      ]

      if (self.strategy == "Stochastic" or self.strategy == "Fuzzy Stochastic"):
          buysignals = np.where(self.signals.positions == 1.0, self.signals.K, '')
          sellsignals = np.where(self.signals.positions == -1.0, self.signals.D, '')
          df = self.signals
          graphs = [
              dict(
                  data=[
                      dict(
                          x=df.index,
                          y=df.K,
                          type='scatter'
                      ),
                      dict(
                          x=df.index,
                          y=df.D,
                          type='scatter'
                      ),
                      dict(
                          x=df.index,
                          y=buysignals,
                          mode="markers",
                          type='plot'
                      ),
                      dict(
                          x=df.index,
                          y=sellsignals,
                          mode="markers",
                          type='plot'
                      )
                  ],
                  layout=dict(
                      title='Backtesting Results',
                  )
              )
          ]

      if (self.strategy == "RSI" or self.strategy == "Fuzzy RSI"):
          buysignals = np.where(self.signals.positions == 1.0, self.signals.RSI, '')
          sellsignals = np.where(self.signals.positions == -1.0, self.signals.RSI, '')
          df = self.signals
          graphs = [
              dict(
                  data=[
                      dict(
                          x=df.index,
                          y=df.RSI,
                          type='scatter'
                      ),
                      dict(
                          x=df.index,
                          y=buysignals,
                          mode="markers",
                          type='plot'
                      ),
                      dict(
                          x=df.index,
                          y=sellsignals,
                          mode="markers",
                          type='plot'
                      )
                  ],
                  layout=dict(
                      title='Backtesting Results',
                  )
              )
          ]
      # Plot the "buy" and "sell" trades against the equity curve
      # ax2.plot(self.returns.ix[self.signals.positions == 1.0].index, self.returns.total[self.signals.positions == 1.0], '^', markersize=10,
      #          color='m')
      # ax2.plot(self.returns.ix[self.signals.positions == -1.0].index, self.returns.total[self.signals.positions == -1.0], 'v', markersize=10,
      #          color='k')
      # # Plot the figure
      # fig.savefig('fig1.pdf')
      # plt.show()


      # Add "ids" to each of the graphs to pass up to the client
      # for templating
      ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

      # Convert the figures to JSON
      # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
      # objects to their JSON equivalents
      graphJSON = json.dumps(graphs, cls=py.utils.PlotlyJSONEncoder)

      return ids, graphJSON