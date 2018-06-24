import pandas as pd
from pandas import to_datetime
import backtesting
from flask import Flask,redirect, url_for, request
from backtesting.backtester.Strategies.ma_cross.ma_cross import MovingAverageCrossStrategy
from backtesting.backtester.Strategies.ma_cross.fuzzy_ma_cross import FuzzyMovingAverageCrossStrategy
from backtesting.backtester.Strategies.BollingerBand.BollingerBand import BollingerBandStrategy
from backtesting.backtester.Strategies.BollingerBand.fuzzyBollingerBand import FuzzyBollingerBandStrategy
from backtesting.backtester.Strategies.MACD.fuzzyMACD  import FuzzyMACDStrategy
from backtesting.backtester.Strategies.Stochastic.Stochastic import StochasticStrategy
from backtesting.backtester.Strategies.Stochastic.fuzzy_stochastic import FuzzyStochasticStrategy
from backtesting.backtester.Strategies.MACD.MACD import MACDStrategy
from backtesting.backtester.Strategies.RSI.RSI import RSIStrategy
from backtesting.backtester.Strategies.RSI.fuzzyRSI  import FuzzyRSIStrategy
from backtesting.backtester.BackTestingResults.BackTestingResults import MarketOnClosePortfolio
from backtesting.backtester.plotCharts.PlotCharts import PlotChart
from backtesting.backtester.plotCharts.plotDistribution import plotDistribution

def optimize(individual,strategy):
   #print("short",short_window)
   #print("long",long_window)
   symbol = 'USD'

   strategyType = strategy
   startDate = request.form["from_date"]
   endDate = request.form["to_date"]

   bars = pd.read_csv("D:/Dilmi Computer Backup/A/Final Year Project/DataSetFinal/hour/hour.csv")
   bars.index = to_datetime(bars['Date'] + ' ' + bars['Time'])
   mask = (bars.index > startDate) & (bars.index <= endDate)
   bars = bars.loc[mask]


   if (strategyType == "Moving Average"):
      strategy = MovingAverageCrossStrategy(symbol, bars, individual[0], individual[1])
      signals = strategy.generate_signals()
   if (strategyType == "Fuzzy Moving Average"):
      print("inside fuzzyMA")
      print("individual",individual)
      strategy = FuzzyMovingAverageCrossStrategy(symbol, bars, individual[0], individual[1])
      signals = strategy.generate_signals()
      #print(signals)
   if (strategyType == "Bollinger Band"):
      strategy = BollingerBandStrategy(symbol, bars, individual[0], individual[1])
      signals = strategy.generate_signals()
   if (strategyType == "Fuzzy Bollinger Band"):
      strategy = FuzzyBollingerBandStrategy(symbol, bars, short_window=short, long_window=long)
      signals = strategy.generate_signals()
   if (strategyType == "MACD"):
      strategy = MACDStrategy(symbol, bars, individual[0], individual[1], individual[2])
      signals = strategy.generate_signals()
   if (strategyType == "Fuzzy MACD"):
      strategy = FuzzyMACDStrategy(symbol, bars, short_window=short, long_window=long)
      signals = strategy.generate_signals()
   if (strategyType == "Stochastic"):
      strategy = StochasticStrategy(symbol, bars,individual[0], individual[1], individual[2], individual[3])
      signals = strategy.generate_signals()
   if (strategyType == "Fuzzy Stochastic"):
      strategy = FuzzyStochasticStrategy(symbol, bars, short_window=short, long_window=long)
      signals = strategy.generate_signals()
   if (strategyType == "RSI"):
      strategy = RSIStrategy(symbol, bars, individual[0], individual[1])
      signals = strategy.generate_signals()
   if (strategyType == "Fuzzy RSI"):
      strategy = FuzzyRSIStrategy(symbol, bars, short_window=short, long_window=long)
      signals = strategy.generate_signals()

   portfolio = MarketOnClosePortfolio(symbol, bars, signals, initial_capital=100000.0)
   returns = portfolio.backtest_portfolio()
   #returns = returns.drop(returns.index[0])

   """with open('file.csv', 'w') as f:
      print(returns, file=f)"""

   #returns.to_csv('dil.csv', encoding='utf-8')

   """returns.to_csv('dil.csv', encoding='utf-8', index=False)"""

   #plot = PlotChart(signals, returns, strategyType)
   #plot.plotCharts()

   return returns




