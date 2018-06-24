import pandas as pd
import requests,  json
from flask import Flask,redirect, url_for, request
from pandas import to_datetime
import backtesting
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
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
fig = pyplot.figure()
ax1 = fig.add_subplot(111)


    # r = requests.get('http://127.0.0.1:5000/pricedata/20170726&20170727')
    # with open(r.json()) as train_file:
    #    dict_train = json.load(train_file)

    # converting json dataset from dictionary to dataframe
    # train = pd.DataFrame.from_dict(dict_train, orient='index')
    # train.reset_index(level=0, inplace=True)
    # print(train)

def refreshGraphData(self):
       line = open("D:/Dilmi Computer Backup/A/Final Year Project/Strategy_Optimizer_MA/GeneticTesting/Data/DataSets/EURUSD/Day_T.csv").read()
       csv_rows = line.split()
       print(csv_rows)
       xvalue = []
       yvalue = []
       for csv_row in csv_rows:
          x,y = csv_row.split(',')
          print("x value"+csv_row)
          xvalue.append(x)
          yvalue.append(y)
       ax1.clear()
       ax1.plot(xvalue,yvalue)
       # ani = animation.FuncAnimation(fig, refreshGraphData, interval=1000)
       pyplot.show()
def app(parameter,strategy):
    symbol = 'USD'
    bars = pd.read_csv("D:/Dilmi Computer Backup/A/Final Year Project/Strategy_Optimizer_MA/GeneticTesting/Data/DataSets/EURUSD/Day_T.csv")
    bars.index = to_datetime(bars ['Date'] +' ' + bars['Time'])
    if(strategy == "1"):
        strategyType = "Moving Average"
    elif (strategy == "2"):
        strategyType = "MACD"
    elif (strategy == "4"):
        strategyType = "Stochastic"
    elif (strategy == "5"):
        strategyType = "RSI"


    if(strategyType == "Moving Average" ):
        strategy = MovingAverageCrossStrategy(symbol, bars, parameter[0], parameter[1])
        signals = strategy.generate_signals()
    if(strategyType == "Fuzzy Moving Average"):
        strategy = FuzzyMovingAverageCrossStrategy(symbol, bars, short, long)
        signals = strategy.generate_signals()
    if(strategyType == "Bollinger Band"):
       strategy = BollingerBandStrategy(symbol, bars, MOV, std)
       signals = strategy.generate_signals()
    if(strategyType == "Fuzzy Bollinger Band"):
       strategy = FuzzyBollingerBandStrategy(symbol, bars, MOV, std)
       signals = strategy.generate_signals()
    if(strategyType == "MACD"):
       strategy = MACDStrategy(symbol, bars, parameter[0], parameter[1], parameter[2])
       signals = strategy.generate_signals()
    if (strategyType == "Fuzzy MACD"):
       strategy = FuzzyMACDStrategy(symbol, bars, short, long,signalLine )
       signals = strategy.generate_signals()
    if (strategyType == "Stochastic"):
       strategy = StochasticStrategy(symbol, bars,parameter[0], parameter[1], parameter[2],parameter[3])
       signals = strategy.generate_signals()
    if (strategyType == "Fuzzy Stochastic"):
       strategy = FuzzyStochasticStrategy(symbol, bars,K_period,D_period, higherLine, lowerLine)
       signals = strategy.generate_signals()
    if (strategyType == "RSI"):
       strategy = RSIStrategy(symbol, bars, parameter[0], parameter[1])
       signals = strategy.generate_signals()
    if (strategyType == "Fuzzy RSI"):
       strategy = FuzzyRSIStrategy(symbol, bars,rup, rdown)
       signals = strategy.generate_signals()

    portfolio = MarketOnClosePortfolio(symbol, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    """with open('file.csv', 'w') as f:
        print(returns, file=f)"""
    plot = PlotChart(signals, returns, strategyType)
    graph = plot.plotCharts()
    return graph
    # plot = plotDistribution(signals, returns,strategyType)
    # plot.distribution()

