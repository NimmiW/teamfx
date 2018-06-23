import pandas as pd
import requests,  json
from datetime import datetime
from flask import Flask,redirect, url_for, request
from pandas import to_datetime
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
import anomalies.config as config
import numpy as np
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

       line = open("D:/coursework/L4S2/GroupProject/repo/TeamFxPortal/backtesting/backtester/RealTimedata.csv").read()
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
def app(blackregion=False):
    symbol = 'USD'
    strategyType = request.form["strategy"]
    startDate = request.form["from_date"]
    endDate = request.form["to_date"]
    short=request.form["minShortMA"]
    long=request.form["minLongMA"]
    MOV = request.form["minMAPeriod"]
    std = request.form["minStdDev"]
    signalLine = request.form["minSignalLine"]
    shortMACD = request.form["minShortMAPeriod"]
    longMACD = request.form["minLongMAPeriod"]
    rup = request.form["rsiUp"]
    rdown = request.form["rsiDown"]
    K_period = request.form["K_period"]
    D_period = request.form["D_period"]
    higherLine = request.form["higher_line"]
    lowerLine = request.form["lower_line"]

    bars = pd.read_csv("E:/coursework/L4S2/GroupProject/repo/TeamFxPortal/backtesting/backtester/Minute.csv")
    bars.index = to_datetime(bars['Date'] + ' ' + bars['Time'])

    if (blackregion == True):
        threshold = request.form["threshold"]
        black_regions = pd.read_csv('E:/coursework/L4S2/GroupProject/repo/TeamFxPortal/static/anomalies/detected_black_regions/'+threshold+'_'+str(config.NEAREST_NEIGHBOURS)+'_EURUSD_all_anomalies.csv')
        print("black regions")
        black_regions = black_regions["DateHour"]

        # for hour in black_regions:
        #     barsTwo = np.where(bars.index > black_regions.index and bars.index<black_regions.index,bars , 0.0)





    bars.index = to_datetime(bars ['Date'] +' ' + bars['Time'])
    mask = (bars.index > startDate) & (bars.index <= endDate)
    bars = bars.loc[mask]
    # series = data["Close"]

    if(strategyType == "Moving Average" ):
     strategy = MovingAverageCrossStrategy(symbol, bars,  short, long)
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
       strategy = MACDStrategy(symbol, bars, shortMACD, longMACD,signalLine )
       signals = strategy.generate_signals()
    if (strategyType == "Fuzzy MACD"):
       strategy = FuzzyMACDStrategy(symbol, bars, shortMACD, longMACD,signalLine )
       signals = strategy.generate_signals()
    if (strategyType == "Stochastic"):
       strategy = StochasticStrategy(symbol, bars,K_period,D_period, higherLine, lowerLine)
       signals = strategy.generate_signals()
    if (strategyType == "Fuzzy Stochastic"):
       strategy = FuzzyStochasticStrategy(symbol, bars,K_period,D_period, higherLine, lowerLine)
       signals = strategy.generate_signals()
    if (strategyType == "RSI"):
       strategy = RSIStrategy(symbol, bars, rup, rdown)
       signals = strategy.generate_signals()
    if (strategyType == "Fuzzy RSI"):
       strategy = FuzzyRSIStrategy(symbol, bars,rup, rdown)
       signals = strategy.generate_signals()

    portfolio = MarketOnClosePortfolio(symbol, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    with open('file.csv', 'w') as f:
        print(returns, file=f)
    plot = PlotChart(signals, returns, strategyType)
    graph = plot.plotCharts()
    return graph
    # plot = plotDistribution(signals, returns,strategyType)
    # plot.distribution()

