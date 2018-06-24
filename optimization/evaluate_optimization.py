from random import *
from . import Application
from tabulate import tabulate
import pandas as pd
import numpy as np


def calculateRisk(individual,strategy):
    print("Calculaing Profit: ", individual)


    stoplossValue = 0.00000
    takeprofitValue = 0.00000
    stoploss_pip = 0.00000
    takeprofit_pip = 0.00000

    if (strategy == "Moving Average"):
        stoploss_pip = individual[2]
        takeprofit_pip = individual[3]

    elif (strategy == "Fuzzy Moving Average"):
        stoploss_pip = individual[2]
        takeprofit_pip = individual[3]

    elif (strategy == "Bollinger Band"):
        stoploss_pip = individual[2]
        takeprofit_pip = individual[3]

    elif (strategy == "MACD"):
        stoploss_pip = individual[3]
        takeprofit_pip = individual[4]

    elif (strategy == "Stochastic"):
        stoploss_pip = individual[4]
        takeprofit_pip = individual[5]

    elif (strategy == "RSI"):
        stoploss_pip = individual[2]
        takeprofit_pip = individual[3]


    inv_return = Application.optimize(individual,strategy)

    #print("inv_return:",inv_return)

    ### Filter Regions ###
    inv_return = inv_return[(inv_return["signal"] == 1) | (inv_return["positions"] == -1)]
    print("inv_return_filtered:",inv_return)

    print("   ")
    #print(inv_return)
    #inv_return.to_csv('without.csv', encoding='utf-8')

    #list = inv_return['close'].tolist()
    #se = pd.Series(list)
    #inv_return['stoploss'] = inv_return['close'] - stoploss_pip / 10000
    inv_return["group"] = (inv_return.positions == -1).shift(1).fillna(0).cumsum()


    stoploss = []
    takeprofit = []
    check = []
    group = -1
    for index, row in inv_return.iterrows():
        if(row['group'] == group):
            stoploss.append(stoplossValue)
            takeprofit.append(takeprofitValue)
            if (row['close'] < stoplossValue):
                check.append('stoploss')

            elif (row['close'] > takeprofitValue):
                check.append('takeprofit')

            else:
                check.append('null')

        else:
            group = row['group']
            stoplossValue = row['close'] - stoploss_pip / 10000
            takeprofitValue = row['close'] + takeprofit_pip / 10000
            stoploss.append(stoplossValue)
            takeprofit.append(takeprofitValue)
            check.append('null')


    inv_return['stoploss']  = stoploss
    inv_return['takeprofit'] = takeprofit
    inv_return['check'] = check


    returns = []
    total_profit = 0
    total_loss = 0


    for index, row in inv_return.iterrows():
        if(row['check'] == 'stoploss'):
            returns.append(0)

        elif (row['check'] == 'takeprofit'):
            returns.append(0)

        else:
            return_value = row['returns']
            returns.append(return_value)
            if (return_value>0):
                total_profit += return_value
            elif(return_value<0):
                total_loss -= return_value

    total = []

    total = [sum(returns[0:x + 1]) for x in range(0, len(returns))]

    inv_return['returns'] = returns
    inv_return['total'] = total

    print("inv_return_final:", inv_return)

    #print(inv_return.iloc[-1].tolist()[5])

    """inv_return.to_csv('finalresult.csv', encoding='utf-8')
    from subprocess import Popen
    p = Popen('finalresult.csv', shell=True)"""

    #inv_return.to_csv('without4.csv', encoding='utf-8')

    #profit = inv_return.iloc[-1].tolist()[5]
    #print("profit : ",profit)
    #print("total_profit",total_profit)
    #print("total_loss",total_loss)

        ##### Calculating #####

    ###    Total Profit    ###
    print("total_profit:", total_profit)


    print ("total_profit", total_profit)
    print ("total_loss", total_loss)
    ###   Profit Factor  ###
    if (total_loss != 0):
        profit_factor = abs(total_profit / total_loss)
    else:
        profit_factor = 0

    print("Profit factor:",profit_factor)


    ###   Total Return   ###
    total_return = total_profit - total_loss
    print("Total_Return:",total_return)

    ###   Profit trades percentage   ####
    #profit_trades_perc = (total_profit/total_return)*100
    #print("Profit_Trades_Percentage: ",profit_trades_perc)

    ###    Risk Factor ###
    max_drawdown = abs(inv_return['returns'].max())
    print(max_drawdown)
    min_drawdown = abs(inv_return['returns'].min())
    print(min_drawdown)

    if (max_drawdown != 0):
        risk_factor = abs(max_drawdown - min_drawdown) / max_drawdown
    else:
        risk_factor = 0

    print("Risk factor:   ", risk_factor)

    fitness = profit_factor
    #fitness = risk_factor + profit_factor
    if(np.isnan(fitness)):
        fitness = 0

        #print("aaaaaaaaa")
        #print(max_drawdown,min_drawdown,risk_factor,total_profit,total_loss,profit_factor,individual)

    #print("Profit Factor: ",profit_factor)

    matrix = [profit_factor, total_return, risk_factor]

    return matrix

#fitness([36, 165, 400, 900])




