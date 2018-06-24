from random import *
from . import Application
from pandas.io.common import file_path_to_url
from tabulate import tabulate
import pandas as pd
import numpy as np

def fitness(individual,strategy):
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

    #inv_return.to_csv('without.csv', encoding='utf-8')


    #print(inv_return)

    ### Filter Regions ###
    inv_return = inv_return[(inv_return["signal"] == 1) | (inv_return["positions"] == -1)]
    #print("output inv_return_filter", inv_return)
    #inv_return.to_csv('filtered.csv', encoding='utf-8')
    #print(inv_return)

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
    #print(inv_return)

    returns = []
    total_profit = 0
    total_loss =0

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
            else:
                total_loss -= return_value


    total = []

    total = [sum(returns[0:x + 1]) for x in range(0, len(returns))]

    inv_return['returns'] = returns
    inv_return['total'] = total

    #print(inv_return.iloc[-1].tolist()[5])

    #inv_return.to_csv('finalresult.csv', encoding='utf-8')
    """from subprocess import Popen
    p = Popen('finalresult.csv', shell=True)"""
    #inv_return.to_csv('without2.csv', encoding='utf-8')
    #profit = inv_return.iloc[-1].tolist()[5]
    #print("profit : ",profit)

    max_drawdown = (inv_return['returns'].max())
    # print(max_drawdown)
    min_drawdown = (inv_return['returns'].min())
    # print(min_drawdown)

    if (max_drawdown != 0):
        risk_factor = abs((max_drawdown - min_drawdown) / max_drawdown)
    else:
        risk_factor = 0

    if (total_loss != 0):
        profit_factor = abs(total_profit / total_loss)
    else:
        profit_factor = 1

    fitness = profit_factor
    # fitness = risk_factor + profit_factor
    if (np.isnan(fitness)):
        fitness = 0

        # print("aaaaaaaaa")
        # print(max_drawdown,min_drawdown,risk_factor,total_profit,total_loss,profit_factor,individual)

    print("fitness", fitness)

    return fitness
#fitness([44, 140, 0, 200])