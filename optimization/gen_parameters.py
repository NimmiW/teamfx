from flask import Flask,redirect, url_for, request

def getPara(strategy):
    para = []
    if(strategy == "Moving Average"):
        para = [int(request.form['ShortMA']),int(request.form['LongMA']),int(request.form['StopLoss']),int(request.form['TakeProfit'])]

    elif (strategy == "Fuzzy Moving Average"):
        para = [int(request.form['fShortMA']), int(request.form['fLongMA']), int(request.form['fStopLoss']), int(request.form['fTakeProfit'])]


    elif (strategy == "MACD"):
        para = [int(request.form['ShortMAPeriod']), int(request.form['LongMAPeriod']), int(request.form['SignalLine']), int(request.form['StopLossMacd']),
                int(request.form['TakeProfitMacd'])]

    elif (strategy == "Bollinger Band"):
        para = [int(request.form['MAPeriodBol']), int(request.form['StdDevBol']), int(request.form['StopLossBol']), int(request.form['TakeProfitBol'])]

    elif (strategy == "Stochastic"):
        para = [int(request.form['K_periods']), int(request.form['D_periods']), int(request.form['higher_lines']), int(request.form['lower_lines']),int(request.form['StopLosss']),int(request.form['TakeProfits'])]

    elif (strategy == "RSI"):
        para = [int(request.form['higher_line_rsi']), int(request.form['lower_line_rsi']),int(request.form['StopLoss_rsi']),int(request.form['TakeProfit_rsi'])]

    return para


