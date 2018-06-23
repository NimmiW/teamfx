from flask import Flask,redirect, url_for, request


def individual(strategy):
    ind = []
    if (strategy == 'Moving Average'):
        ind = [int(request.form['ShortMA']),
               int(request.form['LongMA']),
               int(request.form['StopLoss']),
               int(request.form['TakeProfit'])]

    return ind





