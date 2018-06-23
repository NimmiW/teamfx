from flask import Flask,redirect, url_for, request

def getPara(strategy):
    para = []
    if(strategy == "Moving Average"):
        para = [int(request.form['ShortMA']),int(request.form['LongMA']),int(request.form['StopLoss']),int(request.form['TakeProfit'])]


    return para


