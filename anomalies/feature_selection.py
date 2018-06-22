from flask import Flask,redirect, url_for, request
import pandas as pd
import numpy as np
from pandas import to_datetime,read_csv
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
#import matplotlib.pyplot as plt
import gc
import anomalies.config as config
import plotly.plotly.plotly as py
import json


def feature_selecion():

    start_date = request.form["year"] + "-" + request.form["from_month"] + "-01"
    """if(int(request.form["to_month"])==12):
        end_date = str(int(request.form["year"])+1) + "-" + str(1) + "-01"
    else:
        end_date = request.form["year"] + "-" + str(int(request.form["to_month"])+1) + "-01" """

    #made a change here, see if not works
    if int(request.form["from_month"]) == 12:
        end_date = str(int(request.form["year"]) ) + "-" + str(12) + "-31"
    else:
        end_date = request.form["year"] + "-" + str(int(request.form["from_month"]) + 1) + "-01"

    data_file ="static/data/"+request.form["currency_pair"]+"/DAT_MT_"+request.form["currency_pair"]+"_M1_"+request.form["year"] +".csv"
    #news = ["Brexit","US presidential election 2012"]


    #price
    data = read_csv(data_file)
    data['Time'] = data[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    data['Time'] = data['Time'].apply(lambda x: to_datetime(x)-timedelta(hours=2))
    data.index = data.Time
    mask = (data.index > start_date) & (data.index <= end_date)
    data = data.loc[mask]
    series = data["Close"]


    """"#price and the gradient
    fig = plt.figure()
    #fig.tight_layout()

    ax3 = fig.add_subplot(211)
    ax3.plot(series)
    ax3.set_title(request.form["currency_pair"] + ' prices')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Price')"""

    np_array_series = np.array(data['Close'])
    np_array_dates = np.array(data.index)
    gradients = np.gradient(np_array_series)

    """ax1 = fig.add_subplot(212)
    ax1.set_title('Gradients of the price series')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Gradient')
    ax1.plot(np_array_dates, gradients)

    fig.savefig("static/anomalies/feature_lection_image1.png")
    fig,ax1,ax2 = "","","" """

    price_list = series.values
    """
    ADF_result_price = adfuller(price_list)
    print('ADF Statistic: for series %f' % ADF_result_price[0])
    print('p-value: %f' % ADF_result_price[1])  #p-value: 0.668171
    print('Critical Values:')

    for key, value in ADF_result_price[4].items():
        print('\t%s: %.3f' % (key, value))
    """

    #create log return series

    series_log_ret = np.log(data.Close) - np.log(data.Close.shift(1))
    series_log_ret = series_log_ret.dropna()
    """
    log_return_list = series_log_ret.values
    ADF_result_log_return = adfuller(log_return_list)
    print('ADF Statistic: for series_log_ret %f' % ADF_result_log_return[0])
    print('p-value: %f' % ADF_result_log_return[1])  #p-value: 0.000000 therefore, null hypothesis is rejected. the system is stationary
    print('Critical Values:')

    for key, value in ADF_result_log_return[4].items():
        print('\t%s: %.3f' % (key, value))

    input_series = []
    #testing for stationarity in series
    if ADF_result_price[0]<0.05:
        input_series = price_list
    else :
        input_series = log_return_list

    """
    #Creating the ARIMA model
    arima_model = ARIMA(series_log_ret, order=(4,1,1))
    model_fit = arima_model.fit(disp=0)
    print(model_fit.summary())
    #tsaplots.plot_acf(series_log_ret, lags=30)
    #tsaplots.plot_pacf(series_log_ret, lags=30)

    #Getting the residual series
    residuals = pd.DataFrame(model_fit.resid)
    #np.square(residuals).plot()

    residual_list = residuals.values
    residual_squared = list()
    """
    for x in residual_list:
        residual_squared.append(x[0])

    #checking for stationarity in the residual series
    ADF_result_residual_squared = adfuller(residual_squared)
    print('ADF Statistic: for residuals %f' % ADF_result_residual_squared[0])
    print('p-value: %f' % ADF_result_residual_squared[1])  #p-value: 0.000000 therefore, null hypothesis is rejected. the system is stationary
    print('Critical Values:')
    for key, value in ADF_result_residual_squared[4].items():
        print('\t%s: %.3f' % (key, value))
    """
    #different configurations for GARCH model
    configurations = [
        [2,0,0],
        [2,0,1],
        [1,0,0],
        [1,0,1]
    ]


    opt_model = {}
    #opt_configuration = []

    #getting the most suitable configuration
    for i in range(len(configurations)):
        BIC = np.inf
        garch_model = arch_model(series_log_ret, p=configurations[i][0], o=configurations[i][1], q=configurations[i][2])
        model = garch_model.fit(update_freq=5)
        if BIC > model.bic:
            BIC = model.bic
            opt_model = model
            #opt_configuration = configurations[i]

    print(opt_model.summary())
    conditional_volatility = opt_model.conditional_volatility


    #https://plot.ly/matplotlib/subplots/ for four
    #for three
    #ax1 = fig.add_subplot(221)

    """fig = plt.figure()
    #fig.tight_layout()

    ax3 = fig.add_subplot(221)
    ax3.plot(series)
    ax3.set_title(request.form["currency_pair"] + ' prices')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Price')

    ax2 = fig.add_subplot(222)
    ax2.plot(conditional_volatility)
    ax2.set_title('Conditional Volatility')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Conditional Volatility')


    ax1 = fig.add_subplot(223)
    ax1.plot(np_array_dates, gradients)
    ax1.set_title('Gradients: ' + request.form["currency_pair"] + ' prices')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Gradient')"""


    np_array_CH = np.array(conditional_volatility)
    np_array_CH_dates = np.array(conditional_volatility.index)
    gradients_CH = np.gradient(np_array_CH)

    """ax4 = fig.add_subplot(224)
    ax4.plot(np_array_CH_dates, gradients_CH)
    ax4.set_title('Gradients: Conditional Volatility')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Gradient')

    fig.savefig("static/anomalies/feature_lection_image2.png")
    fig, ax1, ax2, ax3, ax4 = "", "", "", "", "" """

    df_CH = pd.DataFrame()
    df_CH['Index'] =  np_array_CH_dates
    df_CH['CH_Gradient'] =  gradients_CH
    df_CH.index = df_CH['Index']
    df_CH['CH'] = conditional_volatility
    df_CH = df_CH.drop(['Index'],  axis=1)

    df_price = pd.DataFrame()
    df_price['Index'] =  np_array_dates
    df_price['Price_Gradient'] = gradients
    df_price.index = df_price['Index']
    df_price['Price'] = series
    df_price = df_price.drop(['Index'], axis=1)

    features = pd.concat([df_price, df_CH], axis=1)
    features = features.dropna(axis=0)

    gc.collect()

    features.to_csv('static/anomalies/features/'+request.form["currency_pair"]+'_'+request.form["year"]+'.csv')


    labels = series.index
    price_values = series.values
    # gradients_values = gradients
    # volatility_values = conditional_volatility.values
    volatility_gradients_values = gradients_CH

    #labels = []
    #price_values = []
    #gradients_values = []
    #volatility_values = []
    #volatility_gradients_values = []
    labels = list(labels)
    price_values = list(price_values)
    # gradients_values = list(gradients_values)
    # volatility_values = list(volatility_values)
    volatility_gradients_values = list(volatility_gradients_values)

    gc.collect()



    graphs = [
        dict(
            data=[
                dict(
                    x=labels,
                    y=price_values,
                    type='scatter'
                ),
            ],
            layout=dict(
                title='Price'
            )
        ),
        dict(
            data=[
                dict(
                    x=labels,
                    y=volatility_gradients_values,
                    type='scatter'
                ),
            ],
            layout=dict(
                title='Gradients of Condisional Heteroskedasiticity'
            )
        ),
        dict(
            data=[
                dict(
                    x=volatility_gradients_values,
                    y=price_values,
                    type='scatter',
                    mode = 'markers'
                ),
            ],
            layout=dict(
                title='Scattler plot of prices and conditional volatility'
            )
        ),
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=py.utils.PlotlyJSONEncoder)

    return request.form["year"], request.form["from_month"], request.form["from_month"], request.form["currency_pair"],len(price_values),ids, graphJSON

