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

def get_visualize_view():
    start_date = request.form['from_date']
    end_date = request.form['to_date']

    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]

    currency_pair = request.form['currency_pair']



    if start_year==end_year:

        data_file = "static/data/" + currency_pair + "/DAT_MT_" + currency_pair + "_M1_" + start_year + ".csv"
        data = pd.read_csv(data_file)
        print("one year range")
    else:
        data_file = "static/data/" + currency_pair + "/DAT_MT_" + currency_pair + "_M1_" + start_year + ".csv"
        data = pd.read_csv(data_file)
        print("can cause errors")

    data['Time'] = data[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    data['Time'] = data['Time'].apply(lambda x: pd.to_datetime(x) - timedelta(hours=2))
    data['Time'] = data['Time'].astype(str)

    data.index = data.Time
    mask = (data.index > start_date) & (data.index <= end_date)
    data = data.loc[mask]
    series = data["Close"]
    print(series)
    labels = series.index
    values = series.values

    labels = list(labels)
    values = list(values)

    shapes = []

    black_region_day = '2012-02-04'
    black_region_day_end = pd.to_datetime(black_region_day) + timedelta(days=1)
    print(black_region_day_end)
    print(str(black_region_day_end))
    black_region = dict(
        type='rect',
        # x-reference is assigned to the x-values
        xref='x',
        # y-reference is assigned to the plot paper [0,1]
        yref='paper',
        x0=black_region_day,
        y0=0,
        x1=str(black_region_day_end),
        y1=1,
        fillcolor='#d3d3d3',
        opacity=0.2,
        line=dict(
            width=0,
        )
    )

    shapes.append(black_region)

    rng = pd.date_range('1/1/2011', periods=7500, freq='H')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)

    graphs = [
        dict(
            data=[
                dict(
                    x=labels,
                    y=values,
                    type='scatter'
                ),
            ],
            layout=dict(
                title='first graph',
                shapes=shapes
            )
        ),

        dict(
            data=[
                dict(
                    x=[1, 3, 5],
                    y=[10, 50, 30],
                    type='bar'
                ),
            ],
            layout=dict(
                title='second graph'
            )
        ),

        dict(
            data=[
                dict(
                    x=ts.index,  # Can use the pandas data structures directly
                    y=ts
                )
            ]
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=py.utils.PlotlyJSONEncoder)

    return ids, graphJSON


