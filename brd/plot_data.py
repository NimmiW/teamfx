import plotly.plotly.plotly as py
import pandas as pd
import numpy as np
import json
from datetime import timedelta

def plot_function():
    start_date = "2012-02-01"
    end_date = "2012-03-01"
    data_file = "D:/coursework/L4S2/GroupProject/repo/TeamFxPortal/static/data/" + "EURUSD" + "/DAT_MT_" + "EURUSD" + "_M1_" + \
                "2012" + ".csv"
    # news = ["Brexit","US presidential election 2012"]


    # price
    data = pd.read_csv(data_file)
    data['Time'] = data[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    data['Time'] = data['Time'].apply(lambda x: pd.to_datetime(x) - timedelta(hours=2))
    data['Time'] = data['Time'].astype(str)
    # data['Time'] = data['Time'].apply(lambda x: x.strftime('%m/%d/%Y'))
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
    black_region_day_end =pd.to_datetime(black_region_day) + timedelta(days=1)
    print(black_region_day_end)
    print(str(black_region_day_end))
    black_region = dict(
        type= 'rect',
        # x-reference is assigned to the x-values
        xref= 'x',
        # y-reference is assigned to the plot paper [0,1]
        yref= 'paper',
        x0= black_region_day,
        y0= 0,
        x1=str(black_region_day_end),
        y1= 1,
        fillcolor= '#d3d3d3',
        opacity=0.2,
        line= dict(
           width= 0,
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
                shapes= shapes
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

    return ids,graphJSON


def plot_function_original():

    rng = pd.date_range('1/1/2011', periods=7500, freq='H')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)

    graphs = [
        dict(
            data=[
                dict(
                    x=[1, 2, 3],
                    y=[10, 20, 30],
                    type='scatter'
                ),
            ],
            layout=dict(
                title='first graph'
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

    return ids,graphJSON




#https://github.com/plotly/plotlyjs-flask-example