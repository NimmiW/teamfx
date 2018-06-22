from flask import request
import plotly.plotly.plotly as py
import pandas as pd
import json
from datetime import timedelta

def get_visualize_view(threshold,nneighbours,page="none"):
    root = "D:/coursework/L4S2/GroupProject/repo/TeamFxPortal/"

    if(page == 'anomaly_detection'):

        start_month = request.form['from_month']
        end_month = request.form['to_month']
        currency_pair = request.form['currency']
        start_year = request.form['year']
        end_year = request.form['year']
        start_date = start_year+'-'+start_month+'-01'
        end_date = end_year + '-' + end_month + '-01'
    else:
        start_date = request.form['from_date']
        end_date = request.form['to_date']
        currency_pair = request.form['currency_pair']
        start_year = start_date.split('-')[0]
        end_year = end_date.split('-')[0]





    if start_year==end_year:

        data_file = root+"static/data/" + currency_pair + "/DAT_MT_" + currency_pair + "_M1_" + start_year + ".csv"
        data = pd.read_csv(data_file)
        print("one year range")
    else:
        data_file = root+"static/data/" + currency_pair + "/DAT_MT_" + currency_pair + "_M1_" + start_year + ".csv"
        data = pd.read_csv(data_file)
        print("can cause errors")

    data['Time'] = data[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    data['Time'] = data['Time'].apply(lambda x: pd.to_datetime(x) - timedelta(hours=2))
    data['Time'] = data['Time'].astype(str)

    data.index = data.Time
    mask = (data.index > start_date) & (data.index <= end_date)
    print("start_date")
    print(start_date)
    print(end_date)
    data = data.loc[mask]
    series = data["Close"]

    labels = series.index
    values = series.values

    labels = list(labels)
    values = list(values)

    shapes = []

    #anormalies = pd.read_csv("static/anomalies/all_anomalies.csv")
    anormalies = pd.read_csv(root + 'static/anomalies/detected_black_regions/'+str(threshold) + '_' + str(nneighbours) + '_' + currency_pair + '_' + start_year+'_all_anomalies.csv')

    anormalies['Time'] = anormalies['DateHour'].apply(lambda x: pd.to_datetime(x))
    anormalies.index = anormalies.Time
    mask = (anormalies.index > start_date) & (anormalies.index <= end_date)
    anormalies = anormalies.loc[mask]

    for black_hour in anormalies.index:
        black_hour_end = pd.to_datetime(black_hour) + timedelta(hours=1)

        black_region = dict(
            type='rect',
            # x-reference is assigned to the x-values
            xref='x',
            # y-reference is assigned to the plot paper [0,1]
            yref='paper',
            x0=black_hour,
            y0=0,
            x1=str(black_hour_end),
            y1=1,
            fillcolor='#ff0000',
            opacity=0.2,
            line=dict(
                width=0,
            )
        )

        shapes.append(black_region)

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
                title='Anomalies',
                shapes=shapes
            )
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


