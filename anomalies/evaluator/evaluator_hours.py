import pandas as pd
import numpy as np
from pandas import to_datetime
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt
import os
import datetime
import time
import anomalies.config as config
import anomalies.anomaly_identification as ano
import plotly.plotly.plotly as py
import pandas as pd
import json


def evaluate(threshold, nneighbours, year, currency):
    root_evaluate = config.ROOT+"anomalies/evaluator/"
    root_static_anomalies = config.ROOT+"static/anomalies/"
    black_regions = pd.read_csv('black_regions_hours/' + currency +'_'+str(year)+ '_true_anomalies.csv')
    black_regions['hour'] = black_regions['true_anomalies'].apply(lambda x: to_datetime(x))
    black_regions.index = black_regions['hour']
    black_regions['label'] = 1
    black_regions = black_regions.drop(['hour','true_anomalies' ], axis=1)

    results = pd.read_csv(root_static_anomalies+"detected_black_regions/"+str(threshold)+'_'+str(nneighbours)+'_'+ currency + '_'+str(year)+'_all_anomalies.csv')
    results['hour'] = results['DateHour'].apply(lambda x: to_datetime(x))
    results.index = results['hour']
    results['result'] = 1
    results = results.drop(['hour','DateHour','Count','Average_lof', 'Ranking_Factor'], axis=1)

    time_list = pd.read_csv('data/' + currency + '/DAT_MT_'+currency+'_M1_'+str(year)+'.csv')
    time_list['Time'] = time_list[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    time_list = list(set(time_list['Time'].apply(lambda x: to_datetime(x).replace(minute=0, second=0, microsecond=0)).values))
    time_df = pd.DataFrame(time_list)
    time_df.index = time_df[0]
    time_df['sample_loc']=0
    time_df = time_df.drop([0], axis=1)

    #Shape of passed values is (2, 78), indices imply (2, 56)
    joined_table = pd.concat([time_df,black_regions, results], axis=1).fillna(0).sort_index().astype(int)
    joined_table = joined_table.drop(['sample_loc'], axis=1)

    actual_black_regions = joined_table.loc[joined_table['label'] == 1]
    actual_non_black_regions = joined_table.loc[joined_table['result'] == 0]
    true_positive = joined_table.loc[(joined_table['label'] == 1) & (joined_table['result'] == 1)]
    true_negative = joined_table.loc[(joined_table['label'] == 0) & (joined_table['result'] == 0)]
    false_positive = joined_table.loc[(joined_table['label'] == 0) & (joined_table['result'] == 1)]
    false_negative = joined_table.loc[(joined_table['label'] == 1) & (joined_table['result'] == 0)]


    accuracy = (len(true_positive) + len(true_negative)) / len(joined_table)
    precision = len(true_positive) / (len(true_positive)+len(false_positive))
    recall = len(true_positive) / (len(true_positive)+len(false_negative))

    store_df = pd.DataFrame()
    store_df['Column'] = ['year',
                          'Currency',
                          'Threshold',
                          'Neighbours',
                          'Actual Black region',
                          'Actual Non-Black region count',
                          'True-Positive',
                          'False-Positive',
                          'True-Negative',
                          'False-Negative',
                          'Precision',
                          'Recall',
                          'Accuracy'
                          ]
    store_df['value'] = [
                         year,
                         currency,
                         threshold,
                         nneighbours,
                         len(actual_black_regions),
                         len(actual_non_black_regions),
                         len(true_positive),
                         len(false_positive),
                         len(true_negative),
                         len(false_negative),
                         precision,
                         recall,
                         accuracy
                        ]
    store_df.to_csv('results/results_of_'+str(threshold)+'_'+str(nneighbours)+'_'+currency +'_'+str(year)+'.csv')

    print('Actual Black region count = ' + str(actual_black_regions))
    print('Actual Non-Black region count = ' + str(len(actual_non_black_regions)))
    print('Detected True-Positive count = ' + str(len(true_positive)))
    print('Detected False-Positive count = ' + str(len(false_positive)))
    print('Detected True-Negative count = ' + str(len(true_negative)))
    print('Detected False-Negative count = ' + str(len(false_negative)))


def draw_graph():

    ## Create functions and set domain length
    #for threshold
    #x = [1,0.75,0.637931034,0.338983051,0.207446809,0.172413793,0.128358209,0] #Pecision
    #y = [0,0.14516129, 0.596774194,0.64516129, 0.629032258,0.725806452,0.693548387,1]#Recall

    #for NN

    x = [0,0.000486696950032446,0.00340632603406326, 0.0126438644837089,0.0241569390402075,0.0350081037277147, 0.0473181007940366,1] #FPR

    y = [0, 0.14516129, 0.596774194, 0.64516129, 0.629032258, 0.725806452, 0.693548387, 1]  # Recall

    ## Plot functions and a point where they intersect
    plt.plot(x, y)
    plt.plot([0,0],[1,1])
    #plt.plot(x, dy)
    #plt.plot(1, 1, 'or')

    ## Config the graph
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.ylim([0, 4])
    plt.grid(True)
    plt.legend(['Model PR Curve', 'Baseline'], loc='upper left')

    ## Show the graph
    plt.show()


def show_evaluate_results(threshold, nneighbours, year, currency):
    print('show_evaluate_results method')
    print(threshold)
    print(year)
    print(currency)
    root_static_anomalies = config.ROOT+"anomalies/evaluator/"
    results = pd.read_csv(root_static_anomalies+'results/results_of_' + str(threshold) + '_' + str(nneighbours) + '_' + currency + '_' + str(year) + '.csv')
    print(results)
    return results

def PR_curve_visualize():
    x = [1,0.75,0.637931034,0.338983051,0.207446809,0.172413793,0.128358209,0] #Pecision
    y = [0,0.14516129, 0.596774194,0.64516129, 0.629032258,0.725806452,0.693548387,1]#Recall

    graphs = [
        dict(
            data=[
                dict(
                    x=x,
                    y=y,
                    type='scatter',
                    legendgroup= 'group2',
                    name= 'PR'
                ),
                dict(
                    x=[1,0],
                    y=[0,1],
                    type='scatter',
                    legendgroup='group2',
                    name='Baseline'
                ),
            ],
            layout=dict(
                title='PR Curve',
                xaxis=dict(
                    title='Precision',
                    titlefont=dict(
                        #family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f'
                    )
                ),
                yaxis=dict(
                    title='Recall',
                    titlefont=dict(
                        #family='Courier New, monospace',
                        size=18,
                        color='#7f7f7f'
                    )
                )
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


#show_evaluate_results(2, 2, 2016, "EURUSD")

#evaluate(1, 2, 2017, 'EURUSD')
#draw_graph()