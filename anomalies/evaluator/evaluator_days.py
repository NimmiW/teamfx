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

def evaluate(black_regions_file,currency,root):
    black_regions = pd.read_csv(root + 'evaluator/black_regions/' + currency + '_currency_data.csv')
    black_regions.Date = black_regions['Date'].apply(lambda x: to_datetime(x).date())
    black_regions.index = black_regions['Date']
    black_regions = black_regions.drop(['Date'],  axis=1).fillna(0)
    black_regions = black_regions.astype(int)

    results = pd.read_csv(black_regions_file)
    results.Date = results['Date'].apply(lambda x: to_datetime(x).date())
    results.index = results['Date']
    results = results.drop(['Date'], axis=1)
    results = results.astype(int)
    results['is_detected'] = 1
    joined_table = pd.concat([black_regions, results], axis=1).fillna(0).sort_index().astype(int)
    #print(joined_table)
    #joined_table.to_csv('joined_table.csv')

    actual_black_regions = joined_table.loc[joined_table['is_abnormal'] == 1]
    actual_non_black_regions = joined_table.loc[joined_table['is_abnormal'] == 0]
    true_positive = joined_table.loc[(joined_table['is_abnormal'] == 1) & (joined_table['is_detected'] == 1)]
    true_negative = joined_table.loc[(joined_table['is_abnormal'] == 0) & (joined_table['is_detected'] == 0)]
    false_positive = joined_table.loc[(joined_table['is_abnormal'] == 0) & (joined_table['is_detected'] == 1)]
    false_negative = joined_table.loc[(joined_table['is_abnormal'] == 1) & (joined_table['is_detected'] == 0)]


    accuracy = (len(true_positive) + len(true_negative)) / len(joined_table)

    print('Actual Black region count = ' + str(actual_black_regions))
    print('Actual Non-Black region count = ' + str(len(actual_non_black_regions)))
    print('Detected True-Positive count = ' + str(len(true_positive)))
    print('Detected False-Positive count = ' + str(len(false_positive)))
    print('Detected True-Negative count = ' + str(len(true_negative)))
    print('Detected False-Negative count = ' + str(len(false_negative)))

    return accuracy, true_positive, true_negative, false_positive, false_negative
