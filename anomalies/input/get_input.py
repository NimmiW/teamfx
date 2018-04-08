import pandas as pd
import numpy as np
from pandas import to_datetime,read_csv
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt
from anomalies.input.month_range_selector import interpolate_months

def get_input(from_month,to_month,currency):
    list_of_interpolated_months = interpolate_months(from_month,to_month)
    return []