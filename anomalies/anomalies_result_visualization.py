import datetime
from flask import Flask,redirect, url_for, request
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import MONDAY, DateFormatter, DayLocator, WeekdayLocator



def plot_results():
    print("duck")
    year = request.form["year"]
    from_month = request.form["from_month"]
    to_month = request.form["to_month"]

    print('anomalies are detecting...')
    print('year: ' + str(year))
    print('from_month: ' + str(from_month))
    print('to_month: ' + str(to_month))
    currency = request.form["currency"]

    start_date = request.form["year"] + "-" + request.form["from_month"] + "-01"
    #end_date = request.form["year"] + "-" + str(int(request.form["to_month"]) + 1) + "-01"
    if (int(request.form["to_month"]) == 12):
        end_date = str(int(request.form["year"]) + 1) + "-" + str(1) + "-01"
    else:
        end_date = request.form["year"] + "-" + str(int(request.form["to_month"]) + 1) + "-01"

    quotes = pd.read_csv("static/data/"+currency+"/DAT_MT_"+currency+"_M1_" + str(year) + ".csv")


    quotes['Time'] = quotes[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    quotes['Time'] = quotes['Time'].apply(lambda x: pd.to_datetime(x) - datetime.timedelta(hours=2))
    quotes.index = quotes.Time

    print(quotes)
    # select desired range of dates
    mask = (quotes.index > start_date) & (quotes.index <= end_date)
    quotes = quotes.loc[mask]

    print(quotes)

    fig, ax = plt.subplots()
    ax.plot(quotes['Close'])
    ax.set_title('Black Regions')
    anormalies = pd.read_csv("static/anomalies/all_anomalies.csv")

    anormalies['Time'] = anormalies['DateHour'].apply(lambda x: pd.to_datetime(x))
    anormalies.index = anormalies.Time
    mask = (anormalies.index > start_date) & (anormalies.index <= end_date)
    anormalies = anormalies.loc[mask]
    #xdate = [datetime.datetime.fromtimestamp(i) for i in quotes['Time']]

    for index, row in anormalies.iterrows():
        a_index = row["Time"]
        b_index = row["Time"]+datetime.timedelta(hours=1)
        ax.axvspan(a_index, b_index, color='red', alpha=0.5)

    plt.show()

#plot_results()


"""
from flask import Flask,redirect, url_for, request
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mpl_finance as finance
#import candlestick2_ohlc
import  mpld3
import matplotlib.dates as mdates
import datetime as datetime
import pandas as pd



def plot_results():
    year = request.form["year"]
    from_month = request.form["from_month"]
    to_month = request.form["to_month"]

    print('anomalies are detecting...')
    print('year: ' + str(year))
    print('from_month: ' + str(from_month))
    print('to_month: ' + str(to_month))

    #start_date = "2016-06-01"
    #end_date = "2016-07-01"
    currency = "EURUSD"
    start_date = str(year) + '-'+str(from_month) + '-01'
    end_date = str(year) + '-' + str(to_month) + '-01'
    epoch = datetime.datetime(1970, 1, 1)
    data = pd.read_csv("static/data/"+currency+"/DAT_MT_"+currency+"_M1_" + str(year) + ".csv")

    data['Time'] = data[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    data['Time'] = data['Time'].apply(lambda x: pd.to_datetime(x) - datetime.timedelta(hours=2) )
    data.index = data.Time
    mask = (data.index > start_date) & (data.index <= end_date)
    data = data.loc[mask]
    data['Time'] = data['Time'].apply(lambda x: (x- datetime.timedelta(hours=5.5) - epoch).total_seconds())


    fig, ax = plt.subplots()
    finance.candlestick2_ohlc(ax, zip(mdates.date2num(data.index),
                                      data['Open'],data['High'],data['Low'],data['Close'])
                              ,width=0.6)
    xdate = [datetime.datetime.fromtimestamp(i) for i in data['Time']]

    def format_date(x, pos):
        try:
            return xdate[int(x)]
        except IndexError:
            return ''

    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

    #zoom effect
    #ax.set(xlim=(0, 10000),  autoscale_on=False,title='Zoom window')
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    anormalies = pd.read_csv("static/anomalies/all_anomalies.csv")

    anormalies['Time'] = anormalies['DateTime'].apply(lambda x: pd.to_datetime(x))
    anormalies.index = anormalies.Time
    mask = (anormalies.index > start_date) & (anormalies.index <= end_date)
    anormalies = anormalies.loc[mask]


    #for index, row in anormalies.iterrows():
    #   a_index = xdate.index(row["Time"])
    #   b_index = xdate.index(row["Time"]+datetime.timedelta(hours=1))
    #   ax.axvspan(a_index, b_index, color='red', alpha=0.5)

    #fig.autofmt_xdate()
    #fig.tight_layout()

    plt.show()
    #mpld3.show()

#plot_results()

"""

#https://stackoverflow.com/questions/42373104/candlestick-ochl-graph