
import numpy as np
from pandas import to_datetime, read_csv
from datetime import timedelta

def detect(num):
    num = int(num)
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(num * np.pi * t)

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure() and then ax = fig.add_subplot(111)
    #fig, ax = plt.subplots()
    #ax.plot(t, s)

    #ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           #title='About as simple as it gets, folks '+ str(num))
    #ax.grid()

    #fig.savefig("static/brd/test.png")
    labels = ["January", "February", "March", "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return labels,values





def get_data():
    start_date = "2012-02-01"
    end_date = "2012-03-01"
    data_file = "D:/coursework/L4S2/GroupProject/repo/TeamFxPortal/static/data/" + "EURUSD" + "/DAT_MT_" + "EURUSD" + "_M1_" + \
                "2012" + ".csv"
    # news = ["Brexit","US presidential election 2012"]


    # price
    data = read_csv(data_file)
    data['Time'] = data[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
    data['Time'] = data['Time'].apply(lambda x: to_datetime(x) - timedelta(hours=2))
    data['Time'] = data['Time'].astype(str)
    #data['Time'] = data['Time'].apply(lambda x: x.strftime('%m/%d/%Y'))
    data.index = data.Time
    mask = (data.index > start_date) & (data.index <= end_date)
    data = data.loc[mask]
    series = data["Close"]
    print(series)
    labels = series.index
    values = series.values

    labels = list(labels)
    values = list(values)
    return labels, values, len(labels)