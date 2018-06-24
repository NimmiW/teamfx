import numpy as np
import pandas as pd


# Relative Strength Index indicator

def rsi_function(data,window):
    data = pd.DataFrame(data)
    delta = data.diff()
    d_up, d_down = delta.copy(), delta.copy()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0
    rollup = pd.rolling_mean(d_up, window)
    rolldown = pd.rolling_mean(d_down, window).abs()
    rs = rollup / rolldown
    rsi = 1.0 - (1.0 / (1.0 + rs))
    return rsi

# willR indicator

def willR(df, window):
    # %R = -100 * ( ( Highest High - Close) / (Highest High - Lowest Low ) )
    extra = len(df)-window

    df= df[extra:]
    close = df.iloc[len(df)-1,3]
    high_list = df['high'].values.tolist()
    low_list = df['low'].values.tolist()

    highest_high = np.amax(high_list)
    lowest_low = np.amin(low_list)

    willR = ((highest_high - close) / (highest_high - lowest_low)) * -1

    return willR


# Exponential moving average

def expMovingAverage(data, window):
    extra = len(data)- window

    data = data[extra:]

    data = pd.DataFrame({'col': data})
    ema = data.ewm(com=0.5).mean()
    ema = ema.iloc[window-1,0]
    return ema

