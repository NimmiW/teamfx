import numpy as np
from statsmodels.tsa.arima_model import ARIMA
# from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import warnings
import math
import predictions.indicators as ind
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')
path = 'E:/moodle/Level04S02/Teamfx/preprocessing'
indicators_ohlc_and_label = pd.read_csv(path + '/EURUSD_M1_2012_indicators_with_OHLC.csv')
column_list = ['open', 'high', 'low', 'close', 'will_r', 'ema', 'rsi', 'open_close']

indicators_ohlc_and_label['next_open'] = indicators_ohlc_and_label['open'].shift(-1)
indicators_ohlc_and_label['next_high'] = indicators_ohlc_and_label['high'].shift(-1)
indicators_ohlc_and_label['next_low'] = indicators_ohlc_and_label['low'].shift(-1)
indicators_ohlc_and_label['next_close'] = indicators_ohlc_and_label['close'].shift(-1)

indicators_ohlc_and_label.index = indicators_ohlc_and_label.Datetime
indicators_and_label = indicators_ohlc_and_label.drop(['Datetime'], axis=1)
df = indicators_and_label.dropna().iloc[:10100]

train_set_index = 10000
test_set_index = 10100


def get_test_labels():
    label = df[['next_open', 'next_high', 'next_low', 'next_close']].iloc[train_set_index:test_set_index]
    # features = df[['open', 'high', 'low', 'close', 'will_r', 'ema', 'rsi']].iloc[train_set_index:test_set_index]
    return label


def get_test_features():
    # label = df[['next_open', 'next_high', 'next_low', 'next_close']].iloc[train_set_index:test_set_index]
    features = df[['open', 'high', 'low', 'close', 'will_r', 'ema', 'rsi']].iloc[train_set_index:test_set_index]
    return features


# x = tf.placeholder(tf.float32, shape=[None, 7])
y = tf.placeholder(tf.float32, shape=[None, 4])

# settings

layer1_nodes = 15
layer2_nodes = 15
layer3_nodes = 15
output_nodes = 4


# create model
def model(data):
    layer_1 = {
        'weights': tf.Variable(tf.random_normal([7, layer1_nodes])),
        'bias': tf.Variable(tf.random_normal([layer1_nodes]))
    }

    layer_2 = {
        'weights': tf.Variable(tf.random_normal([layer1_nodes, layer2_nodes])),
        'bias': tf.Variable(tf.random_normal([layer2_nodes]))
    }

    layer_3 = {
        'weights': tf.Variable(tf.random_normal([layer2_nodes, layer3_nodes])),
        'bias': tf.Variable(tf.random_normal([layer3_nodes]))
    }

    output_layer = {
        'weights': tf.Variable(tf.random_normal([layer3_nodes, output_nodes])),
        'bias': tf.Variable(tf.random_normal([output_nodes]))
    }

    operation_layer1 = tf.add(tf.matmul(data, layer_1['weights']), layer_1['bias'])
    operation_layer1 = tf.nn.sigmoid(operation_layer1)

    operation_layer2 = tf.add(tf.matmul(operation_layer1, layer_2['weights']), layer_2['bias'])
    operation_layer2 = tf.nn.sigmoid(operation_layer2)

    operation_layer3 = tf.add(tf.matmul(operation_layer2, layer_3['weights']), layer_3['bias'])
    operation_layer3 = tf.nn.sigmoid(operation_layer3)

    operation_output = tf.add(tf.matmul(operation_layer3, output_layer['weights']), output_layer['bias'])
    return operation_output


def predict_2(count, df=get_test_features(), labels=get_test_labels()):
    data = df['close']

    data = data.tolist()

    df_1 = df[-15:-5]  # FWD
    df_2 = df[-15:-5]  # bwd

    df = df[:-5]

    predictions = []
    X = np.array(df_1.values)  # convert to numpy.ndarray
    X = X[len(X) - 2:len(X) - 1]

    errors = []
    predictions_list = []
    predicted_close = []

    prediction = model(x)  # operation to get predictions using the trained model.
    """
    print('prediction', prediction)

    # saved file location to retriev Saver object
    save_file = 'E:\\moodle\\Level04S02\\Teamfx\\weights_record\\T01-2018-05-21_next_ohlc\\nextohlc_10000_nimmi.ckpt'
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, save_file)

    # j = 1
    with sess.as_default():

        # predict forward
        for i in range(count):
            print(X)
            predict_value = sess.run([prediction], feed_dict={x: X})
            open = predict_value[0][0][0]
            high = predict_value[0][0][1]
            low = predict_value[0][0][2]
            close = predict_value[0][0][3]

            data.append(predict_value[0][0][3])
            df = df.append({'high': high, 'low': low, 'open': open, 'close': close}, ignore_index=True)

            predictions.append(predict_value[0][0])
            predicted_close.append(predict_value[0][0][3])

            print(predictions)
            window = 10
            ema = ind.expMovingAverage(data, window)
            willR = ind.willR(df, window)
            rsi = ind.rsi_function(data, window)
            rsi = rsi.iloc[window, 0]
            # 'will_r', 'ema', 'rsi'
            df.iloc[len(df) - 1, 4] = willR  # last row 4th column
            df.iloc[len(df) - 1, 5] = ema  # last row 5th column
            df.iloc[len(df) - 1, 6] = rsi  # last row 6th column
            print('df')
            print(df)
            print('df')
            X = np.array(df)  # convert to numpy.ndarray
            X = X[len(X) - 2:len(X) - 1]

        # predict 10 previous errors
        j = 1
        while j <= len(df_2):
            predict_value = sess.run([prediction], feed_dict={x: df_2[j - 1:j]})
            predictions_list.append(predict_value[0][0][3])

            # predicted_value = predict_value[3]

            # print(predicted_value)
            j += 1

        actual_list = labels.tail(15)['next_close'].values
        actual_list = actual_list.tolist()
        print(actual_list)
        print(predictions_list)

        errors = np.array(actual_list[-15:-5]) - np.array(predictions_list)
        actual_values_forward = actual_list[-5:]

    return errors, predicted_close, actual_values_forward
"""


def predict_arima_ann():
    errors, predictions_from_ann, actual_values_forward = predict(5)
    print('predictions_from_ann')
    print(predictions_from_ann)
    print(errors)
    print(actual_values_forward)

    #############################################################################

    series_error = pd.Series(errors.tolist())  # .map(lambda x: x*1000)
    series_pct_change = series_error  # series_error.pct_change().dropna()
    # here in error negative positive points are important then log can not be use here and pct change used.

    X = series_pct_change.values.tolist()

    # size = int(len(X) * 0.9)
    # train = X[0:size]
    # test = [x for x in X[size:len(X)]][:5]
    #
    # predictions = list()
    history = [x for x in X]

    model = ARIMA(history, order=(2, 0, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    output = model_fit.forecast(steps=5)[0].tolist()

    print(output)

    print(predictions_from_ann)

    final_results = np.array(predictions_from_ann) + np.array(output)
    print(final_results)
    # error = mean_squared_error(test, output)
    # print('Test MSE: %.3f' % error)

    # plt.plot(output)
    # plt.plot(test, color='red')
    # plt.show()

    #############################################################################


def predict(num):
    x = tf.placeholder(tf.float32, shape=[None, 7])

    df = get_test_features()
    labels = get_test_labels()
    data = df['close']

    data = data.tolist()

    df_1 = df[-15:-5]  # FWD
    df_2 = df[-15:-5]  # bwd
    df = df[:-5]

    predictions = []
    X = np.array(df_1.values)  # convert to numpy.ndarray
    X = X[len(X) - 2:len(X) - 1]

    errors = []
    predictions_list = []
    predicted_close = []

    prediction = model(x)
    print('prediction', prediction)

    # saved file location to retriev Saver object
    save_file = 'E:\\moodle\\Level04S02\\Teamfx\\weights_record\\T01-2018-05-21_next_ohlc\\nextohlc_10000_nimmi.ckpt'
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, save_file)

    print("fuck you")

    # j = 1
    with sess.as_default():

        # predict forward
        for i in range(num):
            predict_value = sess.run([prediction], feed_dict={x: X})
            open = predict_value[0][0][0]
            high = predict_value[0][0][1]
            low = predict_value[0][0][2]
            close = predict_value[0][0][3]

            data.append(predict_value[0][0][3])
            df = df.append({'high': high, 'low': low, 'open': open, 'close': close}, ignore_index=True)

            predictions.append(predict_value[0][0])
            predicted_close.append(predict_value[0][0][3])

            print(predictions)
            window = 10
            ema = ind.expMovingAverage(data, window)
            willR = ind.willR(df, window)
            rsi = ind.rsi_function(data, window)
            rsi = rsi.iloc[window, 0]
            # 'will_r', 'ema', 'rsi'
            df.iloc[len(df) - 1, 4] = willR  # last row 4th column
            df.iloc[len(df) - 1, 5] = ema  # last row 5th column
            df.iloc[len(df) - 1, 6] = rsi  # last row 6th column

            X = np.array(df)  # convert to numpy.ndarray
            X = X[len(X) - 2:len(X) - 1]

        # predict 10 previous errors
        j = 1
        while j <= len(df_2):
            predict_value = sess.run([prediction], feed_dict={x: df_2[j - 1:j]})
            predictions_list.append(predict_value[0][0][3])

            # predicted_value = predict_value[3]

            # print(predicted_value)
            j += 1

        actual_list = labels.tail(15)['next_close'].values
        actual_list = actual_list.tolist()

        errors = np.array(actual_list[-15:-5]) - np.array(predictions_list)
        actual_values_forward = actual_list[-5:]

    return errors


    # predict(5)