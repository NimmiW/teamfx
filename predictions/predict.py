import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import warnings
import math
import predictions.snk.indicators as ind
from predictions.preprocessing.min_max import min_max, min_max_predict, invert_min_max, invert_min_max_residual, \
    invert_min_max_for_one_feature
from pandas import to_datetime
import json
import plotly.plotly.plotly as py
from sklearn.metrics import mean_squared_error

# warnings.filterwarnings('ignore')
# path = 'E:/moodle/Level04S02/Teamfx/preprocessing'
# indicators_ohlc_and_label = pd.read_csv(path+'/EURUSD_M1_2012_indicators_with_OHLC_2018-06-18.csv')
# column_list = ['open', 'high', 'low', 'close', 'will_r', 'ema', 'rsi', 'next_close']
#
# indicators_ohlc_and_label['next_close'] = indicators_ohlc_and_label['close'].shift(-1)
#
# indicators_ohlc_and_label.index = indicators_ohlc_and_label.Datetime
# indicators_and_label = indicators_ohlc_and_label.drop(['Datetime'], axis=1)
# df = indicators_and_label.dropna().iloc[:10100]

# min_maxed_dataset = min_max(df, 'minmaxvalues_all_data_2012_20180619', column_list=column_list).iloc[:10100]

# train_set_index = 10000
# test_set_index = 10100
#
# def get_train_labels():
#     label = df[['next_close']].iloc[:train_set_index]
#     return label
#
# def get_train_features():
#     features = df[['close', 'ema', 'rsi']].iloc[:train_set_index]
#     return features
#
# def get_test_labels():
#     #label = df[['next_close']].iloc[train_set_index:test_set_index]
#     label = min_maxed_dataset[['next_close']].iloc[train_set_index:test_set_index]
#     return label
#
# def get_test_features():
#     #features = df[['close', 'ema', 'rsi']].iloc[train_set_index:test_set_index]
#     features = min_maxed_dataset[['close', 'ema', 'rsi']].iloc[train_set_index:test_set_index]
#     return features
#
# def get_normalized_test_set():
#     label = min_maxed_dataset[['next_close']].iloc[train_set_index:test_set_index]
#     features = min_maxed_dataset[['close', 'ema', 'rsi']].iloc[train_set_index:test_set_index]
#     return label, features
y = tf.placeholder(tf.float32, shape=[None, 3])

# settings
batch_size = 10
layer1_nodes = 10
layer2_nodes = 10
layer3_nodes = 10
output_nodes = 3
num_epochs = 1000


# create model
def model(data):
    layer_1 = {
        'weights': tf.Variable(tf.random_normal([3, layer1_nodes])),
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
    operation_layer1 = tf.nn.relu(operation_layer1)

    operation_layer2 = tf.add(tf.matmul(operation_layer1, layer_2['weights']), layer_2['bias'])
    operation_layer2 = tf.nn.relu(operation_layer2)

    operation_layer3 = tf.add(tf.matmul(operation_layer2, layer_3['weights']), layer_3['bias'])
    operation_layer3 = tf.nn.relu(operation_layer3)

    operation_output = tf.add(tf.matmul(operation_layer3, output_layer['weights']), output_layer['bias'])
    return operation_output


def predict(date_time, prediction_type):

    year = date_time[:4]
    count = 5
    x = tf.placeholder(tf.float32, shape=[None, 3])
    path = 'E:/moodle/Level04S02/Teamfx/preprocessing'
    column_list = ['open', 'high', 'low', 'close', 'will_r', 'ema', 'rsi', 'next_close', 'next_next_close',
                   'next_next_next_close']

    if prediction_type == 'minutes':
        data_from2012_to2015 = pd.read_csv(path + '/EURUSD_M1_'+year+'_indicators_with_OHLC_rsi_ema_willr_window_10.csv')  # .iloc[:1000]
    else:
        data_from2012_to2015 = pd.read_csv(path + '/EURUSD_M60_indicators_with_OHLC_rsi_ema_willr_window_10.csv')


    data_from2012_to2015['Datetime'] = data_from2012_to2015['Datetime'].apply(lambda x: to_datetime(x))
    data_from2012_to2015.index = data_from2012_to2015.Datetime
    data_from2012_to2015 = data_from2012_to2015.drop(['Datetime'], axis=1)
    data_from2012_to2015["next_close"] = data_from2012_to2015['close'].shift(-1)
    data_from2012_to2015["next_next_close"] = data_from2012_to2015['close'].shift(-2)
    data_from2012_to2015["next_next_next_close"] = data_from2012_to2015['close'].shift(-3)
    data_from2012_to2015 = data_from2012_to2015.dropna()

    labels_before_nomalized = data_from2012_to2015[['next_close']]
    history_labels_before_normalized = labels_before_nomalized.ix[:to_datetime(date_time)]
    future_labels_before_normalized = labels_before_nomalized.ix[to_datetime(date_time):]
    future_labels_before_normalized = future_labels_before_normalized.iloc[1:count + 1]

    next_next_close_labels_before_nomalized = data_from2012_to2015[['next_next_close']]
    next_next_close_history_labels_before_normalized = next_next_close_labels_before_nomalized.ix[
                                                       :to_datetime(date_time)]
    next_next_close_future_labels_before_normalized = labels_before_nomalized.ix[to_datetime(date_time):]
    next_next_close_future_labels_before_normalized = next_next_close_future_labels_before_normalized.iloc[2:count + 1]

    next_next_next_close_labels_before_nomalized = data_from2012_to2015[['next_next_next_close']]
    next_next_next_close_history_labels_before_normalized = next_next_next_close_labels_before_nomalized.ix[
                                                            :to_datetime(date_time)]
    next_next_next_future_labels_before_normalized = labels_before_nomalized.ix[to_datetime(date_time):]
    next_next_next_close_future_labels_before_normalized = next_next_next_future_labels_before_normalized.iloc[
                                                           3:count + 1]

    # min_maxed_dataset = min_max(data_from2012_to2015, 'minmaxvalues_all_data_2012_20180619', column_list=column_list)
    min_maxed_dataset = min_max(data_from2012_to2015, 'minmaxvalues_all_data_'+year, column_list=column_list)
    # min_maxed_dataset = min_max(data_from2012_to2015, 'minmaxvalues_all_data_2012', column_list=column_list).iloc[:301000]

    features = min_maxed_dataset[['close', 'ema', 'rsi']]
    labels = min_maxed_dataset[['next_close', 'next_next_close', 'next_next_next_close']]

    history_features = features.ix[:to_datetime(date_time)]  # input date is included

    history_labels = labels.ix[:to_datetime(date_time)]

    future_features = features.ix[to_datetime(date_time):]  # input date is included
    future_labels = labels.ix[to_datetime(date_time):][1:count + 1]

    df = history_features
    df_2 = df.iloc[-10:]  # bwd
    backward_date_time = df_2.index

    predictions = []

    X = history_features.iloc[-1:]
    data = history_features['close']
    data = data.tolist()

    errors = []
    predictions_next_close_label = []
    predictions_next_next_close_label = []
    predictions_next_next_next_close_label = []

    predicted_close = []

    prediction = model(x)  # operation to get predictions using the trained model.
    # saved file location to retriev Saver object
    # save_file = 'E:\\moodle\\Level04S02\\Teamfx\\weights_record\\T27-2018-06-18\\2012_nextclose_price_normalized_300000.ckpt'
    # save_file = 'E:\\moodle\\Level04S02\\Teamfx\\weights_record\\T32-2018-06-21\\2017_nextclose_price_normalized_300000_data_1000epochs_relu_3layers.ckpt'
    # save_file = 'E:\\moodle\\Level04S02\\Teamfx\\weights_record\\T34-2018-06-21\\2012_nextclose_price_normalized_300000_data_2000epochs_relu_3layers.ckpt'
    # save_file = 'E:\\moodle\\Level04S02\\Teamfx\\weights_record\\T38-2018-06-22\\2012_epochs_500_relu_next_3_close_price_for_close_ema_rsi.ckpt'
    if prediction_type == 'minutes':
        save_file = 'E:\\moodle\\Level04S02\\Teamfx\\weights_record\\T39-2018-06-23\\'+"2017"+'_epochs_1500_relu_next_3_close_price_for_close_ema_rsi.ckpt'
    else:
        save_file = 'E:\\moodle\\Level04S02\\Teamfx\\weights_record\\T43-2018-06-24\\M60_epochs_800_relu_next_3_close_price_for_close_ema_rsi.ckpt'

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, save_file)

    normalized_next_close = []
    normalized_next_next_close = []
    normalized_next_next_next_close = []

    with sess.as_default():

        # predict forward

        #more_steps = count - 3
        for i in range(count):
            predict_value = sess.run([prediction], feed_dict={x: X})
            data.append(predict_value[0][0][0])
            normalized_next_next_close.append(predict_value[0][0][1])
            normalized_next_next_next_close.append(predict_value[0][0][2])

            df = df.append({'close': predict_value[0][0][0]}, ignore_index=True)
            predicted_close.append(predict_value[0][0][0])

            window = 10
            ema = ind.expMovingAverage(data, window)
            rsi = ind.rsi_function(data, window)
            rsi = rsi.iloc[window, 0]

            df.iloc[len(df) - 1, 1] = ema  # last row 2th column
            df.iloc[len(df) - 1, 2] = rsi  # last row 3th column

            X = np.array(df)  # convert to numpy.ndarray
            X = X[len(X) - 2:len(X) - 1]

        # sess.close()

        predicted_close = invert_min_max_for_one_feature(np.array(predicted_close), 'minmaxvalues_all_data_'+year,
                                                         'next_close')
        predicted_close = predicted_close.values.tolist()
        predicted_close = [item for sublist in predicted_close for item in sublist]
        # print('predicted_close_from_ANN')
        # print(predicted_close)
        actual_values_forward = future_labels_before_normalized
        forward_date_time = actual_values_forward.index


        predicted_next_next_close = invert_min_max_for_one_feature(np.array(normalized_next_next_close),
                                                                   'minmaxvalues_all_data_'+year, 'next_next_close')
        predicted_next_next_close = predicted_next_next_close.values.tolist()
        predicted_next_next_close = [item for sublist in predicted_next_next_close for item in sublist][:-1]

        actual_next_next_value_forward = next_next_close_future_labels_before_normalized
        forward_next_next_date_time = actual_next_next_value_forward.index


        predicted_next_next_next_close = invert_min_max_for_one_feature(np.array(normalized_next_next_next_close),
                                                                        'minmaxvalues_all_data_'+year, 'next_next_close')
        predicted_next_next_next_close = predicted_next_next_next_close.values.tolist()
        predicted_next_next_next_close = [item for sublist in predicted_next_next_next_close for item in sublist][:-2]

        actual_next_next_next_value_forward = next_next_next_close_future_labels_before_normalized
        forward_next_next_next_date_time = actual_next_next_next_value_forward.index

        labels_mix_final_without_arima = []

        labels_mix_final_without_arima.append(predicted_close[0])
        labels_mix_final_without_arima.append((predicted_close[1] + predicted_next_next_close[0]) / 2)
        labels_mix_final_without_arima.append((predicted_next_next_close[1] + predicted_next_next_next_close[0]) / 2)
        labels_mix_final_without_arima.append(predicted_next_next_next_close[1])

        labels_mix_final_without_arima = pd.DataFrame(labels_mix_final_without_arima)
        labels_mix_final_without_arima.index = forward_date_time[:-1]

        # predict 10 previous errors
        j = 1
        while j <= len(df_2):
            predict_value = sess.run([prediction], feed_dict={x: df_2[j - 1:j]})
            predictions_next_close_label.append(predict_value[0][0][0])
            predictions_next_next_close_label.append(predict_value[0][0][1])
            predictions_next_next_next_close_label.append(predict_value[0][0][2])
            j += 1

        sess.close()

        actual_list = history_labels_before_normalized['next_close'].values[-10:]
        actual_list_buffer = history_labels_before_normalized['next_close'][-10:]

        actual_next_next_list = next_next_close_history_labels_before_normalized['next_next_close'].values[-10:]
        actual_next_next_list_buffer = next_next_close_history_labels_before_normalized['next_next_close'][-10:]

        actual_next_next_next_list = next_next_next_close_history_labels_before_normalized['next_next_next_close'].values[-10:]
        actual_next_next_next_list_buffer = next_next_next_close_history_labels_before_normalized['next_next_next_close'][-10:]

        actual_list = actual_list.tolist()

        predictions_list = invert_min_max_for_one_feature(np.array(predictions_next_close_label),
                                                          'minmaxvalues_all_data_'+year, 'next_close')
        predictions_list = predictions_list.values.tolist()
        predictions_list = [item for sublist in predictions_list for item in sublist]

        predictions_next_next_close_list = invert_min_max_for_one_feature(np.array(predictions_next_next_close_label),
                                                                          'minmaxvalues_all_data_'+year,
                                                                          'next_next_close')
        predictions_next_next_close_list = predictions_next_next_close_list.values.tolist()
        predictions_next_next_close_list = [item for sublist in predictions_next_next_close_list for item in sublist]

        predictions_next_next_next_close_list = invert_min_max_for_one_feature(
            np.array(predictions_next_next_next_close_label), 'minmaxvalues_all_data_'+year, 'next_next_next_close')
        predictions_next_next_next_close_list = predictions_next_next_next_close_list.values.tolist()
        predictions_next_next_next_close_list = [item for sublist in predictions_next_next_next_close_list for item in
                                                 sublist]

        errors = np.array(actual_list) - np.array(predictions_list)
        error_next_next_close = np.array(actual_next_next_list) - np.array(predictions_next_next_close_list)
        error_next_next_next_close = np.array(actual_next_next_next_list) - np.array(
            predictions_next_next_next_close_list)

    return errors, error_next_next_close, error_next_next_next_close, predicted_close, forward_date_time, predicted_next_next_close, forward_next_next_date_time, predicted_next_next_next_close, forward_next_next_next_date_time, actual_values_forward, actual_list_buffer, actual_next_next_list_buffer, actual_next_next_next_list_buffer, backward_date_time, labels_mix_final_without_arima


def predict_arima_ann(date_time, prediction_type):

    count = 5
    errors, error_next_next_close, error_next_next_next_close, predictions_from_ann, forwared_date_time, predicted_next_next_close, forward_next_next_date_time, predicted_next_next_next_close, forward_next_next_next_date_time, actual_values_forward, actual_close, actual_next_next_close, actual_next_next_next_close, backward_date_time, labels_mix_final_without_arima = predict(date_time, prediction_type)

    series_error = pd.Series(errors.tolist())
    series_pct_change = series_error
    # here in error negative positive points are important then log can not be used here and pct change is used.

    X = series_pct_change.values.tolist()

    history = [x for x in X]

    model = ARIMA(history, order=(2, 0, 0))
    model_fit = model.fit(disp=0)

    output = model_fit.forecast(steps=count)[0].tolist()

    final_results = np.array(predictions_from_ann) + np.array(output)
    final_results = final_results.tolist()
    # error = mean_squared_error(test, output)
    # print('Test MSE: %.3f' % error)
    final_results = pd.DataFrame({'final_results': final_results})

    series_next_next_error = pd.Series(error_next_next_close.tolist())  # .map(lambda x: x*1000)
    # series_pct_change = series_error  # series_error.pct_change().dropna()

    # here in error negative positive points are important then log can not be used here and pct change is used.

    X_ = series_next_next_error.values.tolist()

    history_next_next = [x for x in X_]

    model = ARIMA(history_next_next, order=(2, 0, 0))
    model_fit = model.fit(disp=0)

    arima_output_for_next_next_close = model_fit.forecast(steps=count - 1)[0].tolist()

    final_results_next_next_close = np.array(predicted_next_next_close) + np.array(arima_output_for_next_next_close)
    final_results_next_next_close = final_results_next_next_close.tolist()
    # error = mean_squared_error(test, output)
    # print('Test MSE: %.3f' % error)
    final_results_next_next_close = pd.DataFrame({'final_results_next_next_close': final_results_next_next_close})


    series_next_next__next_error = pd.Series(error_next_next_next_close.tolist())  # .map(lambda x: x*1000)
    # series_pct_change = series_error  # series_error.pct_change().dropna()

    # here in error negative positive points are important then log can not be used here and pct change is used.

    X__ = series_next_next__next_error.values.tolist()

    history_next_next_next = [x for x in X__]

    model = ARIMA(history_next_next_next, order=(2, 0, 0))
    model_fit = model.fit(disp=0)

    arima_output_for_next_next_next_close = model_fit.forecast(steps=count - 2)[0].tolist()

    final_results_next__next_next_close = np.array(predicted_next_next_next_close) + np.array(arima_output_for_next_next_next_close)
    final_results_next__next_next_close = final_results_next__next_next_close.tolist()
    # error = mean_squared_error(test, output)
    # print('Test MSE: %.3f' % error)
    final_results_next__next_next_close = pd.DataFrame(
        {'final_results_next__next_next_close': final_results_next__next_next_close})


    predictions_from_ann = pd.DataFrame(predictions_from_ann)  # next_close label prediction form ANN only
    predictions_from_ann.index = forwared_date_time

    predicted_next_next_close = pd.DataFrame(
        predicted_next_next_close)  # next_next_close label prediction form ANN only
    predicted_next_next_close.index = forward_next_next_date_time

    predicted_next_next_next_close = pd.DataFrame(
        predicted_next_next_next_close)  # next_next_next_close label prediction form ANN only
    predicted_next_next_next_close.index = forward_next_next_next_date_time

    next_close_ann_arima = final_results.values.tolist()
    f_n = [item for sublist in next_close_ann_arima for item in sublist]  # [x for x in next_close_ann_arima]
    next_next_close_ann_arima = final_results_next_next_close.values.tolist()
    f_n_n = [item for sublist in next_next_close_ann_arima for item in sublist]
    next_next_next_close_ann_arima = final_results_next__next_next_close.values.tolist()
    f_n_n_n = [item for sublist in next_next_next_close_ann_arima for item in sublist]


    labels_mix_final = []
    # labels_mix_final.append( f_n[0])
    # labels_mix_final.append((f_n[1] + f_n_n[0])/2)
    # labels_mix_final.append((f_n[2] + f_n_n[1] + f_n_n_n[0]) / 3)
    # labels_mix_final.append((f_n[3] + f_n_n[2] + f_n_n_n[1]) / 3)
    # labels_mix_final.append((f_n[4] + f_n_n[3] + f_n_n_n[2]) / 3)

    labels_mix_final.append(f_n[0])
    labels_mix_final.append((f_n[1] + f_n_n[0]) / 2)
    labels_mix_final.append((f_n_n[1] + f_n_n_n[0]) / 2)
    labels_mix_final.append(f_n_n_n[1])

    # j = 1
    # while j <= count:
    #
    #     predictions_next_close_label.append(predict_value[0][0][0])
    #     predictions_next_next_close_label.append(predict_value[0][0][1])
    #     predictions_next_next_next_close_label.append(predict_value[0][0][2])
    #     j += 1
    # print('labels_mix_final')
    # print(labels_mix_final)

    error_of_ANN_ARIMA_3_labels = []
    actual_value = actual_values_forward.values.tolist()
    actual_value = [item for sublist in actual_value for item in sublist][:-1]
    # print(actual_value)
    # print(labels_mix_final)
    error_of_ANN_ARIMA_3_labels = np.array(actual_value) - np.array(labels_mix_final)
    # print(error_of_ANN_ARIMA_3_labels)
    error_of_ANN_ARIMA_3_labels  = pd.DataFrame(error_of_ANN_ARIMA_3_labels)
    error_of_ANN_ARIMA_3_labels.index = forwared_date_time[:-1]
    # print(error_of_ANN_ARIMA_3_labels)
    error_of_ANN_ARIMA_3_labels.to_csv(date_time[:10]+'.csv')

    labels_mix_final = pd.DataFrame(labels_mix_final)
    labels_mix_final.index = forwared_date_time[:-1]

    final_results = pd.DataFrame(final_results)
    final_results.index = forwared_date_time

    final_results_next_next_close = pd.DataFrame(final_results_next_next_close)
    final_results_next_next_close.index = forward_next_next_date_time

    final_results_next__next_next_close = pd.DataFrame(final_results_next__next_next_close)
    final_results_next__next_next_close.index = forward_next_next_next_date_time

    a2_values = np.array(actual_values_forward['next_close'].values)
    a2_values = np.insert(a2_values, 0, actual_close.iloc[-1], axis=0)
    a2_indices = actual_values_forward.index
    a2_indices = np.insert(a2_indices, 0, actual_close.index[-1], axis=0)

    a3_values = np.array(predictions_from_ann[0].values)
    a3_values = np.insert(a3_values, 0, actual_close.iloc[-1], axis=0)
    a3_indices = predictions_from_ann.index
    a3_indices = np.insert(a3_indices, 0, actual_close.index[-1], axis=0)

    a4_values = np.array(labels_mix_final[0].values)
    a4_values = np.insert(a4_values, 0, actual_close.iloc[-1], axis=0)
    a4_indices = labels_mix_final.index
    a4_indices = np.insert(a4_indices, 0, actual_close.index[-1], axis=0)

    a5_values = np.array(final_results['final_results'].values)
    a5_values = np.insert(a5_values, 0, actual_close.iloc[-1], axis=0)
    a5_indices = final_results.index
    a5_indices = np.insert(a5_indices, 0, actual_close.index[-1], axis=0)

    a6_values = np.array(labels_mix_final_without_arima[0].values)
    a6_values = np.insert(a6_values, 0, actual_close.iloc[-1], axis=0)
    a6_indices = final_results.index
    a6_indices = np.insert(a6_indices, 0, actual_close.index[-1], axis=0)

    # plt.plot(predicted_next_next_close, color='yellow')
    #plt.plot(final_results_next_next_close, color='yellow', linestyle='dashed')
    # plt.plot(predicted_next_next_next_close, color='cyan')
    #plt.plot(final_results_next__next_next_close, color='cyan', linestyle='dashed')

    # plt.plot(predictions_from_ann, color='green')
    # plt.plot(final_results, color='red', linestyle='dashed')
    # plt.plot(actual_close, color='blue')
    # plt.plot(actual_values_forward, color='blue', linestyle='dashed')
    # plt.plot(labels_mix_final, color='red')
    # plt.title(date_time)
    # plt.show()

    graphs = [
        dict(
            data=[
                dict(
                    x=actual_close.index,
                    y=actual_close.values,
                    type='scatter',
                    legendgroup="G1",
                    name='History Close Price'
                ),
                dict(
                    x=a2_indices[:-1],
                    y=a2_values[:-1],
                    type='scatter',
                    legendgroup="G1",
                    name='Expected close prices',
                    line=dict(
                        color= ('rgb(22, 96, 167)'),
                        #width=4,
                        dash='dot'  # 'dash' 'dashdot'
                    )
                ),
                dict(
                    x=a5_indices[:-1],
                    y=a5_values[:-1],
                    type='scatter',
                    legendgroup="G2",
                    name='Predicted Model with DRNN & ARIMA 1 label',
                    line=dict(
                        color=('rgb(0, 102, 0)'),
                        # width=4,
                        #dash='dash'  # 'dot' 'dashdot'
                    )

                ),
                dict(
                    x=a3_indices[:-1],
                    y=a3_values[:-1],
                    type='scatter',
                    legendgroup="G2",
                    name='Predicted Model Results without ARIMA 1 label',
                    line=dict(
                        color=('rgb(0, 230, 0)'),
                        # width=4,
                        dash='dot'  # 'dash' 'dashdot'
                    )
                ),
                dict(
                    x=a4_indices,
                    y=a4_values,
                    type='scatter',
                    legendgroup="G3",
                    name='Predicted Model Results with DRNN & ARIMA 3 labels',
                    line=dict(
                        color=('rgb(255, 0, 0)'),
                        # width=4,
                        #dash='dash'  # 'dot' 'dashdot'
                    )
                ),
                dict(
                    x=a6_indices,
                    y=a6_values,
                    type='scatter',
                    legendgroup="G3",
                    name='Predicted Model Results without ARIMA 3 labels',
                    line=dict(
                        color=('rgb(255, 64, 0)'),
                        # width=4,
                        dash='dot'  # 'dot' 'dashdot'
                    )

                ),
            ],
            # layout=dict(
            #     title='Prediction',
            # )
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

# predict(5, get_train_features(), get_train_labels())
# predict_arima_ann()

# predict(5)
# date_time = "2016-02-14 18:21:00"
# count = 5
# predict_arima_ann(date_time)
