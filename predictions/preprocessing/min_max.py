import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#data = pd.read_csv('EURUSD_M1_2012_indicators_with_residualwithnew.csv')
#data.index = data.Datetime
#data = data.drop(['Datetime', 'close'], axis=1)

# return the min max normalized data series for the given dataset


def min_max_predict(dataset , descrbe_file_name, column_list=['will_r', 'ema', 'rsi', 'residual']):
    d_describe = pd.read_csv(descrbe_file_name+'.csv')
    print(d_describe)
    min_values = d_describe.iloc[3].values[1:]
    max_values = d_describe.iloc[7].values[1:]
    no_of_attribute = dataset.ndim - 1

    min_max_array = []

    for i in range(no_of_attribute):
        min_max_value = (dataset.iloc[i] - (min_values[i]) )/( max_values[i] - min_values[i])
        min_max_array.append(min_max_value)

    print(min_max_array)

    df_minmax_scalar = pd.DataFrame(min_max_array, columns=column_list)
    df_minmax_scalar['datetime'] = dataset.index
    df_minmax_scalar.index = df_minmax_scalar.datetime
    df_minmax_scalar = df_minmax_scalar.drop(['datetime'], axis=1)

    print("inside min_max_predic", df_minmax_scalar)

    return  min_max_array

    # d_describe = dataset.describe()
    # d_describe.read(descrbe_file_name+'.csv')
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # rescaled = scaler.fit_transform(dataset)
    # df_minmax_scalar = pd.DataFrame(rescaled, columns=column_list)
    # df_minmax_scalar['datetime'] = dataset.index
    # df_minmax_scalar.index = df_minmax_scalar.datetime
    # df_minmax_scalar = df_minmax_scalar.drop(['datetime'],axis=1)
    # return df_minmax_scalar

def min_max(dataset , descrbe_file_name, column_list=['will_r', 'ema', 'rsi', 'residual']):
    d_describe = dataset.describe()
    d_describe.to_csv("E:/moodle/Level04S02/git/teamfx/predictions/data/"+descrbe_file_name+'.csv')
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled = scaler.fit_transform(dataset)
    df_minmax_scalar = pd.DataFrame(rescaled, columns=column_list)
    df_minmax_scalar['datetime'] = dataset.index
    df_minmax_scalar.index = df_minmax_scalar.datetime
    df_minmax_scalar = df_minmax_scalar.drop(['datetime'],axis=1)
    return df_minmax_scalar

def min_max_simple(val, min, max):
    return (val-min)/(max-min)


# return  invert normalized dataset
#data = pd.read_csv('create_csv_nn_expected_and_predicted_5000.csv')
#data.index = data.Datetime
#data = data.drop(['Datetime', 'close'], axis=1)


def invert_min_max(minmaxed, descrbe_file_name):# only on data raw invert
    d_describe = pd.read_csv("E:/moodle/Level04S02/git/teamfx/predictions/data/"+descrbe_file_name+'.csv')
    #d_describe.index = d_describe['Unnamed: 0']
    #d_describe = d_describe.drop(['Unnamed: 0'], axis=1)
    min_values = d_describe.loc[d_describe.index == 'min'].values
    max_values = d_describe.loc[d_describe.index == 'max'].values
    no_of_attribute = minmaxed.ndim - 1

    reverse_value_array = []
    for i in range(no_of_attribute):
        reverse_value = minmaxed.item(i) * (max_values.item(i) - min_values.item(i)) + min_values.item(i)
        reverse_value_array.append(reverse_value)

    return  reverse_value_array

def invert_min_max_residual(minmaxed, describe_file_name):
    d_describe = pd.read_csv(describe_file_name + '.csv')
    d_describe = d_describe['residual'].values
    print(minmaxed)
    # d_describe.index = d_describe['Unnamed: 0']
    # d_describe = d_describe.drop(['Unnamed: 0'], axis=1)
    min_values = d_describe[3]
    max_values = d_describe[7]
    no_of_attribute = minmaxed.size

    inverted_residual_array = []
    for i in range(no_of_attribute):
        inverted_residual_value = minmaxed[i] * (max_values - min_values) + min_values
        inverted_residual_array.append(inverted_residual_value)

    return inverted_residual_array

# minmaxed input should be pd.series
def invert_min_max_for_one_feature(minmaxed, describe_file_name, feature):
    d_describe = pd.read_csv("E:/moodle/Level04S02/git/teamfx/predictions/data/"+describe_file_name + '.csv')
    d_describe = d_describe[feature].values
    min_values = d_describe[3]
    max_values = d_describe[7]
    no_of_attribute = minmaxed.size

    inverted_feature_array = []
    for i in range(no_of_attribute):
        inverted_feature_value = minmaxed[i] * (max_values - min_values) + min_values
        inverted_feature_array.append(inverted_feature_value)

    return pd.DataFrame(inverted_feature_array)

def invert_min_max_all(minmaxed, descrbe_file_name):# all invert
    d_describe = pd.read_csv(descrbe_file_name+'.csv')
    d_describe.index = d_describe['Unnamed: 0']
    d_describe = d_describe.drop(['Unnamed: 0'], axis=1)
    min_values = d_describe.loc[d_describe.index == 'min'].values
    max_values = d_describe.loc[d_describe.index == 'max'].values
    no_of_attribute = min_values.size
    array =[]
    reverse_value_array = []
    for j in range(minmaxed.count()):
        for i in range(no_of_attribute):
            reverse_value = minmaxed[0].item(i) * (max_values.item(i) - min_values.item(i)) + min_values.item(i)
            reverse_value_array.append(reverse_value)
        array.append(reverse_value_array)
    return  array
#invert_min_max_all(data, 'minmaxvalues_first_5000_data')

def convert_min_max(data, descrbe_file_name):
    d_describe = pd.read_csv(descrbe_file_name + '.csv')
    d_describe.index = d_describe['Unnamed: 0']
    d_describe = d_describe.drop(['Unnamed: 0'], axis=1)
    min_values = d_describe.loc[d_describe.index == 'min'].values
    max_values = d_describe.loc[d_describe.index == 'max'].values
    no_of_attribute = min_values.size
    convert_minmax_array = []
    for i in range(no_of_attribute):
        convert_minmax = (data.as_matrix().item(i) - min_values.item(i))/ (max_values.item(i) - min_values.item(i))
        convert_minmax_array.append(convert_minmax)
    return convert_minmax_array
"""
# selecting the relavent raw that has to normalized
min_maxed_dataset = min_max(data,'minmax_describe_value')
selected_raw = min_maxed_dataset.loc[min_maxed_dataset.index == '2012-01-02 02:11:00'].values

# Call the reverse_min_max() function
print(invert_min_max(selected_raw , 'minmax_describe_value'))
"""

