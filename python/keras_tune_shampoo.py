#!/usr/bin/env python3
"""
  keras_tune_shampoo.py
  This program tunes the LSTM hyperparameters for the Shampoo dataset using Keras

  [How to Tune LSTM Hyperparameters with Keras for Time Series Forecasting](https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')

def plot(data):
    # summarize first few rows
    print(data.head())
    # line plot
    data.plot()
    plt.show()


def timeseries_to_supervised(data, lag=1):
    """
    Frame a sequence as a supervised learning problem
    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df = df.drop(0)

    return df

def difference(dataset, interval=1):
    """
    Create a differenced series
    """
    diff = list()
    for i in range(interval, len(dataset)):
    	value = dataset[i] - dataset[i - interval]
    	diff.append(value)
    return pd.Series(diff)

def scale(train, test):
    """
    Scale train and test data to [-1, 1]
    """
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, yhat):
    """
    Inverse scaling for a forecasted value
    """
    new_row = [x for x in X] + [yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def evaluate(model, raw_data, scaled_dataset, scaler, offset, batch_size):
    """
    Evaluate the model on a dataset, returns RMSE in transformed units
    """
    # separate
    X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]

    # reshape
    reshaped = X.reshape(len(X), 1, 1)

    # forecast dataset
    output = model.predict(reshaped, batch_size=batch_size)

    # invert data transforms on forecast
    predictions = list()

    for i in range(len(output)):
    	yhat = output[i,0]
    	# invert scaling
    	yhat = invert_scale(scaler, X[i], yhat)
    	# invert differencing
    	yhat = yhat + raw_data[i]
    	# store forecast
    	predictions.append(yhat)

    # report performance
    rmse = math.sqrt(mean_squared_error(raw_data[1:], predictions))

    return rmse


def fit_lstm(train, test, raw, scaler, batch_size, nb_epoch, neurons):
    """
    Fit an LSTM network to training data
    """
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Define model
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit model
    train_rmse, test_rmse = list(), list()
    for i in range(nb_epoch):
    	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    	model.reset_states()

    	# evaluate model on train data
    	raw_train = raw[-(len(train)+len(test)+1):-len(test)]
    	train_rmse.append(evaluate(model, raw_train, train, scaler, 0, batch_size))
    	model.reset_states()

    	# evaluate model on test data
    	raw_test = raw[-(len(test)+1):]
    	test_rmse.append(evaluate(model, raw_test, test, scaler, 0, batch_size))
    	model.reset_states()

    history = pd.DataFrame()
    history['train'], history['test'] = train_rmse, test_rmse

    return history


def main():
    """
    Run diagnostic experiments
    """
    # load dataset
    series = pd.read_csv('../data/shampoo.csv',
                         header=0,
                         parse_dates=[0],
                         index_col=0,
                         squeeze=True,
                         date_parser=parser)

    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # fit and evaluate model
    train_trimmed = train_scaled[2:, :]

    # config
    repeats = 10
    n_batch = 4
    n_epochs = 500
    n_neurons = 1

    # run diagnostic tests
    for i in range(repeats):
    	history = fit_lstm(train_trimmed, test_scaled, raw_values, scaler, n_batch, n_epochs, n_neurons)
    	plt.plot(history['train'], color='blue')
    	plt.plot(history['test'], color='orange')
    	print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, history['train'].iloc[-1], history['test'].iloc[-1]))

    plt.savefig('epochs_diagnostic.png')


# The driver function (confirm that code is under main function)
if __name__ == "__main__":
    # load_data()

    # series = pd.read_csv('../data/shampoo.csv',
    #                      header=0,
    #                      parse_dates=[0],
    #                      index_col=0,
    #                      squeeze=True,
    #                      date_parser=parser)
    # plot(series)

    main()

    # file_path = os.path.join("../data", "pollution.csv")
    # df_data = pd.read_csv(file_path, header=0, index_col=0)
    # print(df_data.info())
    # print(df_data.head(30))

    print("Done!")
