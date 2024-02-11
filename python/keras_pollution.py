#!/usr/bin/env python3
"""
  keras_pollution.py
  This program implements Multivariate LSTM for the Pollution dataset using Keras.

  [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
  [Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math
import os

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.seasonal import seasonal_decompose, STL

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def parser(x):
    return datetime.strptime(x, "%Y %m %d %H")


def load_data(name):
    """
    Load data from CSV
    """
    file_path = os.path.join("../data", name)

    # Consolidate the date-time info into single value
    data_df = pd.read_csv(
        file_path,
        header=0,
        index_col=0,
        parse_dates=[["year", "month", "day", "hour"]],
        date_parser=parser,
        delimiter=",",
    )

    # data_df = pd.read_csv(name,
    #                       header=0,
    #                       index_col=0,
    #                       parse_dates=["date"],
    #                       delimiter=",")

    # Drop the No column
    data_df.drop("No", axis=1, inplace=True)

    # Rename columns
    data_df.columns = [
        "pollution",
        "dew",
        "temp",
        "press",
        "wnd_dir",
        "wnd_spd",
        "snow",
        "rain",
    ]
    data_df.index.name = "date"

    # Mark all NA values with 0
    data_df["pollution"].fillna(0, inplace=True)

    # Drop the first 24 hours
    data_df = data_df[24:]

    # One-hot encode column
    # data_df['wnd_dir'] = pd.get_dummies(data_df['wnd_dir'])

    # Summarize first 5 rows
    print(data_df.head(5))

    # Save to file
    # data_df.to_csv("../data/pollution.csv")

    return data_df


def plot(df):
    """
    This function plots each series (column) as a separate subplot
    except wnd_dir which is categorical.
    """
    values = df.values
    groups = range(len(df.columns))

    fig = plt.figure(figsize=(10, 8))

    # Plot each column (5 years of data for each variable)
    i = 1
    for index, name in enumerate(df.columns):
        if name != "wnd_dir":
            plt.subplot(len(df.columns), 1, i)
            plt.plot(values[:, index])
            plt.title(df.columns[index], y=0.5, loc="right")
            i += 1

    plt.show()

    # Save image
    # fig.savefig("plot_pollution.png")


def explore(df):
    """
    Perform EDA on dataset
    """
    print("\n", df.describe())

    # Summary statistics of categorical features
    print("\n", df.describe(include="object"))

    # Plot each column
    # values = df.values
    # groups = range(len(df.columns))
    # plt.style.use("fivethirtyeight")
    # df.plot(subplots=True, figsize=(10, 12))

    # Plot each categorical feature
    # for column in data.select_dtypes(include='object'):
    #     if data[column].nunique() < 10:
    #         sns.countplot(y=column, data=data)

    # Plot using seaborn
    # fig = plt.figure(figsize=(12,8))

    # sns.lineplot(data=df['pollution'])
    # sns.lineplot(data=df['temp'])

    # # plt.ylabel("Col_1 and Col_2")
    # plt.xticks(rotation = 25)

    # plt.show()


def plot_decomp(df):
    """
    Plot dataset as a multiplicative model
    Classical Decomposition
    """
    result = seasonal_decompose(df, model="multiplicative")
    plt.rcParams["figure.figsize"] = (10, 8)
    result.plot()
    plt.show()


def heatmap(df):
    """
    Plot heatmap
    """
    # Compute correlation matrix for numerical features
    corrs = df.corr()

    # Plot heatmp
    plt.figure(figsize=(20, 12))
    sns.heatmap(corrs, cmap="RdBu_r", annot=True)
    plt.show()


def train_test_split(df, pct_split=0.8):
    # idx = df.columns.get_loc("target_col")
    n_rows = df.shape[0]
    idx_split = math.ceil(n_rows * pct_split)
    print(n_rows, idx_split)

    row = df.iloc[idx_split - 1]
    val = str(row.name)
    pieces = [x.strip() for x in val.split()]
    date_split = pieces[0]

    train_df = df.query("index <= @date_split")
    test_df = df.query("index > @date_split")

    train_df.to_csv("../data/pollution_train.csv")
    test_df.to_csv("../data/pollution_test.csv")


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Convert series to supervised learning
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def train_one_timestep():
    """
    Train for one timestep
    """
    # Load dataset
    file_path = os.path.join("../data", "pollution.csv")
    df_data = pd.read_csv(file_path, header=0, index_col=0)
    print(df_data.head())

    values = df_data.values

    # Integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])

    # Ensure all data is float
    values = values.astype("float32")

    # Mormalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # Drop columns we do not want to predict
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframed.head())

    # Split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # Split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # Reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # Design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss="mae", optimizer="adam")

    # Fit network
    history = model.fit(
        train_X,
        train_y,
        epochs=50,
        batch_size=72,
        validation_data=(test_X, test_y),
        verbose=2,
        shuffle=False,
    )

    # plot history
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.legend()
    plt.show()

    # Make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # Invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # Invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print("Test RMSE: %.3f" % rmse)


def main(epochs=10, batch_size=72):
    """
    Train On multiple lag timesteps
    """
    # load dataset
    file_path = os.path.join("../data", "pollution.csv")
    df_data = pd.read_csv(file_path, header=0, index_col=0)
    values = df_data.values

    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])

    # ensure all data is float
    values = values.astype("float32")

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # specify the number of lag hours
    n_hours = 3
    n_features = 8

    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    print("reframed: ", reframed.shape)

    # split into train and test sets
    # To speed up the training of the model for this demonstration,
    # we will only fit the model on the first year of data
    # then evaluate it on the remaining 4 years of data.
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print("input/output: ", train_X.shape, len(train_X), train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(
        "[samples, timesteps, features]: ",
        train_X.shape,
        train_y.shape,
        test_X.shape,
        test_y.shape,
    )

    # exit()

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss="mae", optimizer="adam")

    # fit network
    history = model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_X, test_y),
        verbose=2,
        shuffle=False,
    )

    # plot history
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.legend()
    plt.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

    # invert scaling for forecast
    # we concatenate the yhat column with the last 7 features
    # of the test dataset in order to inverse the scaling
    inv_yhat = np.concatenate((yhat, test_X[:, -(n_features - 1) :]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -(n_features - 1) :]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print("\nTest RMSE: %.3f" % rmse)


# The driver function (confirm that code is under main function)
if __name__ == "__main__":
    df = load_data("beijing_pollution_raw.csv")

    # plot(df)
    explore(df)

    # main()

    # file_path = os.path.join("../data", "pollution.csv")
    # df_data = pd.read_csv(file_path, header=0, index_col=0)
    # print(df_data.info())
    # print(df_data.head(30))

    print("\nDone!")
