#!/usr/bin/env python3
"""
  keras_weather.py
  Timeseries forecasting for weather prediction

  This program demonstrates how to perform Multivariate Time Series Forecasting using LSTM with Keras.

  References:

  [Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)

  [Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
  [Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)
  [Timeseries classification with a Transformer model](https://keras.io/examples/timeseries/timeseries_classification_transformer/)
  [Timeseries anomaly detection using an Autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection/)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import tensorflow as tf
# import tensorflow_datasets as tfds

from pprint import pprint

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from zipfile import ZipFile

# Define global variables

epochs = 10
batch_size = 256
learning_rate = 0.001

split_fraction = 0.715

# The model is shown data for past 5 days (24 * 6 * 5 = 720 observations) 
# that are sampled every hour.
# The temperature after 72 (12 hours * 6 observation per hour) 
# observations will be used as a label.
past = 720
future = 72
step = 6

titles = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

date_time_key = "Date Time"


def load_data():
    uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
    zip_file = ZipFile(zip_path)
    zip_file.extractall()
    csv_path = "jena_climate_2009_2016.csv"

    df = pd.read_csv(csv_path)

    return df


def normalize(data, train_split):
    """
    Since every feature has values with varying ranges, we do normalization to confine
    feature values to the range of [0, 1] before training a neural network.

    We do this by subtracting the mean and dividing by the standard deviation of each feature.
    """
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


def data_prep(df):
    """
    Perform data preprocessing

    Here we choose ~300,000 data points for training using the split_fraction global variable
    Thus, 71.5% of the data will be used to train the model (300,693 rows).
    
    An observation is recorded every 10 mins which means 6 times per hour.
    We will resample one point per hour since no drastic change is expected within 60 minutes.
    We do this via the `sampling_rate argument` in the `timeseries_dataset_from_array` utility.

    We are tracking data from the past 720 timestamps (720/6 = 120 hours).
    This data will be used to predict the temperature after 72 timestamps (72/6 = 12 hours).
    
    The model is shown data for the past 5 days (24 * 6 * 5 = 720 observations)
    that are sampled every hour.

    The temperature after 72 (12 hours * 6 observations per hour) observations will be
    used as a label.
    """
    train_split = int(split_fraction * int(df.shape[0]))

    print(
        "The selected parameters are:",
        ", ".join([titles[i] for i in [0, 1, 5, 7, 8, 10, 11]]),
    )

    selected_features = [feature_keys[i] for i in [0, 1, 5, 7, 8, 10, 11]]
    features = df[selected_features]
    features.index = df[date_time_key]
    features.head()

    features = normalize(features.values, train_split)
    features_df = pd.DataFrame(features)

    return features_df


def train_test_split(df):
    """
    Perform train-test split
    """
    train_split = int(split_fraction * int(df.shape[0]))

    train_data = df.loc[0 : train_split - 1]
    val_data = df.loc[train_split:]

    # Training dataset
    # The training dataset labels start from the 792nd observation (720 + 72).
    start = past + future
    end = start + train_split

    x_train = train_data[[i for i in range(7)]].values
    y_train = df.iloc[start:end][[1]]

    sequence_length = int(past / step)

    # Validation dataset
    # The validation dataset must not contain the last 792 rows
    # since we will not have label data for those records,
    # so 792 must be subtracted from the end of the data.
    # The validation label dataset must start from 792 after train_split,
    # so we must add past + future (792) to label_start.
    x_end = len(val_data) - past - future

    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
    y_val = df.iloc[label_start:][[1]]

    print(train_data.shape)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    exit()

    return x_train, y_train, x_val, y_val


def create_datasets(x_train, y_train, x_val, y_val):
    """
    Create train and validation datasets
    """
    sequence_length = int(past / step)

    print('sequence_length:', sequence_length)
    print('past:', past)
    print('future:', future)
    print('step:', step)
    print('batch_size:', batch_size)
    exit()

    # The timeseries_dataset_from_array function takes in a sequence of data-points
    # gathered at equal intervals along with time series parameters such as length of
    # the sequences/windows, spacing between two sequence/windows, etc. to produce
    # batches of sub-timeseries inputs and targets sampled from the main timeseries.
    train_ds = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    val_ds = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    return train_ds, val_ds


def plot_loss(history, title):
    """
    We can visualize the loss with the function below.
    After one point, the loss stops decreasing.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_prediction(plot_data, delta, title):
    """
    Plot predictions for 5 sets of values from validation set.
    """
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()


def plot_raw_data(data):
    """
    Raw Data Visualization
    To give us a sense of the data we are working with, each feature has been plotted below.
    This shows the distinct pattern of each feature over the time period from 2009 to 2016.
    It also shows where anomalies are present which will be addressed during normalization.
    """
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()
    # plt.show()


def plot_heatmap(data):
    """
    This heat map shows the correlation between different features.
    We can see from the correlation heatmap, few parameters like Relative Humidity and
    Specific Humidity are redundant. Hence we will be using select features, not all.
    """
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    # plt.show()


def create_model(inputs, learning_rate):
    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    return model


def main():
    df = load_data()
    data_df = data_prep(df)
    x_train, y_train, x_val, y_val = train_test_split(data_df)

    print('Train:', x_train.shape, y_train.shape)

    train_ds, val_ds = create_datasets(x_train, y_train, x_val, y_val)

    # plot_raw_data(df)
    # plot_heatmap(df)

    for batch in train_ds.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)


    # Training
    model = create_model(inputs, learning_rate)

    print(model.summary())

    # We use the `ModelCheckpoint` callback to regularly save checkpoints and
    # the `EarlyStopping` callback to interrupt training when the validation loss
    # is not longer improving.
    path_checkpoint = "../output/lstm_checkpoint.h5"
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                min_delta=0,
                                                patience=5)

    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[es_callback, modelckpt_callback],
    )

    plot_loss(history, "Training and Validation Loss")

    # The trained model above is now able to make predictions
    # for 5 sets of values from validation set.
    for x, y in val_ds.take(5):
        plot_prediction(
            [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
            12,
            "Single Step Prediction",
        )

# The driver function (confirm that code is under main function)
if __name__ == "__main__":
    main()

    print("\nDone!")
