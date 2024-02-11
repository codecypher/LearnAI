#!/usr/bin/env python3
"""
  time_series_analysis.py

  This code performs time series analysis

  [Time Series Forecast and Decomposition 101 Guide Python](https://datasciencebeginners.com/2020/11/25/time-series-forecast-and-decomposition-101-guide-python/)
"""
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_datareader as pdr
import pandas_datareader.data as web
import matplotlib.pyplot as plt

import datetime as dt
import json
import os
import datetime as dt
import json
import math
import os
import sys

from pandas import option_context
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sys import exit
from time import time


def load_data(name):
    df_data = pd.read_csv(name,
                          header=0,
                          parse_dates=['Timestamp'],
                          index_col='Timestamp')

    print(df_data.info())
    print(df_data.head())

    # Set freq attribute
    df_data = df_data.asfreq(freq='3D', method='pad')
    print('freq:', df_data.index.freq)

    # Replace timestamp with Pandas DateTime
    # df_data['date'] = pd.to_datetime(df_data['timestamp'])

    # Drop the timestamp column
    # df_data = df_data.drop('timestamp', axis='columns')

    # Move column to start of dataframe
    # first_column = df_data.pop('date')
    # df_data.insert(0, 'timestamp', first_column)

    return df_data


def boxplot(df):
    seaborn.set(style='whitegrid')
    fmri = seaborn.load_dataset("fmri")

    seaborn.boxplot(x="timepoint",
                    y="signal",
                    data=fmri)

def heatmap(df):
    """
    Plot heatmap of DataFrame
    """
    # Compute correlation matrix for numerical features
    corrs = df.corr()

    # Plot heatmp
    plt.figure(figsize=(12,8))
    sns.heatmap(corrs, cmap='RdBu_r', annot=True)
    plt.show()


def pca(X_train_scaled):
    pca = PCA().fit(X_train_scaled)
    plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
    plt.title('Cumulative variance by number of principal components', size=12)
    plt.show()


def plot_data(df):
    plt.rcParams["figure.figsize"] = (12,6)
    df.plot()
    plt.show()


def plot_quarter(df):
    """
    Plot the trend looks at a quarterly level
    """
    df["date"] = df["Month"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m"))
    df["year"] = df["date"].apply(lambda x: x.year)
    df["qtr"] = df["date"].apply(lambda x: x.quarter)
    df["yearQtr"] = df['year'].astype(str) + '_' + df['qtr'].astype(str)
    airPassengerByQtr = df[["Passengers", "yearQtr"]].groupby(["yearQtr"]).sum()

    plt.rcParams["figure.figsize"] = (14,6)
    plt.title("Total number of passengers by quarter")
    plt.plot(airPassengerByQtr)
    plt.xticks(airPassengerByQtr.index, rotation='vertical')
    plt.show()


def plot_decomp(df):
    """
    Plot dataset as a multiplicative model
    Classical Decomposition
    """
    data_s = df['market-price']
    result = seasonal_decompose(data_s, model='multiplicative')
    plt.rcParams["figure.figsize"] = (10,8)
    result.plot()
    plt.show()


def plot_stl(df):
    """
    Plot Seasonal and Trend decomposition using Loess (STL)
    STL Decomposition
    STL is robust to outliers and can handle any kind of seasonality.
    """
    stl = STL(df, period=4, robust=True)
    res_robust = stl.fit()
    plt.rcParams["figure.figsize"] = (10,8)
    fig = res_robust.plot()
    plt.show()


# Check that code is under main function
if __name__ == "__main__":
    # concat_files('../data/drift3')
    file_path = "../data/airline_passengers.csv"
    # file_path = "../data/bitcoin.csv"
    df = load_data(file_path)

    # plot_data(df)
    # plot_quarter(df)
    plot_decomp(df)
    # plot_stl(df)

    print('Done')
