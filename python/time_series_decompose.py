#!/usr/bin/env python3
"""
    time_series_decompose.py

    [How to Decompose Time Series Data into Trend and Seasonality](time_series_decomposition.md)

    A useful abstraction for selecting forecasting methods is to decompose (break)
    a time series into systematic and unsystematic components.

    Time Series Components

    A given time series is thought to consist of three systematic components:
    level, trend, and seasonality, plus one non-systematic component called noise.

    These components are defined as follows:

    - Level: The average value in the series.
    - Trend: The increasing or decreasing value in the series.
    - Seasonality: The repeating short-term cycle in the series.
    - Noise: The random variation in the series.

    A time series is thought to be an aggregate or combination of these four components.

    All series have a level and noise. The trend and seasonality components are optional.

    It is also helpful to think of the components as combining either
    additive or multiplicative.

    Reference:
      [How to Decompose Time Series Data into Trend and Seasonality](machinelearningmastery.com)
      [Analyzing time series data in Pandas](towardsdatascience.com)
      [Seasonal Decompose](datacamp.com)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json

from statsmodels.tsa.seasonal import seasonal_decompose


def plot_raw_data(name):
    """
    Graph the raw observations
    """
    series = pd.read_csv(name, header=0, index_col=0)
    print(series.info())
    print(series.head())
    series.plot()
    plt.show()


def plot_mult_model(name):
    """
    Decompose the airline passenger dataset as a multiplicative model
    """
    series = pd.read_csv(name, header=0, index_col=0)
    print(series.info())
    print(series.head())

    # Since the data is monthly you will guess that the seasonality is 12 time periods,
    # but this will not always be the case.
    result = seasonal_decompose(series, model='multiplicative', period=12)
    result.plot()
    plt.show()

    # Seasonal decomposition using moving averages.
    # decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
    # fig = decomposition.plot()
    # plt.show()

    # df = pd.DataFrame(df, columns=['open', 'close', 'high', 'low'])
    # df = df.drop('volume', axis=1)
    # df['close'].plot()
    # plt.show()


# The driver function (confirm that code is under main function)
if __name__ == "__main__":
    name = '../data/airline_passengers.csv'
    plot_raw_data(name)
    # plot_mult_model(name)
