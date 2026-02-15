#!/usr/bin/env python3
"""
eda.py
This program contains code snippets for Exploratory Data Analysis (EDA)
"""
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import option_context
from sklearn import metrics

# import numpy as np
# import pandas_datareader as pdr
# import pandas_datareader.data as web

# import json
# import math
# import os
# import sys

# from sys import exit
# from time import time

# from dataprep.datasets import load_dataset
# from dataprep.eda import create_report

from statsmodels.tsa.seasonal import seasonal_decompose

# from statsmodels.stats import weightstats as stests
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.stats.stattools import jarque_bera
# from statsmodels.stats.stattools import omni_normtest as omb
# from statsmodels.compat import lzip


def parser(x):
    """
    This function is used as date parser when using read_csv
    """
    return dt.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f")


def load_data(name):
    data_df = pd.read_csv(
        name,
        header=0,
        # index_col=0,
        parse_dates=["timestamp"],
        # date_parser=parser,
    )

    # Rename index
    data_df.index.name = "row"

    return data_df


def plot(df):
    """
    This function plots each series (column) as a separate subplot,
    except wind speed dir which is categorical.
    """
    # Plot each column
    values = df.values

    # Specify columns to plot
    groups = range(len(df.columns))

    # Plot each column
    # fig = plt.figure(figsize=(8, 10))
    for i, grp in enumerate(groups):
        plt.subplot(len(groups), 1, i + 1)
        plt.plot(values[:, grp])
        plt.title(df.columns[grp], y=0.5, loc='right')

    # Save image and return fig
    # fig.savefig("box_plot.png")

    # plt.show()

    # return fig


def explore(df, title='Plot each column'):
    """
    Perform EDA on dataset
    """
    # Temporary assignment of settings
    # with option_context('display.max_columns', 25, 'display.float_format', '{:.8f}'.format):
    #     print(df.info())
    #     print(df.describe())

    # Summary statistics of categorical features
    # data_df.describe(include='object')

    # Plot each numeric feature using histogram
    df.hist(figsize=(10, 10), xrot=45)

    # Plot feature using density plot to give more clear summary
    # of the distribution of observations.
    # df['nbdi'].plot(figsize=(6, 6), kind='kde')

    # series = df['nbdi']
    # groups = series.groupby(Grouper(freq='A'))
    # years = pd.DataFrame()
    # for name, group in groups:
    # 	years[name.year] = group.values
    # years.boxplot()

    # Plot boxplot of each categorical feature with Price
    # for column in data.select_dtypes(include=’object’):
    #     if data[column].nunique() < 10:
    #         sns.boxplot(y=column, x=’Price’, data=data)

    # Plot histogram with log transform of the time series
    # X = df['nbdi'].values
    # X = np.log(X)
    # plt.hist(X)
    # plt.plot(X)

    # Plot each column
    # values = df.values
    # groups = range(len(df.columns))
    # plt.style.use("fivethirtyeight")
    # df.plot(subplots=True, figsize=(10, 12))

    # Plot each categorical feature
    # for column in data.select_dtypes(include='object'):
    #     if data[column].nunique() < 10:
    #         sns.countplot(y=column, data=data)
    #         plt.show()

    # fig = plt.figure(figsize=(12, 8))

    sns.lineplot(data=df['nbdi'])
    sns.lineplot(data=df['temp'])

    # plt.ylabel("Col_1 and Col_2")
    plt.xticks(rotation=25)
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, y_hat):
    # Evaluate Model Accuracy on Test Data
    score = model.score(X_test, y_test)

    # Show confusion matrix
    cm = metrics.confusion_matrix(y_test, y_hat)
    # print(cm)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()


def plot_decomp(df):
    """
    Plot dataset as a multiplicative model
    Classical Decomposition
    """
    result = seasonal_decompose(df, model='multiplicative')
    plt.rcParams["figure.figsize"] = (10, 8)
    result.plot()
    # plt.show()


def heatmap(df):
    """
    Plot heatmap
    """
    # Compute correlation matrix for numerical features
    corrs = df.corr()

    # Plot heatmp
    plt.figure(figsize=(20, 12))
    sns.heatmap(corrs, cmap='RdBu_r', annot=True)
    plt.show()


def main():
    df = load_data("../data/BEUTB-E-UST-301.csv")
    with option_context('display.max_columns', 25, 'display.float_format', '{:.8f}'.format):
        print(df.info())
        # print(df.head())

    # report = create_report(df, title='Dijon Report')
    # report.save(filename='report_dijon', to='.')


# Check that code is under main function
if __name__ == "__main__":
    main()
    print('Done')
