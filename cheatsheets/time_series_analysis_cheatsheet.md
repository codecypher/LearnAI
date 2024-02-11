# Time Series Analysis


## Background

[How to Develop LSTM Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)

[How to Load and Explore Time Series Data in Python](https://machinelearningmastery.com/load-explore-time-series-data-python/)

[Mini-Course on Long Short-Term Memory Recurrent Neural Networks with Keras](https://machinelearningmastery.com/long-short-term-memory-recurrent-neural-networks-mini-course/)


## Time Series Analysis

When using classical statistics, the primary concern is the analysis of time series.

The goal of _time series analysis_ (TSA) is to develop models that best capture or describe an observed time series in order to understand the underlying causes. 

TSA seeks the **Why** behind the time series dataset which often involves making assumptions about the form of the data and decomposing the time series into constituent components.

The quality of a descriptive model is determined by how well it describes _all_ available data and the interpretation it provides to better inform the problem domain.


### Patterns in a time series

A time series can be split into the following components: 

    Base Level + Trend + Seasonality + Error

A _trend_ is observed when there is an increasing or decreasing slope in the time series. 

A _seasonality_ is observed when there is a distinct repeated pattern between regular intervals due to seasonal factors. 

The seasonality could be due to the month of the year, the day of the month, weekdays, or even time of day.

Another aspect to consider is _cyclic_ behaviour which happens when the rise and fall pattern in the series does not occur in fixed calendar-based intervals.


### Additive and multiplicative time series

Depending on the nature of the trend and seasonality, a time series can be modeled as an _additive_ or _multiplicative_ where each observation in the series can be expressed as either a sum or a product of the components:

Additive time series:        Value = Base Level + Trend + Seasonality + Error

Multiplicative Time Series:  Value = Base Level x Trend x Seasonality x Error

### How to decompose a time series into its components?

We can perform a classical decomposition of a time series by considering the series as an additive or multiplicative combination of the base level, trend, seasonal index, and the residual using the `seasonal_decompose` function in statsmodels.


### Stationary and Non-Stationary Time Series

A time series is _stationary_ when the values of the series are not a function of time.

The statistical properties of the series such as mean, variance, and autocorrelation are constant over time. 

The _autocorrelation_ of the series is simply the correlation of the series with its previous values.

A stationary time series is also devoid of seasonal effects.

Most statistical forecasting methods are designed to work on a stationary time series. 

The first step in the forecasting process is usually to do some transformation to convert a non-stationary series to stationary.

Here are some more topics to review:

- Why make a time series stationary before forecasting?
- How to make a time series stationary?
- How to test for stationarity?
- What is the difference between white noise and a stationary time series?
- How to detrend a time series?
- How to deseasonalize a time series?
- How to test for seasonality of a time series?
- How to treat missing values in a time series?
- What is autocorrelation and partial autocorrelation functions?
- How to compute partial autocorrelation function?
- Why and How to smoothen a time series?



### Time Series Decomposition

A time series is assumed to be an aggregate or combination of four components: level, trend, seasonality, and noise.

All series have a level and noise. 

The trend and seasonality components are optional.


Time series analysis provides a useful abstraction for selecting forecasting methods which is the _decomposition_ of a time series into systematic and unsystematic components.

- **Systematic:** Components of the time series that have consistency or recurrence which can be described and modeled.

- **Non-Systematic:** Components of the time series that cannot be directly modeled.


A given time series is believed to consist of three systematic components: level, trend, and seasonality plus one non-systematic component called _noise_.

- Level: The average value in the series.

- Trend: The increasing or decreasing value in the series.

- Seasonality: The repeating short-term cycle in the series.

- Noise: The random variation in the series.

All series have a level and noise, but trend and seasonality components are optional.

It is also helpful to think of the components as combining either additive or multiplicative.

#### Classical Decomposition

The **classical decomposition** (CD) method is a relatively simple procedure that is the starting point for most other methods of time series decomposition. 

There are two forms of classical decomposition: an additive decomposition and a multiplicative decomposition. 

The classical method of time series decomposition forms the basis of many time series decomposition methods, so it is important to understand how it works. 

The first step in a classical decomposition is to use a **moving average** method to estimate the trend-cycle.

#### Additive Model

An additive model suggests that the components are added:

```
    y(t) = Level + Trend + Seasonality + Noise
```

An additive model is linear where changes over time are consistently made by the same amount.

A linear trend is a straight line.

A linear seasonality has the same frequency (width of cycles) and amplitude (height of cycles).

#### Multiplicative Model

A multiplicative model suggests that the components are multiplied:

```
    y(t) = Level * Trend * Seasonality * Noise
```

A multiplicative model is nonlinear () quadratic or exponential) in which changes increase or decrease over time.

A nonlinear trend is a curved line.

A nonlinear seasonality has an increasing or decreasing frequency and/or amplitude over time.


### Decomposition as a Tool

Decomposition is primarily used for time series analysis which can be used to inform forecasting models on the given problem.

Decomposition provides a structured way of thinking about a time series forecasting problem, both generally in terms of modeling complexity and specifically in terms of how to best capture each of these components in a given model.

Each of these components are something we may need to think about and address during data preparation, model selection, and model tuning. 

We may address decomposition explicitly in terms of modeling the trend and subtracting it from your data or implicitly by providing enough history for an algorithm to model a trend if it exists.

We may or may not be able to cleanly or perfectly break down the time series as an additive or multiplicative model.

- Real-world problems are messy and noisy.

- There may be additive and multiplicative components. 

- There may be an increasing trend followed by a decreasing trend. 

- There may be non-repeating cycles mixed in with the repeating seasonality components.

However, these abstract models provide a simple framework that we can use to analyze the data and explore ways to think about and forecast the problem.



### Automatic Time Series Decomposition

There are methods to automatically decompose a time series.

The `statsmodels` library provides an implementation of the naive or classical decomposition method in a function called `seasonal_decompose()`. 

The statsmodels linrary requires that we specify whether the model is additive or multiplicative. 

We must be careful to be critical when interpreting the result. 

A review of a plot of the time series and some summary statistics can often be a good start to get an idea of whether the time series appears additive or multiplicative.

The `seasonal_decompose()` function returns a result object that contains arrays to access four pieces of data from the decomposition.

The result object provides access to the trend and seasonal series as arrays. It also provides access to the _residuals_ which are the time series after the trend and  seasonal components are removed. Finally, the original or observed data is also stored.


Although classical methods are common, they are not recommended:

- The technique is not robust to outlier values.

- It tends to over-smooth sudden rises and dips in the time series data.

- It assumes that the seasonal component repeats from year to year.

- The method produces no trend-cycle estimates for the first and last few observations.

- There are better methods that can be used for decomposition such as X11 decomposition, SEAT decomposition, or STL decomposition.

STL has many advantages over classical, X11, and SEAT decomposition techniques. 


Review the following sections:

- Additive Decomposition
- Multiplicative Decomposition
- Airline Passengers Dataset Example


-----------



## Time Series 101 Guide

### STL Decomposition

STL stands for seasonal and trend decomposition using Loess. 

STL is robust to outliers and can handle any kind of seasonality which also makes it a versatile method for decomposition.

There are a few things we can control when using STL:

- Trend cycle smoothness

- Rate of change in seasonal component

- The robustness towards the user outlier or exceptional values which controls the effects of outliers on the seasonal and trend components.

SLT has some disadvantages:

- STL cannot handle calendar variations automatically. 

- STL only provides a decomposition for additive models. 

  We can obtain the multiplicative decomposition by taking the logarithm of the data and then  transforming the components.
  
```py
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import STL
     
    elec_equip = read_csv(r"C:/Users/datas/python/data/elecequip.csv")
    stl = STL(elec_equip, period=12, robust=True)
    res_robust = stl.fit()
    fig = res_robust.plot()
```

### Basic Time Series Forecasting Methods

Although there are many statistical techniques for forecasting time series data, here we only consider the most straight-forward and simple methods that can be used for effective time series forecasting. 

These methods also serve as the foundation for some of the other methods:

- Simple Moving Average (SMA)
- Weighted Moving Average (WMA)
- Exponential Moving Average (EMA)


----------



## How to isolate trend, seasonality, and noise from a time series

The commonly occurring seasonal periods are a day, week, month, quarter (or season), and year.

### The Seasonal component

The seasonal component explains the periodic ups and downs one sees in many data sets such as the one shown below.

Seasonality can also observed on much longer time scales such as in the solar cycle which follows a roughly 11 year period.

### The Trend component

The Trend component refers to the pattern in the data that spans across seasonal periods.

The time series of retail eCommerce sales shown below demonstrates a possibly quadratic trend (y = x²) that spans across the 12 month long seasonal period:

### The Cyclical component

The cyclical component represents phenomena that happen across seasonal periods. 

Cyclical patterns do not have a fixed period like seasonal patterns. 

An example of a cyclical pattern is the cycles of boom and bust that stock markets experience in response to world events.

The cyclical component is hard to isolate, so it is often left alone by combining it with the trend component.

### The Noise component

The noise or random component is what remains after we separate the seasonality and trend from the time series. 

Noise is the effect of factors that you do not know or we cannot measure. 

Noise is the effect of the known unknowns or the unknown unknowns.

### Additive and Multiplicative effects

The trend, seasonal and noise components can combine in an additive or a multiplicative way.

### Time series decomposition using statsmodels

Now that we know how decomposition works, we can use the `seasonal_decompose()` in statsmodels to perform all of the work in one line of code:

```py
    from statsmodels.tsa.seasonal import seasonal_decompose
     
    components = seasonal_decompose(df['Retail_Sales'], model='multiplicative')
    # components = seasonal_decompose(np.array(elecequip), model='multiplicative', freq=4)
    components.plot()
```


## More Topics for Review

- Patterns in a time series
- How to decompose a time series into its components using seasonal_decompose

- Stationary and Non-Stationary Time Series
- How to test for seasonality of a time series
- What is autocorrelation and partial autocorrelation

- How to treat missing values in a time series
- How to test for seasonality of a time series

- How to detrend a time series
- How to deseasonalize a time series
- How to smoothen a time series



## References

[How to Decompose Time Series Data into Trend and Seasonality](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)

[Time Series Forecast and Decomposition 101 Guide Python](https://datasciencebeginners.com/2020/11/25/time-series-forecast-and-decomposition-101-guide-python/)

[How To Isolate Trend, Seasonality, and Noise From A Time Series](https://timeseriesreasoning.com/contents/time-series-decomposition/)




