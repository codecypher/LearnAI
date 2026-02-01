# Stationary Time Series

When modeling, there are assumptions that the summary statistics of observations are consistent. In time series terminology, we refer to this expectation as the time series being stationary.

These assumptions can be easily violated in time series by the addition of a trend, seasonality, and other time-dependent structures.

In this tutorial, you will discover how to check if your time series is stationary with Python.

## Stationary Time Series

The observations in a stationary time series are not dependent on time.

Time series are stationary if they do not have trend or seasonal effects. 

Summary statistics calculated on the time series are consistent over time such as mean or variance of the observations.

If a time series is _stationary_ it can be easier to model. 

Statistical modeling methods assume or require the time series to be stationary to be effective.

Should you make your time series stationary?

Generally, yes.

If you have clear trend and seasonality in your time series then model these components, remove them from observations, and train models on the residuals.

## Non-Stationary Time Series

Observations from a non-stationary time series show seasonal effects, trends, and other structures that depend on the time index.

Summary statistics like the mean and variance do change over time, providing a drift in the concepts a model may try to capture.

Classical time series analysis and forecasting methods are concerned with making non-stationary time series data stationary by identifying and removing trends and removing seasonal effects.

Statistical time series methods and even modern machine learning methods will benefit from the clearer signal in the data.

## Types of Stationary Time Series

The notion of stationarity comes from the theoretical study of time series and it is a useful abstraction when forecasting.

There are some finer-grained notions of stationarity that you may come across if you dive deeper into this topic:

- Stationary Process: A process that generates a stationary series of observations.

- Stationary Model: A model that describes a stationary series of observations.

- Trend Stationary: A time series that does not exhibit a trend.

- Seasonal Stationary: A time series that does not exhibit seasonality.

- Strictly Stationary: A mathematical definition of a stationary process, specifically that the joint distribution of observations is invariant to time shift.

## Checks for Stationarity

There are many methods to check whether a time series (direct observations, residuals, otherwise) is stationary or non-stationary.

1. **Look at Plots:** You can review a time series plot of your data and visually check if there are any obvious trends or seasonality.

2. **Summary Statistics:** You can review the summary statistics for your data for seasons or random partitions and check for obvious or significant differences.

3. **Statistical Tests:** You can use statistical tests to check if the expectations of stationarity are met or have been violated.

## Summary Statistics

Airline Passengers Dataset

We can take one step back and check if assuming a Gaussian distribution makes sense in this case by plotting the values of the time series as a histogram:

Running the example shows that indeed the distribution of values does not look Gaussian, so the mean and variance values are less meaningful.

This squashed distribution of the observations may be another indicator of a non-stationary time series. 

Reviewing the plot of the time series, we can see that there is an obvious seasonality component and it looks like the seasonality component is growing which may suggest an exponential growth from season to season. 

A _log transform_ can be used to flatten the exponential change to a linear relationship.

Below is the same histogram with a log transform of the time series.

Running the example, we can see the more familiar Gaussian or Uniform distribution of values.

We also create a line plot of the log transformed data and can see the exponential growth seems diminished, but we still have a trend and seasonal elements.

We can now calculate the mean and standard deviation of the values of the log transformed dataset:

Running the examples shows mean and standard deviation values for each group that are again similar, but not identical.

Perhaps, from these numbers alone, we would say the time series is stationary, but we strongly believe this to not be the case from reviewing the line plot.

We can use a statistical test to check if the difference between two samples of Gaussian random variables is real or a statistical fluke. 

We could also explore statistical significance tests, like the Student t-test, but things get tricky because of the serial correlation between values.

In the next section, we will use a statistical test designed to explicitly comment on whether a univariate time series is stationary.

## Augmented Dickey-Fuller test

Statistical tests make strong assumptions about your data. 

- The tests can only be used to inform the degree to which a null hypothesis can be rejected or fail to be reject. 

- The result must be interpreted for a given problem to be meaningful.

Nevertheless, they can provide a quick check and confirmatory evidence that your time series is stationary or non-stationary.

The **Augmented Dickey-Fuller (ADF) test** is a type of statistical test called a _unit root_ test.

The intuition behind a unit root test is that it determines how strongly a time series is defined by a _trend_.

There are a number of unit root tests and the Augmented Dickey-Fuller may be one of the more widely used. 

ADF uses an autoregressive model and optimizes an information criterion across multiple different lag values.

The null hypothesis of the test is that the time series can be represented by a unit root whixh means that it is non-stationary (has some time-dependent structure). 

The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.

- **Null Hypothesis (H0):** If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary which means it has some time dependent structure.

- **Alternate Hypothesis (H1):** The null hypothesis is rejected which suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

We interpret this result using the p-value from the test. 

A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).

- `p-value > 0.05`: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.

- `p-value <= 0.05`: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

Below is an example of calculating the Augmented Dickey-Fuller test on the Daily Female Births dataset. 

The statsmodels library provides the ` adfuller()` function that implements the test:

Running the example prints the test statistic value of -4. The more negative this statistic, the more likely we are to reject the null hypothesis (we have a stationary dataset).

As part of the output, we get a look-up table to help determine the ADF statistic. 

We can see that our statistic value of -4 is less than the value of -3.449 at 1%.

This suggests that we can reject the null hypothesis with a significance level of less than 1% (i.e. a low probability that the result is a statistical fluke).

Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.

We can log transform the dataset again to make the distribution of values more linear and better meet the expectations of this statistical test:



## References

[How to Check if Time Series Data is Stationary with Python?](https://machinelearningmastery.com/time-series-data-stationary-python/)

