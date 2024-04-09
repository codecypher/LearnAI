# Time Series Forecasting

TODO:

- Visualize a time series
- Plot the time series
- Pipelines
- Kedro
- Dataprep
- AutoML
- MLflow
- Review Chapter 11 Time Series of "Python for Data Analysis"

- Review notes and references on Data Pipelines
- Review notes and tips on time series data prep

- Using Keras model with scikit-learn
- Speedup TensorFlow Training
- Improve Tensorflow Performance
- Keras GPU Performance


## Background

[How to Develop LSTM Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)

[How to Load and Explore Time Series Data in Python](https://machinelearningmastery.com/load-explore-time-series-data-python/)

[Mini-Course on Long Short-Term Memory Recurrent Neural Networks with Keras](https://machinelearningmastery.com/long-short-term-memory-recurrent-neural-networks-mini-course/)



## Describing vs Predicting

We have different goals depending on whether we are interested in understanding a dataset or making predictions.

Time series analysis (TSA) is concerned with using methods such as decomposition of a time series into its systematic components in order to understand the underlying causes or the _Why_ behind the time series dataset which is **usually not helpful for prediction**.

Time series forecasting is making predictions about the future which is called _extrapolation_ in the classical statistical approach to time series data.

Forecasting involves taking models fit on historical data and using them to predict future observations.

TODO: Add table for time series analysis vs forecasting from Thesis presentation

We have different goals depending on whether we are interested in understanding a dataset or making predictions:


### Time Series Analysis

- When using classical statistics, the primary concern is the analysis of time series.

- The goal of _time series analysis_ (TSA) is to develop models that best capture or describe an observed time series in order to understand the underlying causes.

- TSA seeks the **Why** behind the time series dataset which often involves making assumptions about the form of the data and decomposing the time series into constituent components.

- The quality of a descriptive model is determined by how well it describes _all_ available data and the interpretation it provides to better inform the problem domain.


### Time Series Forecasting (TSF)

- Making predictions about the future is called _extrapolation_ in the classical statistical handling of time series data.

- Forecasting involves taking models fit on historical data and using them to predict future observations.

- Descriptive models can borrow for the future (to smooth or remove noise), they only seek to best describe the data.

- An important distinction in forecasting is that the future is completely unavailable and must only be estimated from what has already happened.

- Time series analysis can be used to remove trend and/or seasonality components which can help with forecasting.


----------


### 11 Time Series

11.1 Date and Time Data Types and Tools

11.2 Time Series Basics

11.3 Date Ranges, Frequencies, and Shifting

11.5 Periods and Period Arithmetic

11.6 Resampling and Frequency Conversion

11.7 Moving Window Functions


Table 11-1: Types in datetime module

Table 11-2: Datetime format specification (ISO C89 compatible)

Table 11-3: Locale-specific date formatting

Table 11-4: Base time series frequencies (not comprehensive)

Table 11-5. Resample method arguments



----------


## Time Series Forecasting

Time series forecasting (TSF) can broadly be categorized into the following categories:

- Classical / Statistical Models: Moving Averages, Exponential smoothing, ARIMA, SARIMA, TBATS

- Machine Learning: Linear Regression, XGBoost, Random Forest (any ML model with reduction methods)

- Deep Learning: RNN, LSTM


Making predictions about the future is called _extrapolation_ in the classical statistical handling of time series data.

More modern fields focus on the topic and refer to it as _time series forecasting_.

Forecasting involves taking models fit on historical data and using them to predict future observations.

Descriptive models can borrow for the future (to smooth or remove noise), they only seek to best describe the data.

An important distinction in forecasting is that the future is completely unavailable and must only be estimated from what has already happened.

TSA can be used to remove trend and/or seasonality components which can help with forecasting, but a literature review of similar TSF problems may be a better approach.


## Time Series Data Preparation

The goal of time series forecasting is to make accurate predictions about the future.

The fast and powerful methods that we rely on in machine learning (such as using train-test splits and k-fold cross validation) do not work in the case of time series data since they ignore the temporal components inherent in the problem [6][7].

NOTE: k-fold Cross Validation Does Not Work for Time Series Data.

The missing values in the time series dataset can be handled using two broad techniques:

- Drop the record with the missing value
- Impute the missing information

However, dropping the missing value is an inappropriate solution, as we may lose the correlation of adjacent observation.

Time Series models work with the complete data so they require us to impute the missing values prior to the modeling or actual time series analysis.

Estimating or imputing the missing values can be an excellent approach to dealing with the missing values.

In [5] the author discusses 4 such techniques that can be used to impute missing values in a time series dataset:

1. Last Observation Carried Forward (LOCF)
2. Next Observation Carried Backward (NOCB)
3. Rolling Statistics
4. Interpolation


## Time Series Decomposition

Time series analysis provides a body of techniques to better understand a dataset.

Perhaps the most useful of these is the _decomposition_ of a time series into four constituent parts:

  1. Level: The baseline value for the series if it were a straight line.

  2. Trend: The optional and often linear increasing or decreasing behavior of the series over time.

  3. Seasonality: The optional repeating patterns or cycles of behavior over time.

  4. Noise: The optional variability in the observations that cannot be explained by the model.

All time series have a level, most have noise, and the trend and seasonality are optional.



## Forecast Performance Baseline

A _baseline_ in forecast performance provides a point of comparison.

- A baseline is a point of reference for all other modeling techniques on your problem.

- If a model achieves performance at or below the baseline, the technique should be fixed or abandoned.

The technique used to generate a forecast to calculate the baseline performance must be easy to implement and naive of problem-specific details.

Before you can establish a performance baseline on your forecast problem, you must develop a test harness which is comprised of:

- The dataset to be used to train and evaluate models.

- The resampling technique to be used to estimate the performance of the technique (such as train/test split).

- The performance measure to be used to evaluate forecasts (such as mean squared error).

Next, we need to select a naive technique that we can use to make a forecast and calculate the baseline performance.

The goal is to get a baseline performance on the time series forecast problem as quickly as possible so that we can get to work better understanding the dataset and developing more advanced models.

Three properties of a good technique for making a baseline forecast are:

- Simple: A method that requires little or no training or intelligence.

- Fast: A method that is fast to implement and computationally trivial to make a prediction.

- Repeatable: A method that is deterministic which means that it produces an expected output given the same input.


## 5-Step Forecasting Task

The 5 basic steps in a forecasting task are summarized by Hyndman and Athana­sopou­los in their book Forecasting: principles and practice. These steps are:

1. Problem Definition. The careful consideration of who requires the forecast and how the forecast will be used.

This is described as the most difficult part of the process, most likely because it is entirely problem specific and subjective.

2. Gather Information. The collection of historical data to analyze and model.

This also includes getting access to domain experts and gathering information that can help to best interpret the historical information and ultimately the forecasts that will be made.

3. Preliminary Exploratory Analysis. The use of simple tools such as graphing and summary statistics to better understand the data.

Review plots and summarize and note obvious temporal structures auch as trends, seasonality, anomalies such missing data, corruption, and outliers, and any other structures that may impact forecasting.

4. Choose and Fit Models. Evaluate two, three, or a suite of models of varying types on the problem.

Models may be chosen for evaluation based on the assumptions they make and whether the dataset conforms.

Models are configured and fit to the historical data.

5. Use and Evaluate a Forecasting Model. The model is used to make forecasts and the performance of those forecasts is evaluated and the skill of the models is estimated.

This may involve back-testing with historical data or waiting for new observations to become available  for comparison.

This 5-step process provides a strong overview from starting off with an idea or problem statement and leading to a model that can be used to make predictions.

The focus of the process is on understanding the problem and fitting a good model.


## Error Metrics for Time Series Forecasting

Some common error metrics used for Time Series Forecasting model assessment:

- Mean Square Error
- Root Mean Square Error
- Mean Absolute Error
- Mean Absolute Percentage Error
- Mean Frequency Error




## Time Series using Keras

[Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting)

[Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)

[Stock Market Predictions using Deep Learning](https://pub.towardsai.net/stock-market-predictions-using-deep-learning-9e471d9cbdb)


## AutoML

[Forecasting Atmospheric CO2 with Python with Darts](https://towardsdatascience.com/forecasting-atmospheric-co2-concentration-with-python-c4a99e4cf142)

[Darts Quickstart](https://unit8co.github.io/darts/quickstart/00-quickstart.html)


[AutoGluon - Predicting Columns in a Table](https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-quickstart.html)

[GluonTS - Probabilistic Time Series Modeling](https://ts.gluon.ai)

[How to Use AutoKeras for Classification and Regression](https://machinelearningmastery.com/autokeras-for-classification-and-regression/)

[AutoKeras TimeSeriesForecaster](https://autokeras.com/tutorial/timeseries_forecaster/)



## Multivariate Time Series Forecasting with LSTMs

[Beijing PM2.5 Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv)

### Data Preparation

- Consolidate the date-time information into a single value, so we can use it as an index in Pandas

- Drop the No column
- Rename columns
- Remove the first row of data (A values for pm2.5)
- Few scattered NA values later in the dataset, but we mark them with zero values for now


## Multivariate Multi-Step Time Series Forecasting

- Handle Missing Data
- Supervised Representation
- Model Evaluation Test Harness
- Evaluate Linear Algorithms
- Evaluate Nonlinear Algorithms
- Tune Lag Size
- Extensions



## References

[1]: W. McKinney, Python for Data Analysis 2nd ed., Oreilly, ISBN: 978-1-491-95766-0, 2018.

[2]: [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)

[3]: [How to Develop Multivariate Multi-Step Time Series Forecasting Models for Air Pollution](https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/)

[4]: [Time Series Data Visualization In Python](https://pub.towardsai.net/time-series-data-visualization-in-python-2b1959726312)

[5]: [4 Techniques to Handle Missing values in Time Series Data](https://satyam-kumar.medium.com/4-techniques-to-handle-missing-values-in-time-series-data-c3568589b5a8)

[6]: [Don’t Use K-fold Validation for Time Series Forecasting](https://towardsdatascience.com/dont-use-k-fold-validation-for-time-series-forecasting-30b724aaea64)

[7]: [Bias-Variance Tradeoff in Time Series](https://towardsdatascience.com/bias-variance-tradeoff-in-time-series-8434f536387a)


[^tflow_optimize_pipeline]: <https://medium.com/@virtualmartire/optimizing-a-tensorflow-input-pipeline-best-practices-in-2022-4ade92ef8736> "Optimizing a TensorFlow Input Pipeline: Best Practices in 2022"

[^speedup_tflow_training]: <https://blog.seeso.io/a-simple-guide-to-speed-up-your-training-in-tensorflow-2-8386e6411be4?gi=55c564475d16> "A simple guide to speed up your training in TensorFlow"

[^accelerate_tflow_training]: <https://towardsdatascience.com/accelerate-your-training-and-inference-running-on-tensorflow-896aa963aa70> "Accelerate your training and inference running on Tensorflow"

[^speedup_sklearn_training]: <https://medium.com/distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1> "How to Speed up Scikit-Learn Model Training"


[^gpu_keras_wandb]: <https://wandb.ai/authors/ayusht/reports/Using-GPUs-With-Keras-A-Tutorial-With-Code--VmlldzoxNjEyNjE> "Using GPUs With Keras and wandb: A Tutorial With Code"

[^gpu_tflow]: <https://www.tensorflow.org/guide/gpu> "Use a GPU with Tensorflow"
