# Time Series Forecasting

## Terminology

The article [14] is a detailed list of concepts related to time series forecasting and their explanations, along with packages for Python.

A univariate time series data contains only one single time-dependent variable. 

A multivariate time series data consists of multiple time-dependent variables.


## Describing vs Predicting

We have different goals depending on whether we are interested in understanding a dataset or making predictions.

Time series analysis is concerned with using methods such as decomposition of a time series into its systematic components in order to understand the underlying causes or the _Why_ behind the time series dataset which is usually not helpful for prediction. However, time series analysis can be used to remove trend and/or seasonality components which can help with forecasting.

Time series forecasting is making predictions about the future which is called _extrapolation_ in the classical statistical approach to time series data.

Forecasting involves taking models fit on historical data and using them to predict future observations.

Time series analysis can be used to remove trend and/or seasonality components which can help with forecasting.


## Time Series Analysis

When using classical statistics, the primary concern is the analysis of time series.

The goal of _time series analysis_ (TSA) is to develop models that best capture or describe an observed time series in order to understand the underlying causes. 

TSA seeks the **Why** behind the time series dataset which often involves making assumptions about the form of the data and decomposing the time series into constituent components.

The quality of a descriptive model is determined by how well it describes _all_ available data and the interpretation it provides to better inform the problem domain.



## Time Series Forecasting

Time series forecasting can broadly be categorized into the following categories:

- Classical / Statistical Models: Moving Averages, Exponential smoothing, ARIMA, SARIMA, TBATS

- Machine Learning: Linear Regression, XGBoost, Random Forest (any ML model with reduction methods)

- Deep Learning: RNN, LSTM


Making predictions about the future is called _extrapolation_ in the classical statistical handling of time series data.

More modern fields focus on the topic and refer to it as _time series forecasting_.

Forecasting involves taking models fit on historical data and using them to predict future observations.

Descriptive models can borrow for the future (to smooth or remove noise), they only seek to best describe the data.

An important distinction in forecasting is that the future is completely unavailable and must only be estimated from what has already happened.

Time series analysis can be used to remove trend and/or seasonality components which can help with forecasting.


In time series forecasting, the evaluation of models on historical data is called _backtesting_. 


## Time Series Decomposition

Time series analysis provides a body of techniques to better understand a dataset.

Perhaps the most useful of these is the _decomposition_ of a time series into four constituent parts:

  1. Level: The baseline value for the series if it were a straight line.
  
  2. Trend: The optional and often linear increasing or decreasing behavior of the series over time.
  
  3. Seasonality: The optional repeating patterns or cycles of behavior over time.
  
  4. Noise: The optional variability in the observations that cannot be explained by the model.

All time series have a level, most have noise, and the trend and seasonality are optional.



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

A common algorithm used in establishing a baseline performance is the _persistence algorithm_.



## 5 Step Forecasting Task

The 5 basic steps in a forecasting task are summarized by Hyndman and Athana­sopou­los in their book Forecasting: principles and practice. These steps are:

1. **Problem Definition:** The careful consideration of who requires the forecast and how the forecast will be used. 

This is described as the most difficult part of the process, most likely because it is entirely problem specific and subjective.

2. **Gather Information:** The collection of historical data to analyze and model. 

This also includes getting access to domain experts and gathering information that can help to best interpret the historical information and ultimately the forecasts that will be made.

3. **Preliminary Exploratory Analysis:** The use of simple tools such as graphing and summary statistics to better understand the data. 

Review plots and summarize and note obvious temporal structures auch as trends, seasonality, anomalies such missing data, corruption, and outliers, and any other structures that may impact forecasting.

4. **Choose and Fit Models:** Evaluate two, three, or a suite of models of varying types on the problem. 

Models may be chosen for evaluation based on the assumptions they make and whether the dataset conforms. 

Models are configured and fit to the historical data.

5. **Use and Evaluate a Forecasting Model:** The model is used to make forecasts and the performance of those forecasts is evaluated and the skill of the models is estimated. 

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


----------



# How to Develop LSTM Models for Time Series Forecasting

There are many time series forecasting tutorials and examples using and air pollution and weather toy datasets. 

Unless you make use of the time series features of tflow Dataset or torch Dataloader you will encounter two issues: 

1. We have to write a custom function to reframe the problem from time series to a supervised learning problem (X, y) which is what most all ML models including NN such as LSTM will require. 

2. The custom function will likely not perform as well as a Dataset/Dataloader and will probably not scale or support parallelization. 


Here, we explore how to develop a suite of different types of LSTM models for time series forecasting.

> The following examples manually convert the time series to supervised problem (split_seauences and series_to_supervised). A better approach is to make use of the time series featues of tflow Dataset or torch Daraloader, especially for multivariate time series.


## The 5 Step Life-Cycle for LSTM Models in Keras

The article [8] discusses the 5 steps in the LSTM model life-cycle in Keras:

1. Define Network
2. Compile Network
3. Fit Network
4. Evaluate Network
5. Make Predictions


## Univariate LSTM Models

LSTMs can be used to model univariate time series forecasting problems which are problems comprised of a single series of observations and a model is required to learn from the series of past observations to predict the next value in the sequence.

We will demonstrate a number of variations of the LSTM model for univariate time series forecasting.

1. Data Preparation
2. Vanilla LSTM
3. Stacked LSTM
4. Bidirectional LSTM
5. CNN LSTM
6. ConvLSTM

Each of these models are demonstrated for one-step univariate time series forecasting, but can easily be adapted and used as the input part of a model for other types of time series forecasting problems.

### Data Preparation

Before a univariate series can be modeled, it must be prepared.

The LSTM model will learn a function that maps a sequence of past observations as input to an output observation. Therefore, the sequence of observations must be transformed into multiple examples from which the LSTM can learn.

### Vanilla LSTM

A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units and an output layer used to make a prediction.

We can define a Vanilla LSTM for univariate time series forecasting as follows:

We are working with a univariate series, so the number of features is one (one variable).

```
[samples, timesteps, features]

[10 20 30] 40
[20 30 40] 50
[30 40 50] 60
[40 50 60] 70
[50 60 70] 80
[60 70 80] 90
```

### Stacked LSTM

Multiple hidden LSTM layers can be stacked one on top of another in what is referred to as a Stacked LSTM model.

An LSTM layer requires a three-dimensional input and LSTMs by default will produce a two-dimensional output as an interpretation from the end of the sequence.

We can address this by having the LSTM output a value for each time step in the input data by setting the `return_sequences=True` argument on the layer which allows us to have 3D output from hidden LSTM layer as input to the next.

We can define a Stacked LSTM as follows:

### Bidirectional LSTM

On some sequence prediction problems, it can be beneficial to allow the LSTM model to learn the input sequence both forward and backwards and concatenate both interpretations which is called a Bidirectional LSTM.

We can implement a Bidirectional LSTM for univariate time series forecasting by wrapping the first hidden layer in a wrapper layer called Bidirectional.

An example of defining a Bidirectional LSTM to read input both forward and backward is as follows:

### CNN LSTM

A convolutional neural network (CNN) is a type of neural network for working with two-dimensional image data.

The CNN can be very effective at automatically extracting and learning features from one-dimensional sequence data such as univariate time series data.

A CNN model can be used in a hybrid model with an LSTM backend where the CNN is used to interpret subsequences of input that together are provided as a sequence to an LSTM model to interpret. This hybrid model is called a CNN-LSTM.

### ConvLSTM

A type of LSTM related to the CNN-LSTM is the ConvLSTM where the convolutional reading of input is built directly into each LSTM unit.

The ConvLSTM was developed for reading two-dimensional spatial-temporal data, but can be adapted for use with univariate time series forecasting.



## Multivariate LSTM Models

Multivariate time series data means data which has more than one observation for each time step.

There are two main models that we may require with multivariate time series data; they are:

  1. Multiple Input Series.
  2. Multiple Parallel Series.

### Multiple Input Series

A problem may have two or more parallel input time series and an output time series that is dependent on the input time series.

The input time series are _parallel_ because each series has an observation at the same time steps.

We can demonstrate this with a simple example of two parallel input time series where the output series is the simple addition of the input series:

We can reshape the three arrays of data into a single dataset where each row is a time step and each column is a separate time series which is a standard way of storing parallel time series in a CSV file.

```py
    # Multivariate data preparation
    from numpy import array
    from numpy import hstack
    # define input sequence
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))

    # horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))
    print(dataset)
```

Running the example prints the dataset with one row per time step and one column for each of the two input and one output parallel time series.

```
    [[ 10  15  25]
     [ 20  25  45]
     [ 30  35  65]
     [ 40  45  85]
     [ 50  55 105]
     [ 60  65 125]
     [ 70  75 145]
     [ 80  85 165]
     [ 90  95 185]]
```

As with the univariate time series, we must structure the data into samples with input and output elements.

An LSTM model needs sufficient context to learn a mapping from an input sequence to an output value. LSTMs can support parallel input time series as separate variables or features. Therefore, we need to split the data into samples maintaining the order of observations across the two input sequences.

If we chose three input time steps, the first sample would look as follows:

```
    # Input
    10, 15
    20, 25
    30, 35

    # Ouput
    65
```

In transforming the time series into input/output samples to train the model, we can see that we will have to discard some values from the output time series where we do not have values in the input time series at prior time steps. In turn, the choice of the size of the number of input time steps will have an important effect on how much of the training data is used.

We can define a function called `split_sequences()` that will take a dataset as we have defined it with rows for time steps and columns for parallel series and return input/output samples.

```py
    def split_sequences(sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
```

Here is the complete example:

```py
    # Multivariate data preparation
    from numpy import array, hstack

    # split a multivariate sequence into samples
    def split_sequences(sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    # define input sequence
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))

    # horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, out_seq))

    # choose a number of time steps
    n_steps = 3

    # convert into input/output
    X, y = split_sequences(dataset, n_steps)
    print(X.shape, y.shape)

    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])
```

We can see that the X component has a three-dimensional structure: [samples, timesteps, features]

- The first dimension is the number of samples (in this case 7). 

- The second dimension is the number of time steps per sample (in this case 3) which is the value specified to the split_sequences function. 

- The last dimension specifies the number of parallel time series or the number of features (in this case 2 for two parallel series).

This is the exact three-dimensional structure expected by an LSTM as input, so the data is ready to use without further reshaping.

We can then see that the input and output for each sample is printed, showing the three time steps for each of the two input series and the associated output for each sample:

```
(7, 3, 2) (7,)

[[10 15]
 [20 25]
 [30 35]] 65
[[20 25]
 [30 35]
 [40 45]] 85
[[30 35]
 [40 45]
 [50 55]] 105
[[40 45]
 [50 55]
 [60 65]] 125
[[50 55]
 [60 65]
 [70 75]] 145
[[60 65]
 [70 75]
 [80 85]] 165
[[70 75]
 [80 85]
 [90 95]] 185
```

We are now ready to fit an LSTM model on this data.

Any of the varieties of LSTMs in the previous section can be used, such as a Vanilla, Stacked, Bidirectional, CNN, or ConvLSTM model.

When making a prediction, the model expects three time steps for two input time series.

The complete example is listed below:


### Multiple Parallel Series

An alternate time series problem is the case where there are multiple parallel time series and a value must be predicted for each.

We may want to predict the value for each of the three time series for the next time step which might be referred to as multivariate forecasting.

Again, the data must be split into input/output samples in order to train a model.

The first sample of this dataset would be:


----------


# Time Series Datasets using Keras

```py
def normalize(data: np.ndarray, train_split: float):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std, data_mean, data_std


def get_ds_shape(ds: tf.data.Dataset):
    # The dataset yields tuples: (batch_of_sequences, batch_of_targets)
    count = 0
    ds_shape = []
    for batch in ds.take(-1):
        inputs, targets = batch
        count += inputs.numpy().shape[0]
        ds_shape = [count, inputs.numpy().shape[1], inputs.numpy().shape[2]]

        # input is (batch_size, time_steps, num_features)
        # print("\ntrain:", inputs.numpy().shape, targets.numpy().shape)

    return ds_shape

    
def create_datasets(params: Params, debug: bool = False):
    """
    Create train/val/test datasets.
    TODO: Update code snippet
    """
    batch_size = params.batch_size

    # use the last past values to forecast the value for future time steps ahead
    step = 1  # sample rate (1 per day)
    past = 10  # tracking data from past 10 timestamps (10/1=1 day)
    future = 1  # predict the target after 1 timestamps (1/1=10 days).

    sequence_length = int(past / step)

    train_dfs, date_dfs = load_data(params)
    train_df, date_df = data_prep(params, train_dfs, date_dfs)

    # target feature is first column
    # filtered_df, filtered_df_ext = select_features(params, train_df)

    date_df.index.rename("date", inplace=True)
    # date_df.drop('day', axis='columns', inplace=True)
    # date_df['day'] = np_date_data
    # date_df.index = date_df['day']

    data, target, mean_array, std_array = preprocess(params, train_df, debug=debug)

    # Create datasets
    train_ds = keras.preprocessing.timeseries_dataset_from_array(
        data,
        target,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
        end_index=params.train_split - 1,
    )

    val_ds = keras.preprocessing.timeseries_dataset_from_array(
        data,
        target,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
        start_index=params.train_split,
        end_index=params.val_split - 1,
    )

    test_ds = keras.preprocessing.timeseries_dataset_from_array(
        data,
        target,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
        start_index=params.val_split + 1,
    )

    train_shape = get_ds_shape(train_ds)
    val_shape = get_ds_shape(val_ds)
    test_shape = get_ds_shape(test_ds)

    if debug:
        print(f"\ntrain: {train_shape}")
        print(f"val: {val_shape}")
        print(f"test: {test_shape}")

    return train_ds, val_ds, test_ds, mean_array, std_array
```

----------



# Categories of Articles

## Time Series Background

[Taxonomy of Time Series Forecasting Problems](https://machinelearningmastery.com/taxonomy-of-time-series-forecasting-problems/)

[Introduction to Time Series Forecasting (Python)](https://machinelearningmastery.com/start-here/#timeseries)

[Time Series Forecasting as Supervised Learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)

[How to Difference a Time Series Dataset with Python](https://machinelearningmastery.com/difference-time-series-dataset-python/)

[How to Reframe Your Time Series Forecasting Problem](https://machinelearningmastery.com/reframe-time-series-forecasting-problem/)

[How to Convert a Time Series to a Supervised Learning Problem in Python](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

[Mini-Course on Long Short-Term Memory Recurrent Neural Networks with Keras](https://machinelearningmastery.com/long-short-term-memory-recurrent-neural-networks-mini-course/)

[How to Load, Visualize, and Explore a Multivariate Multistep Time Series Dataset](https://machinelearningmastery.com/how-to-load-visualize-and-explore-a-complex-multivariate-multistep-time-series-forecasting-dataset/)


## Time Series for Beginners

[Why You Shouldn’t Trade Crypto with Machine Learning](https://medium.com/geekculture/why-you-shouldnt-trade-crypto-with-machine-learning-a25a4af0beb8)

[Time Series From Scratch](https://towardsdatascience.com/time-series-analysis-from-scratch-seeing-the-big-picture-2d0f9d837329)

[Predicting stock prices using Deep Learning LSTM model in Python](https://thinkingneuron.com/predicting-stock-prices-using-deep-learning-lstm-model-in-python/)


## Time Series Analysis

[Time Series Analysis with Statsmodels](https://towardsdatascience.com/time-series-analysis-with-statsmodels-12309890539a)

[An Ultimate Guide to Time Series Analysis in Pandas](https://regenerativetoday.com/a-complete-guide-to-time-series-analysis-in-pandas/)

[How to Check if Time Series Data is Stationary](https://machinelearningmastery.com/time-series-data-stationary-python/)

[Avoid These Mistakes with Time Series Forecasting](https://www.kdnuggets.com/2021/12/avoid-mistakes-time-series-forecasting.html)


## Time Series Decomposition

[How to Decompose Time Series Data into Trend and Seasonality](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)

[How To Isolate Trend, Seasonality, and Noise From A Time Series](https://timeseriesreasoning.com/contents/time-series-decomposition/)

[How to Identify and Remove Seasonality from Time Series Data with Python](https://machinelearningmastery.com/time-series-seasonality-with-python/)


[Time Series Forecast and Decomposition 101 Guide Python](https://datasciencebeginners.com/2020/11/25/time-series-forecast-and-decomposition-101-guide-python/)

[Time Series Data Visualization with Python](https://machinelearningmastery.com/time-series-data-visualization-with-python/)

[Stacking Machine Learning Models for Multivariate Time Series](https://towardsdatascience.com/stacking-machine-learning-models-for-multivariateo-time-series-28a082f881)


## Time Series Data Preparation

[How to Load and Explore Time Series Data in Python](https://machinelearningmastery.com/load-explore-time-series-data-python/)

[Basic Feature Engineering With Time Series Data in Python](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)


## Time Series Feature Engineering

[Building a Tractable, Feature Engineering Pipeline for Multivariate Time Series](https://www.kdnuggets.com/2022/03/building-tractable-feature-engineering-pipeline-multivariate-time-series.html)

[Feature selection for forecasting algorithms](https://towardsdatascience.com/feature-selection-for-forecasting-algorithms-10598e50667f)

[How To Resample and Interpolate Your Time Series Data With Python](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)


## Forecast Performance Baseline

[How to Make Baseline Predictions for Time Series Forecasting with Python](https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/)

[How to Create an ARIMA Model for Time Series Forecasting in Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)

[How To Backtest Machine Learning Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)


## Time Series Classification

[Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)

[LSTMs for Human Activity Recognition Time Series Classification](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/)



## Time Series Forecasting using Python

[The Complete Guide to Time Series Forecasting Using Sklearn, Pandas, and Numpy](https://towardsdatascience.com/the-complete-guide-to-time-series-forecasting-using-sklearn-pandas-and-numpy-7694c90e45c1)


## Time Series Forecasting using Keras

[Quick Keras Recipes](https://keras.io/examples/keras_recipes/)

[Timeseries data preprocessing](https://keras.io/api/preprocessing/timeseries/)

[Working with RNNs](https://keras.io/guides/working_with_rnns/)


[Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)

[Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting)

[Timeseries anomaly detection using an Autoencoder](https://keras.io/examples/timeseries/timeseries_anomaly_detection/)


[Predicting stock prices using Deep Learning LSTM model in Python](https://thinkingneuron.com/predicting-stock-prices-using-deep-learning-lstm-model-in-python/)

[Time Series Forecast Using Deep Learning](https://medium.com/geekculture/time-series-forecast-using-deep-learning-adef5753ec85)

[Time series forecasting using Tensorflow](https://www.tensorflow.org/tutorials/structured_data/time_series)

[How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/)


## Time Series Examples using PyTorch

[LSTM for time series prediction](https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca)

[LSTM for Time Series in PyTorch](http://www.jessicayung.com/lstms-for-time-series-in-pytorch/)

[PyTorch LSTMs for time series forecasting of Indian Stocks](https://medium.com/analytics-vidhya/pytorch-lstms-for-time-series-forecasting-of-indian-stocks-8a49157da8b9)

[Training Time Series Forecasting Models in PyTorch](https://towardsdatascience.com/training-time-series-forecasting-models-in-pytorch-81ef9a66bd3a)


## Time Series Examples using AutoML

[How to Use AutoKeras for Classification and Regression](https://machinelearningmastery.com/autokeras-for-classification-and-regression/)

[Multiple Series? Forecast Them together with any Sklearn Model](https://towardsdatascience.com/multiple-series-forecast-them-together-with-any-sklearn-model-96319d46269)

[Forecasting Atmospheric CO2 with Python with Darts](https://towardsdatascience.com/forecasting-atmospheric-co2-concentration-with-python-c4a99e4cf142)

[Forecasting with Machine Learning Models using mlforecast](https://towardsdatascience.com/forecasting-with-machine-learning-models-95a6b6579090)


## Time Series Examples using PyCaret

[Introduction to Binary Classification with PyCaret](https://towardsdatascience.com/introduction-to-binary-classification-with-pycaret-a37b3e89ad8d?source=rss----7f60cf5620c9---4)

[PyCaret + SKORCH: Build PyTorch Neural Networks using Minimal Code](https://towardsdatascience.com/pycaret-skorch-build-pytorch-neural-networks-using-minimal-code-57079e197f33)

[Multiple Time Series Forecasting with PyCaret](https://towardsdatascience.com/multiple-time-series-forecasting-with-pycaret-bc0a779a22fe)

[Time Series Forecasting with PyCaret Regression Module](https://towardsdatascience.com/time-series-forecasting-with-pycaret-regression-module-237b703a0c63)



----------



# Time Series Forecasting Books

W. W. S. Wei, Multivariate Time Series Analysis and Applications, 1st ed., Wiley, 2019. 

D. C. Montgomery and C. L. Jennings, Introduction to Time Series Analysis and Forecasting, 2nd ed., Wiley, 2015. 

G. Shmueli and  K. C. Lichtendahl Jr., Practical Time Series Forecasting with R, 2nd ed., Axelrod Schnall Publishers, 2016. 

R. J. Hyndman, Forecasting: Principles and Practice, 3rd ed., Otexts, 2021, Available online: https://otexts.com/fpp3/



----------



# Confidence Intervals

Confidence intervals are a way of quantifying the uncertainty of an estimate. 

Confidence intervals can be used to add a bounds or likelihood on a population parameter (such as a mean) estimated from a sample of independent observations from the population. 

Confidence intervals come from the field of estimation statistics. 

Here are some key facts about CI [9]:

- A confidence interval is a bounds on an estimate of a population parameter.

- The confidence interval for the estimated skill of a classification method can be calculated directly.

- The confidence interval for any arbitrary population statistic can be estimated in a distribution-free way using the bootstrap.


A _confidence interval_ (CI) is a bounds on the estimate of a population variable. 

CI is an interval statistic used to quantify the _uncertainty_ on an estimate.


CI is different from a _tolerance interval_ that describes the bounds of data sampled from the distribution. 

CI is different from a _prediction interval_ that describes the bounds on a single observation. 


A confidence interval provides bounds on a population parameter such as a mean, standard deviation, etc.

In applied machine learning, we may want to use confidence intervals in the presentation of the skill of a predictive model.

A confidence interval could be used in presenting the skill of a classification model:

> Given the sample, there is a 95% likelihood that the range x to y covers the true model accuracy.

or

> The accuracy of the model was x +/- y at the 95% confidence level.

Confidence intervals can also be used in the presentation of the error of a regression predictive model:

> There is a 95% likelihood that the range x to y covers the true error of the model.

or

> The error of the model was x +/- y at the 95% confidence level.


The value of a confidence interval is its ability to quantify the uncertainty of the estimate. 

CI provides both a lower and upper bound and a likelihood. 

As a radius measure, the confidence interval is often referred to as the _margin of error_ and may be used to graphically depict the uncertainty of an estimate on graphs through the use of _error bars_.

The larger the sample from which the estimate was drawn, the more precise the estimate and the smaller (better) the confidence interval.

- Smaller Confidence Interval: A more precise estimate.

- Larger Confidence Interval: A less precise estimate.


Confidence intervals belong to a field of statistics called estimation statistics that can be used to present and interpret experimental results instead of (or in addition to) statistical significance tests.

Confidence intervals may be preferred in practice over the use of statistical significance tests.

The reason is that they are easier for practitioners and stakeholders to relate directly to the domain. 

CIs can also be interpreted and used to compare machine learning models.

Running the example, we see the calculated radius of the confidence interval calculated and printed:

0.111

- The classification error of the model is 20% +/- 11%

- The true classification error of the model is likely between 9% and 31%.


## Nonparametric Confidence Interval

We may not know the distribution for a chosen performance measure, or we may not know the analytical way to calculate a confidence interval for a skill score.

The assumptions that underlie parametric confidence intervals are often violated. 

The predicted variable sometimes is not normally distributed, or the variance of the normal distribution might not be equal at all levels of the predictor variable.

In these cases, the _bootstrap_ resampling method can be used as a nonparametric method for calculating confidence intervals called **bootstrap confidence intervals**.

The bootstrap is a simulated Monte Carlo method where samples are drawn from a fixed finite dataset with replacement and a parameter is estimated on each sample. 

The bootstrap procedure leads to a robust estimate of the true population parameter via sampling.


Running the example summarizes the distribution of bootstrap sample statistics including the 2.5th, 50th (median) and 97.5th percentile.

50th percentile (median) = 0.750
2.5th percentile = 0.741
97.5th percentile = 0.757

We can then use these observations to make a claim about the sample distribution:

> There is a 95% likelihood that the range 0.741 to 0.757 covers the true statistic mean.


# References

[1] W. McKinney, Python for Data Analysis, 2nd ed., Oreilly, ISBN: 978-1-491-95766-0, 2018.

[2]: [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)

[3]: [How to Develop Multivariate Multi-Step Time Series Forecasting Models for Air Pollution](https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/)

[4]: [Time Series Data Visualization In Python](https://pub.towardsai.net/time-series-data-visualization-in-python-2b1959726312)

[5]: [4 Techniques to Handle Missing values in Time Series Data](https://satyam-kumar.medium.com/4-techniques-to-handle-missing-values-in-time-series-data-c3568589b5a8)

[6]: [Don’t Use K-fold Validation for Time Series Forecasting](https://towardsdatascience.com/dont-use-k-fold-validation-for-time-series-forecasting-30b724aaea64)

[7]: [Bias-Variance Tradeoff in Time Series](https://towardsdatascience.com/bias-variance-tradeoff-in-time-series-8434f536387a)

[8]: [The 5 Step Life-Cycle for Long Short-Term Memory Models in Keras](https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)


[9]: [Confidence Intervals for Machine Learning](https://machinelearningmastery.com/confidence-intervals-for-machine-learning/)

[10]: [How to Report Classifier Performance with Confidence Intervals](https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/)

[11]: [How to Calculate Bootstrap Confidence Intervals For Machine Learning Results in Python](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)

[12]: [A Gentle Introduction to the Bootstrap Method](https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/)

[13]: [How To Backtest Machine Learning Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)

[14]: [A Curated List of Important Time Series Forecasting Concepts](https://pub.towardsai.net/a-curated-list-of-important-time-series-forecasting-concepts-eb2306b55c19)


[A Gentle Introduction to Statistical Tolerance Intervals in Machine Learning](https://machinelearningmastery.com/statistical-tolerance-intervals-in-machine-learning/)

[Prediction Intervals for Machine Learning](https://machinelearningmastery.com/prediction-intervals-for-machine-learning/)

[Understand Time Series Forecast Uncertainty Using Prediction Intervals with Python](https://machinelearningmastery.com/time-series-forecast-uncertainty-using-confidence-intervals-python/)

[Prediction Intervals for Deep Learning Neural Networks](https://machinelearningmastery.com/prediction-intervals-for-deep-learning-neural-networks/)


[Local vs Global Forecasting: What You Need to Know](https://towardsdatascience.com/local-vs-global-forecasting-what-you-need-to-know-1cc29e66cae0)

----------

[^what_is_tsf]: <https://machinelearningmastery.com/time-series-forecasting/> "What Is Time Series Forecasting?"

[^tsf_taxonomoy]: <https://machinelearningmastery.com/taxonomy-of-time-series-forecasting-problems/> "Taxonomy of Time Series Forecasting Problems"

[^tsf_error_metrics]: <https://medium.com/analytics-vidhya/error-metrics-used-in-time-series-forecasting-modeling-9f068bdd31ca> "Error Metrics used in Time Series Forecasting"

[^tsf_cnn]: <https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/> "How to Develop CNN Models for Time Series Forecasting"
