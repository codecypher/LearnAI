# Time Series Tips


## How to check for gaps in time series data?

- Understand the time range and tick granularity of the times series with visual examination of the example time series.

- Compare the actual number of ticks in each time series against the number of theoretical ticks implied by (max minus min timestamp) divided by tick spacing. 

  This ratio is sometimes referred as the _fill ratio_ where a value much less than 1 means there is a lot of ticks missing.

- Filter out series that have low fill ratio. 

  We can use 40% as the cutoff of insufficient information content but this may vary depending on the specific task at hand.

- Standardize tick spacing across time series by upsampling to a more granular resolution.

- Fill in the upsampled ticks with the appropriate interpolation method for your analysis such as take the last known value, linear/quadratic interpolation, etc.


----------


## Time Series Data Preparation

TODO: Add some notes



## Using to_categorical

[Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/)

TODO: Add some notes


## Tips and Tricks for Time Series

- Do not shuffle train/test datasets

- Convert dataset to 3D supervised shape [samples, ntimesteps, n_feature]

- Reframe the problem as supervised learning problem with (X, y) datasets


---------



## Tips for Working with Time Series

### Removing Noise with Fourier Transform

We often need to study the underlying process that drives a particular time series. Thus, we may want to remove the _noise_ from the time series and analyze the _signal_.

The fourier transform can help remove the noise from the time series. 

By moving our time series from the time domain to the frequency domain, we can filter out the frequencies that pollute the data. Then, we just apply the inverse fourier transform to get a filtered version of the time series.


### Removing Noise with Kalman Filter

With the fourier transform, we obtain the frequencies that exist in a given time series but we do not have any information about when these frequencies occur in time which meanst the fourier transform is not the best choice for _non-stationary_ time series.

For example, financial time series are considered non-stationary (although any attempt to prove it statistically is doomed) which makes fourier transform a bad choice.

At this point, we can choose to apply the fourier transform on a rolling-basis or use a wavelet transform but there is a much more interesting algorithm called the **Kalman Filter**.

The Kalman Filter is essentially a Bayesian Linear Regression that can optimally estimate the hidden state of a process using its observable variables.

By carefully selecting the right parameters, we can tweak the algorithm to extract the underlying signal.


### Dealing with Outliers

**Outliers** are usually undesirable because they affect our conclusions if we are not careful when dealing with them. 

For example, the Pearson correlation formula can have a very different result if there are large  outliers in our data.

Outlier analysis and filtering in time series requires a more sophisticated approach than in normal data because **we cannot use future information to filter past outliers**.

One quick way to remove outliers is on a rolling/expanding basis.

A common algorithm to find outliers is to compute the mean and standard deviation of the data and check which values are _n_ standard deviations above or below the mean (say n = 3) -- those values are marked as outliers.

NOTE: This particular approach will usually work best if we **standardize the data** (and it is conceptually more correct to use it that way).


### The right way to normalize time series data

Many posts use the classical fit-transform approach with time series as if they could be treated as normal data. 

As with outliers, we cannot use future information to normalize data from the past unless we are 100% sure the values you are using to normalize are constant over time.

The right way to normalize time series is on a **rolling/expanding** basis.

We can use the scikit-learn API to create a class to normalize data to avoid look-ahead bias. Since it inherits `BaseEstimator` and `TransformerMixin`, it is possible to embed this class in a scikit-learn pipeline.


### A flexible way to compute returns

The last tip is focused on quantitative analysis of financial time series.

When working with returns, it is usually necessary to have a basic framework to quickly compute log and arithmetic returns in different periods of time.

When filtering financial time series, the ideal procedure filters returns first and then goes back to prices, so we are free to add this step to the code from section 4.


## More Time Series Tips and Tricks

[Feature selection for forecasting algorithms](https://jackbakerds.com/posts/forecasting-feature-selection/)

[Stop One-Hot Encoding your Time-based Features](https://towardsdatascience.com/stop-one-hot-encoding-your-time-based-features-24c699face2f)


[Don’t Use K-fold Validation for Time Series Forecasting](https://towardsdatascience.com/dont-use-k-fold-validation-for-time-series-forecasting-30b724aaea64)

[How To Resample and Interpolate Your Time Series Data With Python](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)


## Generating Synthetic Data

Generative Adversarial Networks (GANs) can generate several types of synthetic data including image data, tabular data, and sound/speech data. However, synthetic data is discouraged in Russell and Norvig and other textbooks.

[GAN Tips](../tips/gan_tips.md)


----------


## Common Data Transforms for Time Series Forecasting

### Transforms for Time Series Data

Given a univariate time series dataset, there are four transforms that are popular when using machine learning methods to model and make predictions: power, difference, standarize, and normalize.

#### Power Transform

A _power transform_ removes a shift from a data distribution to make the distribution more normal (Gaussian).

On a time series dataset, this can have the effect of removing a change in variance over time.

Popular examples are the the log transform (positive values) or generalized versions such as the Box-Cox transform (positive values) or the Yeo-Johnson transform (positive and negative values).

#### Difference Transform

A _difference transform_ is a simple way for removing a systematic structure from the time series.

For example, a trend can be removed by subtracting the previous value from each value in the series which is called _first order differencing_. The process can be repeated (difference the differenced series) to remove second order trends, and so on.

A seasonal structure can be removed in a similar way by subtracting the observation from the prior season, say 12 time steps ago for monthly data with a yearly seasonal structure.

A single differenced value in a series can be calculated with a custom function named ``difference()`` shown below which takes the time series and the interval for the difference calculation, say 1 for a trend difference or 12 for a seasonal difference.


#### Standardization

Standardization is a transform for data with a Gaussian distribution that subtracts the mean and divides the result by the standard deviation of the data sample which has the effect of transforming the data to have mean of zero (or centered) and a standard deviation of 1.

This resulting distribution is called a _standard Gaussian distribution_ or _standard normal_ transform.

We can perform standardization using the `StandardScaler` class from the scikit-learn library.

This class allows the transform to be fit on a training dataset by calling `fit()`, applied to one or more datasets (train and test) by calling `transform()` and also provides a function to reverse the transform by calling `inverse_transform()`.

```py
    from sklearn.preprocessing import StandardScaler
    from numpy import array

    # define dataset
    data = [x for x in range(1, 10)]
    data = array(data).reshape(len(data), 1)
    print(data)

    # fit transform
    transformer = StandardScaler()
    transformer.fit(data)

    # difference transform
    transformed = transformer.transform(data)
    print(transformed)

    # invert difference
    inverted = transformer.inverse_transform(transformed)
    print(inverted)
```


#### Normalization

Normalization is a rescaling of data from the original range to a new range between 0 and 1.

As with standardization, this can be implemented using a transform object from the scikit-learn library called the `MinMaxScaler` class. 

In addition to normalization, `MinMaxScaler` can be used to rescale data to any range we wish by specifying the preferred range in the constructor of the object.

The `MinMaxScaler` class can be used in the same way to fit, transform, and inverse the transform.

```py
    from sklearn.preprocessing import MinMaxScaler
    from numpy import array

    # define dataset
    data = [x for x in range(1, 10)]
    data = array(data).reshape(len(data), 1)
    print(data)

    # fit transform
    transformer = MinMaxScaler()
    transformer.fit(data)

    # difference transform
    transformed = transformer.transform(data)
    print(transformed)

    # invert difference
    inverted = transformer.inverse_transform(transformed)
    print(inverted)
```


### Considerations for Model Evaluation

We have mentioned the importance of being able to invert a transform on the predictions of a model in order to calculate a model performance statistic that is directly comparable to other methods.

Another concern is the problem of **data leakage**.

Three of the above data transforms estimate coefficients from a provided dataset that are then used to transform the data:

  - Power Transform: lambda parameter

  - Standardization: mean and standard deviation statistics

  - Normalization: min and max values

**These coefficients must be estimated on the training dataset only.**


```py
    scaler = MinMaxScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train),
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)
```


Once estimated, the transform can be applied using the coefficients to the training and test datasets before evaluating the model.

If the coefficients are estimated using the entire dataset prior to splitting into train and test sets, there is a _small leakage_ of information from the test dataset to the training dataset which can result in estimates of model skill that are optimistically _biased_.

Thus, we may want to enhance the estimates of the coefficients with domain knowledge such as expected min/max values for all time in the future.

Generally, differencing does not suffer from the same problems. In most cases, such as one-step forecasting, the lag observations are available to perform the difference calculation. If not, the lag predictions can be used wherever needed as a proxy for the true observations in difference calculations.


### Order of Data Transforms

We may want to experiment with applying multiple data transforms to a time series prior to modeling.

It is quite common to;

- apply power transform to remove an increasing variance
- apply seasonal differencing to remove seasonality
- apply one-step differencing to remove a trend

**The order that the transform operations are applied is important.**

Intuitively, we can perceive how the transforms may interact.

  - Power transforms should be performed prior to differencing.

  - Seasonal differencing should be performed prior to one-step differencing.

  - Standardization is linear and should be performed on the sample after any nonlinear transforms and differencing.

  - Normalization is a linear operation but it should be the final transform performed to maintain the preferred scale.

A suggested ordering for data transforms is:

  1. Power Transform
  2. Seasonal Difference
  3. Trend Difference
  4. Standardization
  5. Normalization

Obviously, we would only use the transforms required for the specific dataset.

It is important to remember that when the transform operations are inverted, the order of the inverse transform operations must be reversed. 

Thus, the inverse operations must be performed in the following order:

  1. Normalization
  2. Standardization
  3. Trend Difference
  4. Seasonal Difference
  5. Power Transform


----------


## Training Time Series Forecasting Models in PyTorch

Here are some lessons and tips learned from training hundreds of PyTorch time series forecasting models in many different domains.

Before getting started, we should determine if the problem is actually a forecasting problem since this will guide how we should proceed.

Sometimes it might better to cast a forecasting problem as a classification problem. 

For example, if the exact number forecasted is not that important we could bucket it into ranges then use a classification model.

In addition, we should have some understanding of deployment and what the end product will look like. If we require millisecond latency for stock trading then a huge transformer model with 20 encoder layers probably will not function no matter what the test MAE.


- **Anomaly detection:** A general technique to detect outliers in time series data.

  Anomalies usually form a very small part of the dataset and are substantially different from other data points. Thus, anomaly detection can be seen as an extreme form of binary classification but it is usually treated as a separate area.

  Most anomaly detection methods are unsupervised since we are often unlikely to recognize anomalies until they occur (see reference).

  
- **Time Series Classification:** This is similar to other forms of classification where we take a time series and we want to classify it into a number of categories. 

  Unlike anomaly detection, we generally have a more balanced number of examples of each class (it may still be skewed such as 10%, 80%, 10%).


- **Time Series Forecasting:** In forecasting, we usually want to predict the next value or the next N values in a sequence of temporal data which is the focus of this article.


- **Time Series Prediction:** This term is ambiguous and could mean many things but it usually refers to either forecasting or classification.


- **Time Series Analysis:** A general umbrella term that can include all of the above. However, it is usually associated with just analyzing time series data and comparing different temporal structures than inherently designing a predictive model. 

  For example, if we developed a time series forecasting model then it could possibly tell us more about the casual factors in the time series and enable more time series analysis.

  
### Data Preprocessing

- **Always scale or normalize data:** Scaling or normalizing the data improves performance in most all uses cases. 

  Unless we have very small values then we should always normalize. 
  
  Flow Forecast has built-in scalers and normalizers that are easy to use. 
  
  Failure to scale the data can often cause the loss to explode, especially when training some transformers.
  

- **Double check for null, improperly encoded, or missing values:** Sometimes missing values are encoded in a unusual way. 

  For example, some weather stations encode missing precipitation values as -9999 which can cause a lot of problems as a regular NA check will not find them. 
  
  Flow Forecast provided a module for interpolating missing values and warning about possibly incorrectly entered values.

  
- **Start with a fewer number of features:** In general, it is easier to start with fewer features and add more, depending on performance.


### Model Choice and hyper-parameter selection

- **Visualize time lags to determine forecast_history:** In time series forecasting, the number of time-steps that we want to pass into the model will vary somewhat with architecture since some models are better at learning long-range dependencies but finding an initial range is useful. 

  In some cases, really long-term dependencies may not be useful.

  
- **Start with DA-RNN:** The DA-RNN model creates a very strong time series baseline. 

  Usually transformers can outperform DA-RNN, but they usually require more data and more careful hyperparameter tuning. 

  Flow Forecast provides an easy to use implementation of DA-RNN.

  
- **Determining a length to forecast:** The number of time steps the model forecasts at once is a tricky hyperparameter to determine what values to search. 

  We can still generate longer forecasts but we do this by appending the previous forecasts. 
  
  If the goal is to predict a long range of time steps, we may want them weighed into the loss function but having too many time steps at once can confuse the model. 
  
  In most hyper-parameter sweeps, a shorter forecast length works best.


- **Start with a low learning rate:** It is best to pick a low learning rate for most time series forecasting models.


- **Adam is not always the best:** Sometimes other optimizers may work better. 

  For example, BertAdam is good for transformer type models whereas for DA-RNN vanilla can work well.
  

### Robustness

- **Simulate and run play-by-play analysis of different scenarios:** 

  Flow Forecast makes it easy to simulate model performance under different conditions. 
  
  For example, if we are forecasting stream flows we can try inputting very large precipitation values and see how the model responds.


- **Double check heatmaps and other interpretability metrics:** 

  Sometimes we may think a model is performing well then we check the heatmap and see the model is not using the important features for forecasting. 
  
  When we perform further testing, it may become obvious that the model is jut learning to memorize rather than the actual casual effects of features.



## References

[5 Tips for Working With Time Series in Python](https://medium.com/swlh/5-tips-for-working-with-time-series-in-python-d889109e676d)

[4 Common Machine Learning Data Transforms for Time Series Forecasting](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/)

[Training Time Series Forecasting Models in PyTorch](https://towardsdatascience.com/training-time-series-forecasting-models-in-pytorch-81ef9a66bd3a)



[How NOT to Analyze Time Series](https://towardsdatascience.com/a-common-mistake-to-avoid-when-working-with-time-series-data-eedf60a8b4c1)

[How to Check if Time Series Data is Stationary with Python?](https://gist.github.com/codecypher/80d6bc9ade74a33ae8b5ca863662896d)

