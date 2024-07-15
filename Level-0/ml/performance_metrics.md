# Machine Learning Performance Metrics

There are several standard methods to evaluate the performance of ML models. In fact, scikit-learn has a plethora of functions for computing performance metrics in the `sklearn.metrics` library.

In AI, scoring functions such as MSE are used for three main purposes:

- Performance evaluation: How well is the model doing?

- Model optimization: Which model gets closest to our datapoints?

- Statistical decision-making: Does the model pass the hypothesis testing criteria?

## Performance Metrics For Classification

Here are the common performance measures for the  classification problems:

- Accuracy
- Precision and Recall
- F1 Score
- Logarithmic loss

- Classification report
- Confusion matrix (binary classification)

- Receiver operating curve (ROC)
- Area under ROC (AUC)

For a classification problem, it is common to compute the following metrics and graphs: Accuracy, classification report, confusion matrix, and AUC.

We can also compute the classification error (error rate) on the training and test datasets.

Accuracy = (TP + TN) / N = 1 - percent error

Error Rate = Percent Error = (FP + FN) / N = 1 - Accuracy

FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

<img width="405" alt="image" src="https://user-images.githubusercontent.com/2695661/138717691-255e8946-b87f-46fc-9e32-3562a95ec7ae.png">

## Performance Metrics For Regression

Here are the common performance measures for the  in regression problems:

- Accuracy
- Mean absolute error (MAE)
- Mean square error (MSE)
- Root mean squared error (RMSE)
- Mean absolute percentage error (MAPE)
- Mean percentage error (MPE)
- R^2

For a regression problem, it is common to compute all of the following metrics: Accuracy, MAE, RMSE, MAPE, and R^2.

Thus, we can compute the training error and test error using MAE, MSE, etc.

MAPE = relative error

<img width="337" alt="image" src="https://user-images.githubusercontent.com/2695661/138717800-7b11ef6d-a9e2-403a-bc91-79c5f9bcfd18.png">

If you are going to use a relative measure of error (MPE/MAPE) rather than an absolute measure of error (MAE/MSE), you must be wary of data that will work against the calculation (zeroes).

In mathematics, accuracy is a kind of _relative error_ which is undefined when the denominator (true value) is zero. Thus, you cannot compute accuracy when the true values can be zero. [CRC Mathematical Tables 33 ed, p. 646]

Therefore, you should use MAE and MSE to evaluate your model when the true value can be zero.

In a sense, MAE is similar to mean, MSE is similar to variance, and RMSE is similar to standard deviation.

When values can be zero, accuracy (MPE/MAPE) does not make sense. If you exclude the zero values then the calculation is meaningless (the error metric must be computed on all the samples).

## Compute Average Metrics

In general, ML models are _stochastic_ which means that you will get different results for each run or trial. Therefore, you must run several trials (say 10 or 100) and compute the average of several performance metrics.

```txt
Average metrics on 10 trials:

     train_acc  train_mae  train_rmse  train_mape  test_acc    test_mae   test_rmse  test_mape
0     0.966114  51.862437   61.589781    0.033886  0.975992   71.262462   90.968560   0.024008
1     0.989419  17.364505   23.621316    0.010581  0.979806   58.396311   79.787652   0.020194
2     0.988656  18.333088   24.960092    0.011344  0.978941   61.487530   84.414621   0.021059
3     0.983232  27.145072   32.716903    0.016768  0.966736   99.446103  119.955462   0.033264
4     0.987102  21.268157   28.235588    0.012898  0.963391  109.194594  129.714324   0.036609
5     0.981368  28.204099   36.886347    0.018632  0.979347   59.652040   83.378827   0.020653
6     0.947403  80.983944   92.722108    0.052597  0.973736   77.595416  102.693152   0.026264
7     0.986551  22.535717   29.933627    0.013449  0.957849  125.900201  147.230188   0.042151
8     0.983055  26.716388   33.393365    0.016945  0.972420   82.131250   98.375309   0.027580
9     0.989172  18.111209   24.411895    0.010828  0.978705   61.881334   82.536847   0.021295
avg   0.980207  31.252462   38.847102    0.019793  0.972692   80.694724  101.905494   0.027308
```

----------

## Confidence Intervals

[How to Calculate Bootstrap Confidence Intervals For Machine Learning Results in Python](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)

[How to Report Classifier Performance with Confidence Intervals](https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/)

It is important to present the expected skill of a machine learning model as well as confidence intervals for the model.

Confidence intervals are a way of quantifying the uncertainty of an estimate which can be used to add a bounds or likelihood on a population parameter such as mean or standard deviation that is estimated from a sample of independent observations from the population.

Confidence intervals provide a range of model skills and a likelihood that the model will fall between the ranges when making predictions on new data.

A confidence interval could be used in presenting the skill of a classification model:

Given the sample, there is a 95% likelihood that the range x to y covers the true model accuracy.

OR

The accuracy of the model was x +/- y at the 95% confidence level.

A confidence interval can also be used in the presentation of the error of a regression predictive model.

There is a 95% likelihood that the range x to y covers the true error of the model.

OR

The error of the model was x +/- y at the 95% confidence level.

Confidence intervals belong to a field of statistics called estimation statistics that can be used to present and interpret experimental results instead of or in addition to statistical significance tests.

Confidence intervals may be preferred in practice over the use of statistical significance tests since they are easier for practitioners and stakeholders to relate directly to the domain; they can also be interpreted and used to compare machine learning models.

### Classifier Error with Confidence Intervals

Rather than presenting just a single error score, a confidence interval can be calculated and presented as part of the model skill.

A confidence interval is comprised of two things:

- Range. This is the lower and upper limit on the skill that can be expected on the model.

- Probability. This is the probability that the skill of the model will fall within the range.

In general, the confidence interval for classification error can be calculated as follows:

```txt
  error +/- const * sqrt( (error * (1 - error)) / n)
```

where error is the classification error, const is a constant value that defines the chosen probability, sqrt is the square root function, and n is the number of observations (rows) used to evaluate the model. Technically, this is called the Wilson score interval.

The values for const are provided from statistics, and common values used are:

```txt
  1.64 (90%)
  1.96 (95%)
  2.33 (98%)
  2.58 (99%)
```

Use of these confidence intervals makes some assumptions that you need to ensure you can meet:

- Observations in the validation data set were drawn from the domain independently (e.g. they are independent and identically distributed).

- At least 30 observations were used to evaluate the model.

This is based on some statistics of sampling theory that takes calculating the error of a classifier as a binomial distribution, that we have sufficient observations to approximate a normal distribution for the binomial distribution, and that via the central limit theorem that the more observations we classify, the closer we will get to the true (but unknown) model skill.

### Validation Dataset

What dataset do you use to calculate model skill?

It is a good practice to hold out a validation dataset from the modeling process.

This means a sample of the available data is randomly selected and removed from the available data, such that it is not used during model selection or configuration.

After the final model has been prepared on the training data, it can be used to make predictions on the validation dataset which are used to calculate a classification accuracy or classification error.

### Confidence Interval Example

Consider a model with an error of 0.02 (error = 0.02) on a validation dataset with 50 examples (n = 50).

We can calculate the 95% confidence interval (const = 1.96) as follows:

```mathml
  error +/- const * sqrt( (error * (1 - error)) / n)

  0.02 +/- 1.96 * sqrt( (0.02 * (1 - 0.02)) / 50)

  0.02 +/- 0.0388
```

Or stated another way:

There is a 95% likelihood that the confidence interval [0.0, 0.0588] covers the true classification error of the model on unseen data.

Notice that the confidence intervals on the classification error must be clipped to the values 0.0 and 1.0. It is impossible to have a negative error (e.g. less than 0.0) or an error more than 1.0.

### Nonparametric Confidence Interval

Often we do not know the distribution for a chosen performance measure. In this case, we may not know the analytical way to calculate a confidence interval for a skill score.

In these cases, the _bootstrap resampling method_ can be used as a nonparametric method for calculating confidence interval called bootstrap confidence intervals.

The _bootstrap_ is a simulated Monte Carlo method where samples are drawn from a fixed finite dataset with replacement and a parameter is estimated on each sample which leads to a robust estimate of the true population parameter via sampling.

_Bootstrapping_ is any test or metric that uses random sampling with replacement and falls under the broader class of resampling methods.

Bootstrapping estimates the properties of a population parameter or estimator (bias, variance, confidence intervals, prediction error, etc.) by measuring those properties when sampling from an approximating distribution using random sampling methods.

```py
  # calculate 95% confidence intervals (100 - alpha)
  alpha = 5.0
```

```txt
50th percentile (median) = 0.750
2.5th percentile = 0.741
97.5th percentile = 0.757
```

There is a 95% likelihood that the range 0.741 to 0.757 covers the true statistic mean.

## References

[1]: [Performance analysis of models](https://mclguide.readthedocs.io/en/latest/sklearn/performance.html)

[2]: [Evaluation Metrics for Machine Learning](https://towardsdatascience.com/evaluation-metrics-for-machine-learning-2167fca1a291?gi=2512e2b9b1c0)

[3]: [Error Metrics in Machine learning](https://medium.com/analytics-vidhya/error-metrics-in-machine-learning-f9eed7b139f)

[4]: [An Overview of Performance Evaluation Metrics of Machine Learning (Classification) Algorithms](https://towardsdatascience.com/an-overview-of-performance-evaluation-metrics-of-machine-learning-classification-algorithms-7a95783a762f?gi=884943f12b27)

[5]: [Tutorial: Understanding Regression Error Metrics in Python](https://www.dataquest.io/blog/understanding-regression-error-metrics/)

[6]: [The Complete Guide to Keras Loss Functions](https://python.plainenglish.io/the-complete-guide-to-keras-loss-functions-776d319e729d)

[7]: [Keras Loss Functions: Everything You Need to Know](https://neptune.ai/blog/keras-loss-functions)

### Confidence Interval References

[Understand Time Series Forecast Uncertainty Using Prediction Intervals with Python](https://machinelearningmastery.com/time-series-forecast-uncertainty-using-confidence-intervals-python/)

[Confidence Intervals for Machine Learning](https://machinelearningmastery.com/confidence-intervals-for-machine-learning/)

[PSI and CSI: Top 2 model monitoring metrics](https://towardsdatascience.com/psi-and-csi-top-2-model-monitoring-metrics-924a2540bed8?gi=7d4c901abece)

----------

[Approximation Error](https://en.wikipedia.org/wiki/Approximation_error?wprov=sfti1)

[Assessing the Performance (Types and Sources of Error) in Machine Learning](https://medium.com/analytics-vidhya/assessing-the-performance-types-and-sources-of-error-in-machine-learning-e5d28b71da6b)

[Accuracy and Error Rate from Confusion Matrix](https://medium.com/analytics-vidhya/why-do-we-need-a-confusion-matrix-73bf8a2acf09)

[What does RMSE really mean?](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)

[Stop Using Accuracy to Evaluate Your Classification Models](https://towardsdatascience.com/evaluating-ml-models-with-a-confusion-matrix-3fd9c3ab07dd)
