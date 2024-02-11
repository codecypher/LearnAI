# Feature Engineering

[Data Preparation](./data_prep.md)

[Dimensionality Reduction](./dimensionality_reduction.md)


## Overview

**Feature engineering (FE)** is a subset of data transformation which is a critical element of data preprocessing [13].

There are three fundamenatal processes in FE [13]: 

1. Feature Extraction
2. Feature Selection
3. Feature Projection

Feature Projection (FP) is the process of mapping high-dimensional data to a lower-dimensional space which usually involves reducing the number of features fed to an ML algorithm [13]. 

FP can help reduce the complexity of the resulting model which reduces the chance of overfitting. 

FP can help reduce computational time and effort while not significantly affecting the model's accuracy.

There are two main classes of feature projection techniques: linear projection [linear discriminant analysis (LDA) and principal component analysis (PCA)] and non-linear projection [kernel principal component analysis (KPCA) and principal curves].


Feature engineering consists of various processes [1]:

- **Exploratory Data Analysis:** Exploratory data analysis (EDA) is a powerful and simple tool that can be used to improve your understanding of your data by exploring its properties. 

  EDA is often applied when the goal is to create hypotheses or find patterns in the data. 

  EDA is often used on large amounts of qualitative or quantitative data that have not been analyzed before.

- **Transformations:** Feature transformation is simply a function that transforms features from one representation to another. 

  The goal is to plot and visualize the data. If something is not adding up with the new features we can reduce the number of features used, speed up training, or increase the accuracy of a model.

  
- **Feature Selection:** The process of creating new variables that will be most helpful for our model which can include adding or removing some features. 

  Feature Selection (FS) is the process of choosing the most relevant features for the training process: correlation analysis, feature importance, and mutual information.
  
  Correlation analysis: The correlation coefficient measures the relationship between two variables and takes a value between -1 and +1. 
  
  A positive correlation means both variables move in the same direction (that is, as one increases, the other increases, and vice versa). 
  
  The larger the magnitude of the coefficient the stronger the correlation between the variables. 
  
  In feature selection, features having a higher correlation with the target variable are chosen because they have a higher predictive power.
  
  Feature importance: Some tree methods such as random forests and gradient-boosting algorithms provide feature importance scores that show the effect of each feature on the target prediction. These scores may be used the choose the most relevant features.
 
  Mutual information measures the reduction in the uncertainty of one variable based on the knowledge of another variable. A reduction in uncertainty results from having more information about the variable. Features with high mutual information scores are considered more relevant and are chosen for ML modeling.
  

- **Feature Extraction:** The process of extracting new features from a dataset to identify useful information which often involves domain-specific knowledge. 

  Without distorting the original relationships or other information, we compress the amount of data into manageable quantities for algorithms to process.


- **Benchmark:** A Benchmark Model is the most user-friendly, dependable, transparent, and interpretable model against which you can measure your final model. 

  It is a good idea to run test datasets to see if your new machine learning model outperforms a recognized benchmark which are often used as measures for comparing the performance of different ML models. 

  
-----



## Dimensionality Reduction

Dimensionality reduction refers to the process of reducing the number of attributes in a dataset while keeping as much of the variation in the original dataset as possible. 

Dimensionality reduction is a data preprocessing step, so it is done before training the model.

There are two main methods for reducing dimensionality:

- In **feature selection**, we only keep the most important features in the dataset and remove the redundant features. 

  There are no transformations applied to the set of features.

  Thus, feature selection selects a minimal subset of the variables that contain all predictive information necessary to produce a predictive model for the target variable (outcome).

  Examples: Backward elimination, Forward selection, and Random forests. 

  
- In **feature extraction**, we find a combination of new features and an appropriate transformation is applied to the set of features. 

  The new set of features contains different values rather than the original values. 

  Feature extraction can be further divided into _linear_ methods and _non-linear_ methods.


----------



## Feature Selection vs Feature Engineering

Dimensionality reduction can be done using feature selection methods as well as feature engineering methods.

_Feature selection_ is the process of identifying and selecting relevant features for a sample. 

_Feature engineering_ is manually generating new features from existing features by applying some transformation or performing some operation on them.



## Feature Importance

How to determine which features are the most important?

### Dataset loading and preparation

The article [2] has some code samples of data loading and preparation.

### Feature Importance Techniques

1. Obtain importances from correlation coefficients

2. Obtain importances from a tree-based model

When determining which features are the most important, algorithms such as Random Forest and ExtraTreesClassifier can be used.

3. Obtain importances from PCA loading scores

PCA is an algorithm used for dimensionality reduction based on the idea to choose features with high variance as high variance features contain more information. 



## Correlation

**Correlation** is a statistical measure that expresses the relation between two variables [10].


A positive correlation occurs when an increase in variable A leads to an increase in variable B. 

A negative correlation occurs when an increase in variable A leads to a decrease in variable B.

The range of correlation values is -1 to 1 where 1 represents completely, positively correlated features and -1 represents completely negatively correlated features.

Having two or more highly correlated features in our training data will lead to the problem of **multicollinearity** which affects model performance.



----------


## Feature Engineering Techniques

1. Convert text data features into vectors

Since ML is based on linear algebra, it is therefore necessary to convert textual data features into vectors. Techniques used to convert text data to features are Bag of Words(BOW), TF-IDF(Term Frequency-Inverse Document Frequency), Avg Word2Vec, and TF-IDF Word2Vec.

2. Modify features

Some features need to be modified using feature binning, mathematical transform, feature slicing, and indicator variable techniques for best results.

3. Create new features

New features can also be created using existing features such as the featurization of categorical data using one-hot encoding.


----------


## Scaling vs Normalization

The process of scaling and normalization are very similar. In both cases, you are transforming the values of numeric variables so that the transformed data points have specific helpful properties.

- In scaling, we are changing the _range_ of your data.
- In normalization, we are changing the _shape_ of the distribution of your data.

### Scaling

Some machine learning algorithms perform much better if all of the variables are scaled to the same range such as scaling all variables to values between 0 and 1 which is called normalization.

This effects algorithms that use a weighted sum of the input (such as linear models and neural networks) as well as models that use distance measures (such as support vector machines and k-nearest neighbors).

Therefore, it is a best practice to scale input data.

In scaling, you are transforming your data so that it fits within a specific scale such as 0-100 or 0-1.

It is common to have data where the scale of values differs from variable to variable. For example, one variable may be in feet and another in meters (pounds vs inches, kilograms vs meters).

By scaling your variables, you can help compare different variables on equal footing.

You especially want to scale data when you are using methods based on **measures of the distance between data points** such as support vector machines (SVM) or k-nearest neighbors (KNN). With these algorithms, a change of "1" in any numeric feature is given the same importance.

### Normalization

Scaling just changes the range of your data. 

The point of normalization is to change your observations so that they can be described as a _normal distribution_.

**Normal distribution:** This is a specific statistical distribution (bell curve) where a roughly equal number of observations fall above and below the mean, the mean and the median are the same, and there are more observations closer to the mean. The normal distribution is also known as the _Gaussian distribution_.

In general, you only want to normalize your data if you are going to be using a machine learning or statistics technique that assumes your data is normally distributed. 

Some examples are: t-tests, ANOVAs, linear regression, linear discriminant analysis (LDA), and Gaussian naive Bayes.

TIP: any method with "Gaussian" in the name probably assumes normality.

The method we are using to normalize here is called the Box-Cox Transformation.

Let us take a quick peek at what normalizing some data looks like:

Notice that the shape of our data has changed.

### Normalization vs Standardization

_Feature scaling_ is a crucial step in a preprocessing pipeline that can easily be forgotten.

Decision trees and random forests are one of the few machine learning algorithms where we do not need to worry about feature scaling.

The majority of machine learning and optimization algorithms behave much better if features are on the same scale. 

- **Normalization:** All values are scaled in a specified range between 0 and 1 via normalization or min-max scaling. 

- **Standardization:** The process of scaling values while accounting for standard deviation or z-score normalization.

  If the standard deviation of features differs, the range of those features will also differ. Therefore, the effect of outliers is reduced. 

  To arrive at a distribution with a 0 mean and 1 variance, all the data points are subtracted by their mean and the result divided by the distribution’s variance.

> Note that we fit the `StandardScaler` on the training data then use those parameters to transform the test set or any new data point.

> Regularization is another reason to use feature scaling such as standardization. For regularization to work properly, all features must be on comparable scales.


### Choose between Standardization vs Normalization

Data-centric heuristics include the following:

1. If your data has outliers, use standardization or robust scaling.
2. If your data has a gaussian distribution, use standardization.
3. If your data has a non-normal distribution, use normalization.

Model-centric rules include these:

1. If your modeling algorithm assumes (but does not require) a normal distribution of the residuals (such as regularized linear regression, regularized logistic regression, or linear discriminant analysis), use standardization.

2. If your modeling algorithm makes no assumptions about the distribution of the data (such as k-nearest neighbors, support vector machines, and artificial neural networks), then use normalization.

In each use case, the rule proposes a mathematical fit with either the data or the learning model. 


Normalization does not affect the feature distribution, but it does exacerbate the effects of outliers due to lower standard deviations. Thus, outliers should be dealt with prior to normalization.

Standardization can be more practical for many machine learning algorithms since many linear models such as logistic regression and SVM initialize the weights to 0 or small random values close to 0. Using standardization, we center the feature columns at mean 0 with standard deviation 1 so that the feature columns take the form of a normal distribution which makes it easier to learn the weights.

In addition, standardization maintains useful information about outliers and makes the algorithm less sensitive to them whereas min-max only scales the data to a limited range of values.



## Non-Normal Distribution of Data

It is important for most machine learning algorithms that features should follow a normal distribution since non-normal distribution of data affects model performance and generates wrong predictions [11].

There are many transformation methods that are used to convert non-normal distribution into a normal distribution [11]:

- Log Transform: we take the log of values of a particular feature.

- Box-Cox Transformation: Box-Cox transformation is a part of the power transformers family.

### Log Transform

Log Transform is the most used technique among data scientists to turn a skewed distribution into a normal or less-skewed distribution. 

We take the log of the values in a column and utilize those values as the column in this transform. 

Log transform is used to handle confusing data so that the data becomes more approximative to normal applications.


----------


## Normalization Techniques

Data Normalization is a typical practice in machine learning which consists of transforming numeric columns to a _standard scale_. 

Some feature values may differ from others multiple times. Therefore, the features with higher values will dominate the learning process [9].

One way to overcome this quirk is to use tree-based models like Random Forest, but if your problem requires the use of regularized linear models or neural networks then you should scale the feature ranges (say 0 to 1).


### Using maximum absolute scaling

The _maximum absolute_ scaling rescales each feature between -1 and 1 by dividing every observation by its maximum absolute value. 

We can apply the maximum absolute scaling in Pandas using the `.max()` and `.abs()` methods.

```py
    # copy the data
    df_max_scaled = df.copy()
      
    # apply normalization from scratch
    for column in df_max_scaled.columns:
        df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()
```

### Using min-max scaling

The _min-max_ scaling (normalization) rescales the feature to the range of [0, 1] by subtracting the minimum value of the feature then dividing by the range. 

We can use `MinMaxScaler` class from sklearn.

```py
    from sklearn.preprocessing import MinMaxScaler

    # define scaler
    scaler = MinMaxScaler()

    # transform data
    scaled = scaler.fit_transform(data)
```

We can apply the min-max scaling in Pandas using the `.min()` and `.max()` methods which preserves the column headers/names.

```py
    # copy the data
    df_min_max_scaled = df.copy()
      
    # apply normalization from scratch
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
```

We can also use `RobustScaler` when we want to reduce the effects of outliers compared to `MinMaxScaler`.


### Using z-score scaling

The _z-score_ scaling (standardization) transforms the data into a **normal (Gaussian) distribution** with a mean of 0 and a typical deviation of 1. Each standardized value is computed by subtracting the mean of the corresponding feature then dividing by the quality deviation.

We can apply standardization using `StandardScaler` class from sklearn.

```py
    from sklearn.preprocessing import StandardScaler

    # define scaler
    scaler = StandardScaler()

    # transform data
    scaled = scaler.fit_transform(data)
```

We can apply the standardization in Pandas using the `.mean()` and `.std()` methods which preserves the column headers/names.

```py
    # copy the data
    df_z_scaled = df.copy()
      
    # apply normalization from scratch
    for column in df_z_scaled.columns:
        df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()    
```


----------


## Transform Target Variables for Regression

Performing data preparation operations such as scaling is relatively straightforward for input variables and has been made routine in Python via the `Pipeline` scikit-learn class [4].

On regression predictive modeling problems where a numerical value must be predicted, it can also be crucial to scale and perform other data transformations on the target variable which can be achieved using the `TransformedTargetRegressor` class.

For regression problems, it is often desirable to scale or transform both the input and the target variables.

### Importance of Data Scaling

It is common to have data where the scale of values differs from variable to variable.

For example, one variable may be in feet, another in meters, etc.

Some machine learning algorithms perform much better if all of the variables are scaled to the same range, such as scaling all variables to values between 0 and 1 called normalization.

This effects algorithms that use a weighted sum of the input such as linear models and neural networks as well as models that use distance measures such as support vector machines and k-nearest neighbors.

Therefore, it is a good practice to scale input data and perhaps even try other data transforms such as making the data more normal (Gaussian probability distribution) using a power transform.

This also applies to output variables called _target_ variables such as numerical values that are predicted when modeling regression predictive modeling problems.

For regression problems, it is often desirable to scale or transform both the input and the target variables.

Scaling input variables is straightforward. In scikit-learn, you can use the scale objects manually or the more convenient `Pipeline` that allows you to chain a series of data transform objects together before using your model.

The `Pipeline` will fit the scale objects on the training data for you and apply the transform to new data, such as when using a model to make a prediction.

```py
    from sklearn.pipeline import Pipeline
    
    # prepare the model with input scaling
    pipeline = Pipeline(steps=[
        ('normalize', MinMaxScaler()), 
        ('model', LinearRegression())])
    
    # fit pipeline
    pipeline.fit(train_x, train_y)
    
    # make predictions
    yhat = pipeline.predict(test_x)
```

### How to Scale Target Variables

There are two ways that you can scale target variables:

1. Manually transform the target variable.
2. Automatically transform the target variable.

Manually managing the scaling of the target variable involves creating and applying the scaling object to the data manually.

1. Create the transform object such as `MinMaxScaler`.
2. Fit the transform on the training dataset.
3. Apply the transform to the train and test datasets.
4. Invert the transform on any predictions made.

```py
    # create target scaler object
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_y)

    # transform target variables
    train_y = target_scaler.transform(train_y)
    test_y = target_scaler.transform(test_y)

    # invert transform on predictions
    yhat = model.predict(test_X)
    yhat = target_scaler.inverse_transform(yhat)
```

However, if you use this approach then you cannot use convenience functions in scikit-learn such as `cross_val_score()` to quickly evaluate a model.

### Automatic Transform of Target Variable

An alternate approach is to automatically manage the transform and inverse transform by using the `TransformedTargetRegressor` object that wraps a given model and a scaling object.

It will prepare the transform of the target variable using the same training data used to fit the model, then apply that inverse transform on any new data provided when calling predict(), returning predictions in the correct scale.

```py
    # define the target transform wrapper
    wrapped_model = TransformedTargetRegressor(regressor=model, transformer=MinMaxScaler())

    # use the target transform wrapper
    wrapped_model.fit(train_X, train_y)
    yhat = wrapped_model.predict(test_X)

    # use the target transform wrapper
    wrapped_model.fit(train_X, train_y)
    yhat = wrapped_model.predict(test_X)
```

This is much easier and allows the use of helper functions such as `cross_val_score()` to evaluate a model.


## Complete Regression Example

```py
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.compose import TransformedTargetRegressor
 
    # load data
    dataset = np.loadtxt('housing.csv', delimiter=",")

    # split into inputs and outputs
    X, y = dataset[:, :-1], dataset[:, -1]
    
    # prepare the model with input scaling
    pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', HuberRegressor())])
    
    # prepare the model with target scaling
    model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
    
    # evaluate model
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    
    # convert scores to positive
    scores = np.absolute(scores)
    
    # summarize the result
    s_mean = np.mean(scores)
    print('Mean MAE: %.3f' % (s_mean))
```

We can also explore using other data transforms on the target variable such as the `PowerTransformer` to make each variable more Gaussian-like (using the Yeo-Johnson transform) and improve the performance of linear models.

By default, the `PowerTransformer` also performs a standardization of each variable after performing the transform.

```py
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import PowerTransformer
    from sklearn.compose import TransformedTargetRegressor

    # load data
    dataset = np.loadtxt('housing.csv', delimiter=",")

    # split into inputs and outputs
    X, y = dataset[:, :-1], dataset[:, -1]

    # prepare the model with input scaling
    pipeline = Pipeline(steps=[('power', PowerTransformer()), ('model', HuberRegressor())])

    # prepare the model with target scaling
    model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())

    # evaluate model
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

    # convert scores to positive
    scores = np.absolute(scores)

    # summarize the result
    s_mean = np.mean(scores)
    print('Mean MAE: %.3f' % (s_mean))
```


## Feature Selection in Scikit-learn

Simple ways to filter features for simpler and faster model [5].

It is often useful to filter out non-predictive features and keep the model lean so that the model is faster, easier to explain to stakeholders and simpler to deploy.

The article [5] five different ways to do feature selection for supervised machine learning problem.


Determining which features are most relevant to a particular task is not always straightforward. It may be helpful to use a feature selection technique to help identify features with less importance [10]. 

These feature selection techniques often rank features based on their importance in predicting an outcome. Then we can filter all the features using a threshold value for extracting the most important features.


## Distance Metrics

There are number of machine learning algorithms (both supervised and unsupervised) that use distance metrics to calculate the similarity between two data points [6] [7] [8]. 

The distance metrics calculate the distance between various data points available in the dataset. If the distance between two data points is less that means they are more likely to be similar and vice-versa.

There are several distance metrics that can be used, depending on the type of data that we have and the algorithm that we are using for our problem statement.

## Why can we learn from non-linear models?

Linear models (such as linear and logistic regression) are less complicated than non-linear models. 

To capture non-linear relationships with these models we need to do feature engineering [8] to create new features to represent non-linear relationships as linear ones. 

To understand what feature to create, we will need to explore our data and we will likely need domain knowledge.

In a sense, black-box models do this feature engineering within the model’s structure since black-box models are good at finding patterns in data. 

The problem is that the inner workings of the blackbox models are too complex, so we cannot understand how they make predictions by looking at the model parameters/ structure alone. Therefore, we need to use additional techniques to understand how they work which is where the field of interpretable machine learning (IML) comes in. 

IML aims to develop methods that we can use to understand how black-box models make predictions. 

### Important relationships

We can understand the important features in our data which are the features that can be used to predict the target variable. 

Most non-linear models are able to handle a large number of features (over 50) and it does not matter if those features are highly correlated. Therefore, we can understand the importance of a large set of features by using non-linear models. 

In comparison, models like linear regression require a smaller set (8–10) of uncorrelated features.

One measure we can use is **permutation feature importance**.

If we use correlation to identify important relationships we may miss some of these features since correlation only gives a measure of linear association [8]. 

In comparison, feature importance tells us how important a feature is to a model's predictions. A model's ability to predict could come from interactions and non-linear relationships which will be reflected in the feature importance score.

SHAP values can also be used to find the absolute mean SHAP value for each feature. Similar to feature importance, this can tell which features have made large contributions to predictions. The problem with both of these measures is they do not tell us the nature of the relationships captured by the model.

### non-linear relationships

To find and visualise non-linear relationships captured by a black-box model, we can use other IML methods.

Examples of visualisations are **partial dependence plots (PDP)** and **accumulated local effects (ALE)** plots. 

Both of these methods show the general relationship between a feature and the model’s predictions. 

- PDPs are easy to understand but they can be negatively impacted by feature dependencies. 

- ALE Plots do not have this problem but they are more complicated and harder to explain.

Another approach is to use SHAP values. For a given prediction, they give the contribution of each feature to that prediction. 

We can plot the SHAP values vs the feature values called a dependencies plot which can show the relationship between a feature’s value and its contributions to predictions.

### Interactions

Interactions are considered to be a type of non-linear relationship since they are non-linear w.r.t. to one feature. 

The relationship of the feature will depend on the value of a third feature. In fact, interactions can be even more complicated. 

The relationship of a feature could depend on the values of two or more features. For most applications, we are only concerned with relationships between two features.

One method of visualising interactions is **ICE Plots**. 

Another approach is to use **SHAP interaction values** which are an extension of SHAP values where both main and interaction effects are considered. We can use these values to visualise interactions. 

We can also calculate the mean absolute interaction effect to highlight important interactions which is similar to a feature importance score but for interactions. 

Another metric used to highlight interactions is the **Friedman’s h-statistic**.



## Time Series Data Preparation

[How to Normalize and Standardize Time Series Data in Python](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

[4 Common Machine Learning Data Transforms for Time Series Forecasting](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/)

[How to Scale Data for Long Short-Term Memory Networks in Python](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

### Order of Data Transforms for Time Series

You may want to experiment with applying multiple data transforms to a time series prior to modeling.

It is common to:

- apply a power transform to remove an increasing variance
- apply seasonal differencing to remove seasonality
- apply one-step differencing to remove a trend.

The order that the transform operations are applied is important.


## References

[1] [What is Feature Engineering?](https://towardsdatascience.com/what-is-feature-engineering-importance-tools-and-techniques-for-machine-learning-2080b0269f10?source=rss----7f60cf5620c9---4)

[2] [3 Essential Ways to Calculate Feature Importance in Python](https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155)


[3] [The Lazy Data Scientist’s Guide to AI/ML Troubleshooting](https://medium.com/@ODSC/the-lazy-data-scientists-guide-to-ai-ml-troubleshooting-abaf20479317?source=linkShare-d5796c2c39d5-1638394993&_branch_referrer=H4sIAAAAAAAAA8soKSkottLXz8nMy9bLTU3JLM3VS87P1Xcxy8xID4gMc8lJAgCSs4wwIwAAAA%3D%3D&_branch_match_id=994707642716437243)

[4][How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)

[5] [Feature Selection in Scikit-learn](https://towardsdatascience.com/feature-selection-in-scikit-learn-dc005dcf38b7?source=rss----7f60cf5620c9---4)

[6] [Guide to Distance Metrics in Machine Learning Algorithms](https://heartbeat.comet.ml/guide-to-distance-metrics-in-machine-learning-algorithms-33754203526f)

[7] [How to decide the perfect distance metric for your machine learning model](https://towardsdatascience.com/how-to-decide-the-perfect-distance-metric-for-your-machine-learning-model-2fa6e5810f11)

[8] [What we can Learn from Black-box Models](https://towardsdatascience.com/what-we-can-learn-from-black-box-models-eda3677c42f2)


[9] [Major Problems of Machine Learning Datasets: Part 1](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-1-5d5a06221c90)

[10] [Major Problems of Machine Learning Datasets: Part 2](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-2-ba82e551fee2)

[11] [Major Problems of Machine Learning Datasets: Part 3](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-3-eae18ab40eda)

[12] [How to Use Power Transforms for Machine Learning](https://machinelearningmastery.com/power-transforms-with-scikit-learn/)

[13] [3 Fundamental Processes in Feature Engineering](https://towardsdatascience.com/3-fundamental-processes-in-feature-engineering-d6b84983754)


[Best Practice to Calculate and Interpret Model Feature Importance](https://towardsdatascience.com/best-practice-to-calculate-and-interpret-model-feature-importance-14f0e11ee660)

[How to Perform Feature Selection in a Data Science Project](https://towardsdatascience.com/how-to-perform-feature-selection-in-a-data-science-project-591ba96f86eb)


[Representation: Feature Engineering](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)

[Basic Feature Discovering for Machine Learning](https://medium.com/diko-hary-adhanto-portfolio/basic-feature-discovering-for-machine-learning-cbd47bf4b651)

[Alternative Feature Selection Methods in Machine Learning](https://www.kdnuggets.com/2021/12/alternative-feature-selection-methods-machine-learning.html)


[Techniques for Dimensionality Reduction](https://towardsdatascience.com/techniques-for-dimensionality-reduction-927a10135356)

