# Guide to Classification and Regression

## Overview

Linear regression is a good starting point when dealing with regression problems and can be considered the _baseline_ model. 

Linear regression finds the optimal linear relationship between independent variables and dependent variables in order to make predictions.

There are two main types:

- Simple regression (y = mx + b)
- Multivariable regression


## Exploratory Data Analysis (EDA)

EDA is essential to both investigate the data quality and reveal hidden correlations among variables.

1. Univariate Analysis

Visualize the data distribution using histogram for numeric variables and bar chart for categorical variables.

### Why do we need univariate analysis?

- Determine if dataset contains outliers

- Detrmine if we need data transformations or feature engineering

In this case, we found out that “expenses” follows a power law distribution, which means that log transformation is required as a step of feature engineering step, to convert it to normal distribution.

2. Multivariate Analysis

When thinking of linear regression, the first visualization technique that we can think of is scatterplot. 

By plotting the target variable against the independent variables using a single line of code `sns.pairplot(df)`, the underlying linear relationship becomes more evident.

3. Correlation Analysis

Correlation analysis examines the linear correlation between variable pairs which can be achieved by combining `corr()` function with `sns.heatmap()`. 

### Why do we need correlation analysis?

- To identify collinearity between independent variables — linear regression assumes no collinearity among independent features, so it is essential to drop some features if collinearity exists. 

- To identify independent variables that are strongly correlated with the target — strong predictors.


## Feature Engineering

EDA brought some insights of what types of feature engineering techniques are suitable for the dataset.

1. Log Transformation

We  found that the target variable  “expenses” is right skewed and follows a power law distribution. 

Since linear regression assumes linear relationship between input and output variable, it is necessary to use log transformation to “expenses” variable. 

As shown below, the data tends to be more normally distributed after applying `np.log2()`.

2. Encoding Categorical Variable

Another requirement of machine learning algorithms is to encode categorical variable into numbers.

Two common methods are one-hot encoding and label encoding. 


## Model Implementation

A simple linear regression y = b0 + b1x predicts relationship between one independent variable x and one dependent variable y. 

As more features/independent variables are introduced, it becomes multiple linear regression y = b0 + b1x1 + b2x2 + ... + bnxn, which cannot be easily plotted using a line in a two dimensional space.

Here we use `LinearRegression()` class from scikit-learn to implement the linear regression. 

We specify `normalize = True` so that independent variables will be normalized and transformed into same scale. 

Note that scikit-learn linear regression utilizes **Ordinary Least Squares** to find the optimal line to fit the data which means the line, defined by coefficients b0, b1, b2 … bn, minimizes the residual sum of squares between the observed targets and the predictions (the blue lines in chart). 


## Model Evaluation

Linear regression model can be qualitatively evaluated by visualizing error distribution. 

There are also quantitative measures such as MAE, MSE, RMSE and R squared.

1. Error Distribution

Here we use a histogram to visualize the distribution of error which should somewhat conform to a normal distribution. 

A non-normal error distribution may indicates that there is non-linear relationship that model failed to pick up or more data transformations are needed.

2. MAE, MSE, RMSE

All three methods measure the errors by calculating the difference between predicted values ŷ and actual value y, so the smaller the better. 

The main difference is that MSE/RMSE penalized large errors and are differentiable whereas MAE is not differentiable which makes it hard to apply in gradient descent. 

Compared to MSE, RMSE takes the square root which maintains the original data scale.

3. R Squared

R squared or _coefficient of determination_ is a value between 0 and 1 that indicates the amount of variance in actual target variables explained by the model. 

R squared is defined as 1 — RSS/TSS which is 1 minus the ratio between sum of squares of residuals (RSS) and total sum of squares (TSS). 

Higher R squared means better model performance.

In this case, a R squared value of 0.78 indicating that the model explains 78% of variation in target variable, which is generally considered as a good rate and not reaching the level of overfitting.


----------


## Guide to Logistic Regression

Logistic regression is a good starting point when dealing with classification problems and can be considered the _baseline_ model. 

Supervised learning refers to machine learning that is based on a training set of labeled examples. 

A supervised learning model trains on a dataset containing features that explain a target.

Here we review the following: 

- Logistic Regression (binary classification)

- Sigmoid Function

- Decision Boundaries for Multi-Class Problems

- Fitting Logistic Regression Models in Python

- Classification Error Metrics

- Error Metrics in Python


### Terminology

The parameters of the train function are called _hyperparameters_ such as iterations and learning rate which are set so that the train function can find parameters such as w and b.


### Limitations of Logistic Regression

Logistic regression is a simple and powerful linear classification algorithm, hut it has limitations that suggest the need for alternate linear classification algorithms [7]:

- Two-Class Problems. Logistic regression is intended for two-class or binary classification problems. It can be extended for multi-class classification, but is rarely used for this purpose.

- Unstable With Well Separated Classes. Logistic regression can become unstable when the classes are well separated.

- Unstable With Few Examples. Logistic regression can become unstable when there are few examples from which to estimate the parameters.

Linear Discriminant Analysis (LDA) attempts to address each of these points and is the go-to linear method for multi-class classification problems. 

Even with binary-classification problems, it is a good idea to try both logistic regression and linear discriminant analysis.



## Linear Discriminant Analysis

### Representation of LDA Models

The representation of Linear Discriminant Analysis (LDA) consists of statistical properties of your data, calculated for each class [7]. 

For a single input variable (x) this is the mean and the variance of the variable for each class. 

For multiple variables, this is the same properties calculated over the multivariate Gaussian - the means and covariance matrix.

These statistical properties are estimated from your data and plug into the LDA equation to make predictions. These are the model values that you would save to file for your model.

### Learning LDA Models

LDA makes some simplifying assumptions about your data:

1. The data is Gaussian - each variable is shaped like a bell curve when plotted.

2. Each attribute has the same variance - values of each variable vary around the mean by the same amount on average.

With these assumptions, the LDA model estimates the mean and variance from your data for each class. 

It is easy to think about this in the univariate (single input variable) case with two classes.

### Making Predictions with LDA

LDA makes predictions by estimating the probability that a new set of inputs belongs to each class. 

The class that gets the highest probability is the output class and a prediction is made.

The model uses **Bayes Theorem** to estimate the probabilities. 

Bayes’ Theorem can be used to estimate the probability of the output class (k) given the input (x) using the probability of each class and the probability of the data belonging to each class:

The f(x) above is the estimated probability of x belonging to the class. 

A Gaussian distribution function is used for f(x). 


### How to Prepare Data for LDA

This section lists some suggestions you may consider when preparing your data for use with LDA.

- Classification Problems. LDA is intended for classification problems where the output variable is categorical. LDA supports both binary and multi-class classification.

- Gaussian Distribution. The standard implementation of the model assumes a Gaussian distribution of the input variables. 

  Consider reviewing the univariate distributions of each attribute and using transforms to make them more Gaussian-looking (such as log and root for exponential distributions and Box-Cox for skewed distributions).

- Remove Outliers. Consider removing outliers from your data that can skew the basic statistics used to separate classes in LDA such the mean and the standard deviation.

- Same Variance. LDA assumes that each input variable has the same variance. 

  It is always a good idea to standardize your data before using LDA so that it has a mean of 0 and a standard deviation of 1.
  
### Extensions to LDA

Linear Discriminant Analysis is a simple and effective method for classification. 

Since it is simple and so well understood, there are many extensions and variations to the method [7]:

- Quadratic Discriminant Analysis (QDA): Each class uses its own estimate of variance (or covariance when there are multiple input variables).

- Flexible Discriminant Analysis (FDA): Where non-linear combinations of inputs is used such as splines.

- Regularized Discriminant Analysis (RDA): Introduces regularization into the estimate of the variance (actually covariance), moderating the influence of different variables on LDA.

The original development was called the Linear Discriminant or Fisher’s Discriminant Analysis. 

The multi-class version was referred to Multiple Discriminant Analysis. 

These are all now referred to as Linear Discriminant Analysis.

  
----------



## Classification

Machine learning is the field of study concerned with algorithms that learn from examples.

Classification is a task that requires the use of machine learning algorithms that learn how to assign a class label to examples from the problem domain [6].

There are many different types of classification tasks that you may encounter in machine learning and specialized approaches to modeling that may be used for each.


### Classification Predictive Modeling

In machine learning, classification refers to a predictive modeling problem where a class label is predicted for a given example of input data [6].

Examples of classification problems:

- Given an example, classify if it is spam or not.

- Given a handwritten character, classify it as one of the known characters.

- Given recent user behavior, classify as churn or not.

From a modeling perspective, classification requires a training dataset with many examples of inputs and outputs from which to learn.

A model will use the training dataset and will calculate how to best map examples of input data to specific class labels. As such, the training dataset must be sufficiently representative of the problem and have many examples of each class label.

Class labels are often string values such as “spam” or “not spam” that must be mapped to numeric values before being provided to an algorithm for modeling which is often referred to as _label encoding_ where a unique integer is assigned to each class label such as “spam” = 0 and “no spam” = 1.

There are many different types of classification algorithms for modeling classification predictive modeling problems.

There is no good theory on how to map algorithms onto problem types; it is generally recommended that a practitioner use controlled experiments and discover which algorithm and algorithm configuration results in the best performance for a given classification task.

Classification predictive modeling algorithms are evaluated based on their results. 

Classification _accuracy_ is a popular metric used to evaluate the performance of a model based on the predicted class labels. 

Classification accuracy is not perfect but is a good starting point for many classification tasks.

Instead of class labels, some tasks may require the prediction of a probability of class membership for each example which provides additional uncertainty in the prediction that an application or user can then interpret. 

A popular diagnostic for evaluating predicted probabilities is the ROC Curve.

There are four main types of classification tasks that you may encounter [6]:

- **Binary Classification** refers to classification tasks that have two class labels.

- **Multi-Class Classification** refers to classification tasks that have more than two class labels. 

- **Multi-Label Classification** refers to classification tasks that have two or more class labels where one or more class labels may be predicted for each example.

- **Imbalanced Classification** refers to classification tasks where the number of examples in each class is unequally distributed.



## Classification Examples

[End-to-End Machine Learning Workflow (Part 1)](https://medium.com/mlearning-ai/end-to-end-machine-learning-workflow-part-1-b5aa2e3d30e2)

[End-to-End Machine Learning Workflow (Part 2)](https://medium.com/mlearning-ai/end-to-end-machine-learning-workflow-part-2-e7b6d3fb1d53)

[Regression for Classification](https://towardsdatascience.com/regression-for-classification-hands-on-experience-8754a909a298)


[An End-to-End Machine Learning Project — Heart Failure Prediction](https://towardsdatascience.com/an-end-to-end-machine-learning-project-heart-failure-prediction-part-1-ccad0b3b468a?gi=498f31004bdf)

[One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)


[Classification And Regression Trees for Machine Learning](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)

[Building a Random Forest Classifier to Predict Neural Spikes](https://medium.com/@mzabolocki/building-a-random-forest-classifier-for-neural-spike-data-8e523f3639e1)

[KNN Algorithm for Classification and Regression: Hands-On With Scikit- Learn](https://cdanielaam.medium.com/knn-algorithm-for-classification-and-regression-hands-on-with-scikit-learn-4c5ec558cdba)


## Classification Examples using Keras

[Structured data classification from scratch](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/)

[Image classification with modern MLP models](https://keras.io/examples/vision/mlp_image_classification/)

[3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification/)


## Regression Examples

[A Beginner’s Guide to End to End Machine Learning](https://towardsdatascience.com/a-beginners-guide-to-end-to-end-machine-learning-a42949e15a47?gi=1736097101b9)

[A Practical Guide to Linear Regression](https://towardsdatascience.com/a-practical-guide-to-linear-regression-3b1cb9e501a6?gi=ba29357dcc8)

[A Practical Introduction to 9 Regression Algorithms](https://towardsdatascience.com/a-practical-introduction-to-9-regression-algorithms-389057f86eb9)




## References

[1] [A Practical Guide to Linear Regression](https://towardsdatascience.com/a-practical-guide-to-linear-regression-3b1cb9e501a6?gi=ba29357dcc8)

[2] [End-to-End Machine Learning Workflow](https://medium.com/mlearning-ai/end-to-end-machine-learning-workflow-part-1-b5aa2e3d30e2)

[3] [Essential guide to Multi-Class and Multi-Output Algorithms in Python](https://satyam-kumar.medium.com/essential-guide-to-multi-class-and-multi-output-algorithms-in-python-3041fea55214)

[4] [Supervised Machine Learning: Classification, Logistic Regression, and Classification Error Metrics](https://medium.com/the-quant-journey/supervised-machine-learning-classification-logistic-regression-and-classification-error-metrics-6c128263ac64?source=rss------artificial_intelligence-5)

[5] [Improve Your Classification Models With Threshold Tuning](https://pub.towardsai.net/improve-your-classification-models-with-threshold-tuning-bb69fca15114)

[6] [4 Types of Classification Tasks in Machine Learning](https://machinelearningmastery.com/types-of-classification-in-machine-learning/)

[7] [Linear Discriminant Analysis for Machine Learning](https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/)


[Five Regression Python Modules That Every Data Scientist Must Know](https://towardsdatascience.com/five-regression-python-modules-that-every-data-scientist-must-know-a4e03a886853)

[How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)


[A Gentle Introduction to Multiple-Model Machine Learning](https://machinelearningmastery.com/multiple-model-machine-learning/)

