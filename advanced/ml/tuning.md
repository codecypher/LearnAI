# Hyperparameter Tuning

## Overview

The process of searching for optimal hyperparameters is called hyperparameter tuning or _hypertuning_ which is essential in any machine learning project. 

Hypertuning helps boost performance and reduces model complexity by removing unnecessary parameters (such as number of units in a dense layer).


> The first rule of optimization is to not do it. If you really have to then use profiling tools to find bottlenecks and optimize where appropriate.


There are two types of hyperparameters:

- Model hyperparameters: influence model architecture (such as number and width of hidden layers in a DNN)

- Algorithm hyperparameters: influence the speed and quality of training (such as learning rate and activation function).


A practical guide to hyperparameter optimization using three methods: grid, random and bayesian search (with skopt) [1]. 

1. Introduction to hyperparameter tuning.

2. Explanation about hyperparameter search methods.

3. Code examples for each method.

4. Comparison and conclusions.


### Simple pipeline used in all the examples

```py
    from sklearn.pipeline import Pipeline #sklearn==0.23.2
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import make_column_transformer
    from lightgbm import LGBMClassifier
    
    tuples = list()
    
    tuples.append((Pipeline([
            ('scaler', StandardScaler()),
        ]), numeric_var))
    
    tuples.append((Pipeline([
            ('onehot', OneHotEncoder()),
        ]), categorical_var))
    
    preprocess = make_column_transformer(*tuples)
    
    pipe = Pipeline([
        ('preprocess', preprocess),
        ('classifier', LGBMClassifier())
        ])
```


### Grid Search

The basic method to perform hyperparameter tuning is to try all the possible combinations of parameters.

Here we try to find the best values for learning_rate (5 values), max_depth (5 values), and n_estimators (5 values) — 125 iterations in Total.

### Random Search

in randomized search, only part of the parameter values are evaluated. 

The parameter values are sampled from a given list or specified distribution. 

The number of parameter settings that are sampled is given by `n_iter`. 

Sampling without replacement is performed when the parameters are presented as a list (similar to grid search), but if the parameter is given as a distribution then sampling with replacement is used (recommended).

The advantage of randomized search is that you can extend your search limits without increasing the number of iterations. You can also use random search to find narrow limits to continue a thorough search in a smaller area.


### Bayesian Search

The main difference with Bayesian search is that the algorithm optimizes its parameter selection in each round according to the previous round score [1]. 

Therefore, the algorithm optimizes the choice and theoretically reaches the best parameter set faster than the other methods which means that this method will choose only the relevant search space and discard the range of values that will most likely not deliver the best solution. 

Thus, Bayesian search can be beneficial when you have a large amount of data and/or the learning process is slow and you want to minimize the tuning time.

```py
    from skopt import BayesSearchCV
    
    # Bayesian
    n_iter = 70
    
    param_grid = {
        "classifier__learning_rate": (0.0001, 0.1, "log-uniform"),
        "classifier__n_estimators": (100,  1000) ,
        "classifier__max_depth": (4, 400) 
    }
    
    reg_bay = BayesSearchCV(estimator=pipe,
                        search_spaces=param_grid,
                        n_iter=n_iter,
                        cv=5,
                        n_jobs=8,
                        scoring='roc_auc',
                        random_state=123)
                        
    model_bay = reg_bay.fit(X, y)
```


### Visualization of parameter search (learning rate)


### Visualization of mean score for each iteration



## AutoML Tools for Tuning

### KerasTuner

KerasTuner is an easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search. 

The process of selecting the right set of hyperparameters for your machine learning (ML) application is called _hyperparameter tuning_ or _hypertuning_.

Hyperparameters are the variables that govern the training process and the topology of an ML model. These variables remain constant over the training process and directly impact the performance of your ML program. Hyperparameters are of two types:

- **Model hyperparameters** which influence model selection such as the number and width of hidden layers

- **Algorithm hyperparameters** which influence the speed and quality of the learning algorithm such as the learning rate for Stochastic Gradient Descent (SGD) and the number of nearest neighbors for a k Nearest Neighbors (KNN) classifier



### Optuna

Optuna is an automatic hyperparameter optimization software framework designed for machine learning. 

Optuna is framework agnostic, so you can use it with any machine learning or deep learning framework



## Improve Model Performance

Here we discuss some ways to improve the accuracy of machine learning models. 

1. Handling Missing Values and Outliers

One of the easiest ways to improve the accuracy of your machine learning models is to handle missing values and outliers.

If you have data that is missing values or contains outliers, your models will likely be less accurate. This is because missing values and outliers can cause the model to make incorrect assumptions about your data.

Its is important to note that missing values and outliers can cause your models to overfit or underfit!

There are a number of ways that you can handle missing values and outliers:

- Remove the data points that contain missing values or outliers from your training dataset.

- Impute the missing values using a technique like k-nearest neighbors or linear regression.

- Use a technique like bootstrapping to remove the influence of the outlier data


2. Feature Engineering

Feature engineering is the art of creating new features from your existing ones.

Feature engineering helps improve the accuracy of machine learning models by allowing them to make more accurate predictions.

One of the most common ways to create new features is by combining multiple existing features into one or more new features.

There are many different ways to engineer features, and the best way to do it often depends on the dataset: 

- Try to find correlations between different features and create new ones that capture these relationships.

- Use transforms like logarithmic transformation or standardization to make your features more comparable and easier to work with.

- Make use of data pre-processing techniques like feature extraction and selection to help you find the most important features in your dataset.


3. Feature Selection

Feature selection is a process that helps you identify the most useful features in your dataset.

The goal is to reduce or eliminate noise and improve the accuracy of machine learning models by removing redundant information from them such as data points containing only one feature.

There are many different ways to select features, but they all involve using some form of statistical analysis or filtering out features with low importance scores (those that do not contribute much to model accuracy).

Some common techniques for feature selection include:

- Ranking features based on their correlation with other variables in the dataset, then removing those that are less correlated than others. For example, you could use the Pearson Correlation Coefficient to measure the strength of the relationship between two variables.

- Filtering features based on their importance scores, which are usually calculated using a technique like gradient descent or random forests.

- Selecting a subset of features that have a high correlation with the target variable but low correlations among themselves (i.e., they are uncorrelated or independent of each other).


4. Try Multiple Algorithms

A common mistake is to only try one algorithm when training your model. 

There will likely be some features in your dataset that do not contribute much to the accuracy of the model and removing them will only make things worse which is where multiple algorithms can be helpful.

By trying different algorithms, you can identify which ones work best for your data and then use that information to improve the accuracy of your models.

There are many different types of machine learning algorithms, so it can be difficult to know which ones are right for your data. A good place to start is by using cross-validation with multiple algorithms on the same dataset and then comparing their accuracy scores against each other.

If you are using scikit-learn, it has a nice list of common machine learning models that you can try out on your data including:

- Linear Regression
- Support Vector Machines
- Decision Trees
- Random Forests
- Neural Networks
- Ensemble Models

Another approach is to use an ensemble method, which combines two or more algorithms together into one model. Ensembles are often more accurate than any individual algorithm because they leverage the strengths of each and compensate for their weaknesses.

Thus, you can combine multiple weak learners (models that perform poorly on their own) into one ensemble to get a stronger learner (a model that performs well as an individual).


5. Adjusting Hyperparameters

Hyperparameters are the parameters in machine learning models which include things such as number of layers in a deep neural network or how many trees there should be in an ensemble model.

We usually need to adjust the hyperparameters since they are not automatically set during model training and cross-validation can be helpful. 

By splitting the data into training and test sets, we can try different combinations of hyperparameters on the training set to see how well they perform on the test set to find the best combination of hyperparameters for the model.

We can also use grid search which is a method of finding the optimal combination of hyperparameters for a model.

Grid search searches all the possible combinations of parameters to find one that provides the best performance on your metric such as accuracy. Then, we can then use that combination of hyperparameters to train the model.

We can use Grid Search in the scikit-learn library.


## Pruning in Tensorflow

Tensorflow provides Model Optimization Toolkit for pruning and other post-training optimizations. 

[Accelerate your training and inference running on Tensorflow](https://towardsdatascience.com/accelerate-your-training-and-inference-running-on-tensorflow-896aa963aa70)



## References

[1] [Hyperparameter Tuning Methods](https://towardsdatascience.com/bayesian-optimization-for-hyperparameter-tuning-how-and-why-655b0ee0b399)

[2] [A Practical Introduction to Grid Search, Random Search, and Bayes Search](https://towardsdatascience.com/a-practical-introduction-to-grid-search-random-search-and-bayes-search-d5580b1d941d)


[3] [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

[4] [Hyperparameter Tuning with KerasTuner and TensorFlow](https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a)

[5] [Introduction to the Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[6] [How to Grid Search Deep Learning Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-grid-search-deep-learning-models-for-time-series-forecasting/)


[5 Effective Ways to Improve the Accuracy of Your Machine Learning Models](https://towardsdatascience.com/5-effective-ways-to-improve-the-accuracy-of-your-machine-learning-models-f1ea1f2b5d65)

[Are You Sure That You Can Implement Image Classification Networks?](https://pub.towardsai.net/are-you-sure-that-you-can-implement-image-classification-networks-d5f0bffb242d)

[Profiling Neural Networks to improve model training and inference speed](https://pub.towardsai.net/profiling-neural-networks-to-improve-model-training-and-inference-speed-22be473492bf)

[How to Speed up Scikit-Learn Model Training](https://medium.com/distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1)

[How to Speed Up XGBoost Model Training](https://towardsdatascience.com/how-to-speed-up-xgboost-model-training-fcf4dc5dbe5f?source=rss----7f60cf5620c9---4)


[^sklearn_hypertuning]: https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/ "Scikit-Optimize for Hyperparameter Tuning in Machine Learning"

