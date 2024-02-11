# Imbalanced Classification Framework

[Step-By-Step Framework for Imbalanced Classification Projects](https://machinelearningmastery.com/framework-for-imbalanced-classification-projects/)

In this tutorial, you will discover a systematic framework for working through an imbalanced classification dataset.

## Tutorial Overview

This tutorial is divided into three parts; they are:

1. What Algorithm To Use?
2. Use a Systematic Framework
3. Detailed Framework for Imbalanced Classification
  3.1 Select a Metric
  3.2 Spot Check Algorithms
  3.3 Spot Check Imbalanced Algorithms
  3.4 Hyperparameter Tuning

## Use a Systematic Framework

Consider a balanced classification task.

You are faced with the same challenge of selecting which algorithm to use to address your dataset.

Once you have a dataset, the process involves three steps:

1. Select a Metric
2. Spot Check Algorithms
3. Hyperparameter Tuning

We can use the same three-step procedure and insert an additional step to evaluate imbalanced classification algorithms.

We can summarize this process as follows:

1. Select a Metric
2. Spot Check Algorithms
3. Spot Check Imbalanced Algorithms
4. Hyperparameter Tuning

This provides a high-level systematic framework to work through an imbalanced classification problem.

## Select a Metric

Selecting a metric might be the most important step in the project.

The metric must capture those details about a model or its predictions that are most important to the project or project stakeholders.

First, you must decide whether you want to predict probabilities or crisp class labels. 

For binary imbalanced classification tasks, the majority class is called the “negative class“ and the minority class is called the “positive class“.

- Probabilities: Predict the probability of class membership for each example.

- Class Labels: Predict a crisp class label for each example.

### Predict Probabilities

If probabilities are intended to be used directly, a good metric might be the Brier Score and the Brier Skill score.

As an alternative, You may want to predict probabilities and allow the user to map them to crisp class labels themselves via a user-selected threshold. 

In this case, a measure can be chosen that summarizes the performance of the model across the range of possible thresholds.

If the positive class is the most important, then the precision-recall curve and area under curve (PR AUC) can be used which will optimize both precision and recall across all thresholds.

If both classes are equally important, the ROC Curve and area under curve (ROC AUC) can be used which will maximize the true positive rate and minimize the false positive rate.

### Predict Class Labels

If class labels are required and both classes are equally important, a good default metric is classification accuracy. 

Accuracy only makes sense if the majority class is less than about 80 percent off the data. A majority class that has a greater than 80 percent or 90 percent skew will swamp the accuracy metric and it will lose its meaning for comparing algorithms.

If the class distribution is severely skewed, the G-mean metric can be used which will optimize the sensitivity and specificity metrics.

If the positive class is more important, variations of the F-Measure can be used that optimize the precision and recall. 

If both false positive and false negatives are equally important, F1 can be used. 

If false negatives are more costly, the F2-Measure can be used. 

If false positives are more costly, the F0.5-Measure can be used.


### Framework for Choosing a Metric

These are just heuristics but provide a useful starting point if you feel lost choosing a metric for your imbalanced classification task.

Figure: How to Choose a Metric for Imbalanced Classification

We can summarize these heuristics into a framework as follows:

Are you predicting probabilities?

- Do you need class labels?
- Is the positive class more important? Use Precision-Recall AUC
- Are both classes important? Use ROC AUC
- Do you need probabilities? Use Brier Score and Brier Skill Score

Are you predicting class labels?

- Is the positive class more important?
  - Are False Negatives and False Positives Equally Costly? Use F1-Measure
  - Are False Negatives More Costly? Use F2-Measure
  - Are False Positives More Costly? Use F0.5-Measure

- Are both classes important?
  - Do you have < 80%-90% examples for the Majority Class? Use Accuracy
  - Do you have > 80%-90% examples for the Majority Class?  Use G-Mean



## Spot Check Algorithms

Spot checking machine learning algorithms means evaluating a suite of different types of algorithms with minimal hyperparameter tuning.

This means giving each algorithm a good chance to learn about the problem, including performing any required data preparation expected by the algorithm and using best-practice configuration options or defaults.

The objective is to quickly test a range of standard machine learning algorithms and provide a baseline in performance to which techniques specialized for imbalanced classification must be compared and outperform in order to be considered skillful. 

The idea is that there is little point in using fancy imbalanced algorithms if they cannot outperform so-called unbalanced algorithms.

A robust test harness must be defined which often involves k-fold cross-validation usually with k-10 as a sensible default. 

Stratified cross-validation is often required to ensure that each fold has the same class distribution as the original dataset. 

The cross-validation procedure is often repeated multiple times such as 3, 10, or 30 in order to effectively capture a sample of model performance on the dataset and summarized with a mean and standard deviation of the scores.

There are perhaps four levels of algorithms to spot check:

1. Naive Algorithms
2. Linear Algorithms
3. Nonlinear Algorithms
4. Ensemble Algorithms

### Naive Algorithms

First. a naive classification must be evaluated which provides a rock-bottom baseline in performance that any algorithm must overcome in order to have skill on the dataset.

Naive means that the algorithm has no logic other than an if-statement or predicting a constant value. 

The choice of naive algorithm is based on the choice of performance metric.

A suggested mapping of performance metrics to naive algorithms is as follows:


- Accuracy: Predict the majority class (class 0).

- G-Mean: Predict a uniformly random class.

- F-Measure: Predict the minority class (class 1).

- ROC AUC: Predict a stratified random class.

- PR ROC: Predict a stratified random class.

- Brier Score: Predict majority class prior.

If you are unsure of the “best” naive algorithm for your metric, perhaps test a few and discover which results in the better performance that you can use as your rock-bottom baseline.

Some options include:

- Predict the majority class in all cases.

- Predict the minority class in all cases.

- Predict a uniform randomly selected class.

- Predict a randomly selected class selected with the prior probabilities of each class.

- Predict the class prior probabilities.


### Linear Algorithms

Linear algorithms are those that are often drawn from the field of statistics and make strong assumptions about the functional form of the problem.

We can refer to them as linear because the output is a linear combination of the inputs or weighted inputs, although this definition is stretched. 

You might also refer to these algorithms as probabilistic algorithms since they are often fit under a probabilistic framework.

They are often fast to train and often perform very well. 

Examples of linear algorithms you should consider trying include:

- Logistic Regression
- Linear Discriminant Analysis
- Naive Bayes

### Nonlinear Algorithms

Nonlinear algorithms are drawn from the field of machine learning and make few assumptions about the functional form of the problem.

We can refer to them as nonlinear because the output is often a nonlinear mapping of inputs to outputs.

They often require more data than linear algorithms and are slower to train. 

Examples of nonlinear algorithms you should consider trying include:

- Decision Tree
- k-Nearest Neighbors
- Artificial Neural Networks
- Support Vector Machine

### Ensemble Algorithms

Ensemble algorithms are also drawn from the field of machine learning and combine the predictions from two or more models.

There are many ensemble algorithms to choose from but when spot-checking algorithms it is a good idea to focus on ensembles of decision tree algorithms since they are known to perform well in practice on a wide range of problems.

Examples of ensembles of decision tree algorithms you should consider trying include:

- Bagged Decision Trees
- Random Forest
- Extra Trees
- Stochastic Gradient Boosting



### Framework for Spot-Checking Machine Learning Algorithms

We can summarize these suggestions into a framework for testing machine learning algorithms on a dataset.

Naive Algorithms

- Majority Class
- Minority Class
- Class Priors

Linear Algorithms

- Logistic Regression
- Linear Discriminant Analysis
- Naive Bayes

Nonlinear Algorithms

- Decision Tree
- k-Nearest Neighbors
- Artificial Neural Networks
- Support Vector Machine

Ensemble Algorithms

- Bagged Decision Trees
- Random Forest
- Extra Trees
- Stochastic Gradient Boosting

The order of the steps is probably not flexible. Think of the order of algorithms as increasing in complexity and capability.


## Spot Check Imbalanced Algorithms

Spot-checking imbalanced algorithms is much like spot-checking machine learning algorithms.

The objective is to quickly test a large number of techniques in order to discover what shows promise so that you can focus more attention on it later during hyperparameter tuning.

The spot-checking performed in the previous section provides both naive and modestly skillful models by which all imbalanced techniques can be compared which allows you to focus on the methods that truly show promise on the problem.

There are perhaps four types of imbalanced classification techniques to spot check:

1. Data Sampling Algorithms
2. Cost-Sensitive Algorithms
3. One-Class Algorithms
4. Probability Tuning Algorithms


### Data Sampling Algorithms

Data sampling algorithms change the composition of the training dataset to improve the performance of a standard machine learning algorithm on an imbalanced classification problem.

There are perhaps three main types of data sampling techniques; they are:

- Data Oversampling
- Data Undersampling
- Combined Oversampling and Undersampling.

Data oversampling involves duplicating examples of the minority class or synthesizing new examples from the minority class from existing examples. 

Perhaps the most popular methods is SMOTE and variations such as Borderline SMOTE. 

Perhaps the most important hyperparameter to tune is the amount of oversampling to perform.

Examples of data oversampling methods include:

- Random Oversampling
- SMOTE
- Borderline SMOTE
- SVM SMote
- k-Means SMOTE
- ADASYN

Undersampling involves deleting examples from the majority class, such as randomly or using an algorithm to carefully choose which examples to delete. Popular editing algorithms include the edited nearest neighbors and Tomek links.

Examples of data undersampling methods include:

- Random Undersampling
- Condensed Nearest Neighbor
- Tomek Links
- Edited Nearest Neighbors
- Neighborhood Cleaning Rule
- One-Sided Selection

Almost any oversampling method can be combined with almost any undersampling technique.

Therefore, it may be beneficial to test a suite of different combinations of oversampling and undersampling techniques.

Data sampling algorithms may perform differently depending on the choice of machine learning algorithm.

Therefore, it may be beneficial to test a suite of standard machine learning algorithms such as all or a subset of those algorithms used when spot checking in the previous section.

Most data sampling algorithms also make use of the k-nearest neighbor algorithm internally which is very sensitive to the data types and scale of input variables. 

Thus, it may be important to at least normalize input variables that have differing scales prior to testing the methods and perhaps use specialized methods if some input variables are categorical instead of numerical.

### Cost-Sensitive Algorithms

Cost-sensitive algorithms are modified versions of machine learning algorithms designed to take the differing costs of misclassification into account when fitting the model on the training dataset.

These algorithms can be effective when used on imbalanced classification where the cost of misclassification is configured to be inversely proportional to the distribution of examples in the training dataset.

There are many cost-sensitive algorithms to choose from, bur it may be practical to test a range of cost-sensitive versions of linear, nonlinear, and ensemble algorithms.

Some examples of machine learning algorithms that can be configured using cost-sensitive training include:

- Logistic Regression
- Decision Trees
- Support Vector Machines
- Artificial Neural Networks
- Bagged Decision Trees
- Random Forest
- Stochastic Gradient Boosting


### One-Class Algorithms

Algorithms used for outlier detection and anomaly detection can be used for classification tasks.

Although unusual when used in this way, they are often referred to as one-class classification algorithms.

In some cases, one-class classification algorithms can be very effective such as when there is a severe class imbalance with very few examples of the positive class.

Examples of one-class classification algorithms to try include:

- One-Class Support Vector Machines
- Isolation Forests
- Minimum Covariance Determinant
- Local Outlier Factor


### Probability Tuning Algorithms

Predicted probabilities can be improved in two ways:

- Calibrating Probabilities
- Tuning the Classification Threshold

#### Calibrating Probabilities

Some algorithms are fit using a probabilistic framework so they have calibrated probabilities.

This means that when 100 examples are predicted to have the positive class label with a probability of 80 percent, then the algorithm will predict the correct class label 80 percent of the time.

Calibrated probabilities are required from a model to be considered skillful on a binary classification task when probabilities are either required as the output or used to evaluate the model (ROC AUC or PR AUC).

Most nonlinear algorithms do not predict calibrated probabilities, therefore algorithms can be used to post-process the predicted probabilities in order to calibrate them.

Therefore, when probabilities are required directly or are used to evaluate a model, and nonlinear algorithms are being used, it is important to calibrate the predicted probabilities.

#### Tuning the Classification Threshold

Some algorithms are designed to naively predict probabilities that later must be mapped to crisp class labels.

This is the case if class labels are required as output for the problem or the model is evaluated using class labels.

Probabilities are mapped to class labels using a _threshold_ probability value. 

All probabilities below the threshold are mapped to class 0 and all probabilities equal-to or above the threshold are mapped to class 1.

The default threshold is 0.5 but different thresholds can be used that will dramatically impact the class labels which impacts the performance of a machine learning model that natively predicts probabilities.

Thus, if probabilistic algorithms are used that natively predict a probability and class labels are required as output or used to evaluate models, it is a good idea to try tuning the classification threshold.


### Framework for Spot-Checking Imbalanced Algorithms

We can summarize these suggestions into a framework for testing imbalanced machine learning algorithms on a dataset.


## References

[Multi-Class Classification Tutorial with the Keras Deep Learning Library](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)

[Imbalanced Multiclass Classification with the Glass Identification Dataset](https://machinelearningmastery.com/imbalanced-multiclass-classification-with-the-glass-identification-dataset/)


