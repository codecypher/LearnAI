# Applied Machine Learning Process

The benefits of machine learning are the predictions and the models that make predictions.

To have skill at applied machine learning means knowing how to consistently and reliably deliver high-quality predictions on problem after problem, so you need to follow a systematic process.

Here is a 5-step process that you can follow to consistently achieve above average results on predictive modeling problems:

## 5-Step Systematic Process

1. Define the Problem
2. Prepare Data
3. Spot Check Algorithms
4. Improve Results
5. Present Results


## **Step 1:** Define the problem

Describe the problem informally and formally and list assumptions and similar problems.

[How to Define Your Machine Learning Problem](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)

The field of _machine learning_ is concerned with the question of how to construct computer programs that automatically improve with experience.

### Step 1-1: What is the Problem

Tom Mitchell’s machine learning formalism:

A computer program is said to _learn_ from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.

Use this formalism to define the T, P, and E for your problem:

- Task (T): Classify a tweet that has not been published as going to get retweets or not.

- Experience (E): A corpus of tweets for an account where some have retweets and some do not.

- Performance (P): Classification accuracy, the number of tweets predicted correctly out of all tweets considered as a percentage.


#### Assumptions

Create a list of assumptions about the problem and its phrasing. 

These may be rules of thumb and domain specific information that you think will get you to a viable solution faster.

It can also be useful to highlight areas of the problem specification that may need to be challenged, relaxed 
, or tightened.

#### Similar problems

What other problems have you seen or can you think of that are similar to the problem you are trying to solve? 

Other problems can inform the problem you are trying to solve by highlighting limitations in your phrasing of the problem such as time dimensions and conceptual drift (where the concept being modeled changes over time). 

Other problems can also point to algorithms and data transformations that could be adopted to spot check performance.


### Step 1-2: Why does the the problem need to be solved?

Think deeply about why you want or need the problem solved.

### Step 1-3: How would I solve the problem?

Describe how the problem would be solved manually to flush domain knowledge.

List out step-by-step what data you would collect, how you would prepare it and how you would design a program to solve the problem. 

This may include prototypes and experiments you would need to perform which are a gold mine because they will highlight questions and uncertainties you have about the domain that could be explored.


## **Step 2:** Prepare your data

### How to Prepare Data For Machine Learning

Machine learning algorithms learn from data, so it is critical that you feed them the right data for the problem you want to solve.

Even if you have good data, you need to make sure that it is in a useful scale, format, and that meaningful features are included.

#### Data Preparation Process

The more disciplined you are in your handling of data, the more consistent and better results you are like likely to achieve. 

The process for getting data ready for a machine learning algorithm can be summarized in three steps:

- Step 2-1: Data Selection: Consider what data is available, what data is missing, and what data can be removed.

- Step 2-2: Data Preprocessing: Organize your selected data by formatting, cleaning, and sampling from it.

- Step 2-3: Data Transformation: Transform preprocessed data ready for machine learning by engineering features using scaling, attribute decomposition, and attribute aggregation.


### How to Identify Outliers in your Data

Many machine learning algorithms are sensitive to the range and distribution of attribute values in the input data.

Outliers in input data can skew and mislead the training process of machine learning algorithms resulting in longer training times, less accurate models, and ultimately poorer results.

#### Outlier Modeling

Outliers are extreme values that fall a long way outside of the other observations.

In his book “Outlier Analysis”, Aggarwal provides a useful taxonomy of outlier detection methods:

- Extreme Value Analysis: Determine the statistical tails of the underlying distribution of the data. For example, statistical methods like the z-scores on univariate data.

- Probabilistic and Statistical Models: Determine unlikely instances from a probabilistic model of the data. For example, gaussian mixture models optimized using expectation-maximization.

- Linear Models: Projection methods that model the data into lower dimensions using linear correlations. For example, principle component analysis and data with large residual errors may be outliers.

- Proximity-based Models: Data instances that are isolated from the mass of the data as determined by cluster, density or nearest neighbor analysis.

- Information Theoretic Models: Outliers are detected as data instances that increase the complexity (minimum code length) of the dataset.

- High-Dimensional Outlier Detection: Methods that search subspaces for outliers give the breakdown of distance based measures in higher dimensions (curse of dimensionality).

NOTE: the interpretability of an outlier model is critically important. Context or rationale is required around decisions why a specific data instance is or is not an outlier.

#### Get Started

There are many methods and much research put into outlier detection. 

Start by making some assumptions and design experiments where you can clearly observe the effects of the those assumptions against some performance or accuracy measure.

Work through a stepped process from extreme value analysis, proximity methods, and projection methods.

#### Methods Robust to Outliers

An alternative strategy is to move to models that are robust to outliers. 

There are robust forms of regression that minimize the median least square errors rather than mean (so-called robust regression) but are more computationally intensive. 

There are also methods like decision trees that are robust to outliers.

You could spot check some methods that are robust to outliers. If there are significant model accuracy benefits then there may be an opportunity to model and filter out outliers from your training data.


- Improve Model Accuracy with Data Pre-Processing
- Discover Feature Engineering
- An Introduction to Feature Selection
- Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset
- Data Leakage in Machine Learning


## Step 3: Spot-check algorithms.

- How to Evaluate Machine Learning Algorithms
- Why you should be Spot-Checking Algorithms on your Machine Learning Problems
- How To Choose The Right Test Options When Evaluating Machine Learning Algorithms
- A Data-Driven Approach to Choosing Machine Learning Algorithms


## Step 4: Improve results.

- How to Improve Machine Learning Results
- Machine Learning Performance Improvement Cheat Sheet
- How To Improve Deep Learning Performance


## Step 5: Present results.

- How to Use Machine Learning Results
- How to Train a Final Machine Learning Model
- How To Deploy Your Predictive Model To Production



## References

[Applied Machine Learning Process (Outline)](https://machinelearningmastery.com/start-here/#process)

[Applied Machine Learning Process](https://machinelearningmastery.com/process-for-working-through-machine-learning-problems/)

[How to Use a Machine Learning Checklist](https://machinelearningmastery.com/machine-learning-checklist/)



