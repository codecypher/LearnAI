# Applied Machine Learning Checklist

## Overview

You can use a checklist to structure your search for the right combination of elements to consistently deliver a good solution to any machine learning (ML) problem.

A checklist is a simple tool that guarantees an outcome, not a panacea or silver bullet. 


## Each Data Problem is Different

You have no idea what algorithm will work best on a problem before you start. Even expert data scientists cannot tell you.

This problem is not limited to the selection of ML algorithms. 

You cannot know what data transforms and what features in the data that if exposed would best present the structure of the problem to the algorithms.

You may have some ideas and you may have some favorite techniques, but how do you know that the techniques that got you results last time will get you good results this time?

How do you know that the techniques are transferable from one problem to another?

_Heuristics_ are a good starting point (such as random forest does well on most problems) but they are just a starting point, not the end.


## The Checklist

### 5-Step Systematic Process

1. Define the Problem
2. Prepare Data
3. Spot Check Algorithms
4. Improve Results
5. Finalize Project

### 1. Define The Problem

It is important to have a well developed understanding of the problem before touching any data or algorithms [1].

#### 1.1 What is the problem

This section is intended to capture a clear statement of the problem plus any expectations that may have been set and biases that may exist.

- Define the problem informally and formally.
- List the assumptions about the problem or about the data. 
- List known problems similar to your problem.

#### 1.2 Why does the problem need to be solved

Here we try to capture the motivation for solving the problem and force ourselves to think about the expected outcome.

- Describe the motivation for solving the problem.
- Describe the benefits of the solution (model or the predictions).
- Describe how the solution will be used.

#### 1.3 How could the problem be solved manually

Here we try to discover any remaining domain knowledge and to determine if a machine learning solution is really required.

- Describe how the problem is currently solved (if at all).
- Describe how a subject matter expert would make manual predictions.
- Describe how a programmer might hand code a classifier.


### 2. Prepare The Data

You should spend most of your time trying to understand your data [2]: [3]. 

#### 2.1 Data Description

Here we try to think about all of the data that is and is not available.

- Describe the extent of the data that is available.
- Describe data that is not available but is desirable.
- Describe the data that is available that you do not need.

#### 2.2 Data Preprocessing

Here we try to to organize the raw data into a form that we can work with in modeling tools.

- Format data so that it is in a form that you can work with.
- Clean the data so that it is uniform and consistent.
- Sample the data in order to best trade-off redundancy and problem fidelity.

**Shortlist of Data Sampling**

**Sample instances:** Create a sample of the data that is both representative of the various attribute densities and small enough that you can build and evaluate models quickly. 

**Sample attributes:** Select attributes that best expose the structures in the data to the models. Different models have different requirements,

Below are some ideas for different approaches that you can use to sample your data.

You should use each one in turn and let the results from your test harness tell you which representation to use.

- Random or stratified samples
- Rebalance instances by class (more on rebalancing methods)
- Remove outliers (more on outlier methods)
- Remove highly correlated attributes
- Apply dimensionality reduction methods (principle components or t-SNE)

#### 2.3 Data Transformation

This section is intended to create multiple views on the data in order to expose more of the problem structure in the data to modeling algorithms in later steps.

- Create linear and non-linear transformations of all attributes.
- Decompose complex attributes into their constituent parts.
- Aggregate denormalized attributes into higher-order quantities.

**Shortlist of Data Transformations**

There are a limited number of data transforms that you can use.

Here is a list of some univariate (single attribute) data transforms:

- Square and Cube
- Square root
- Standardize (0 mean and unit variance)
- Normalize (rescale to 0-1)
- Descritize (convert a real to categorical)

Again you should try each of them in turn and let the results from your test harness show the best transformations for your problem.

#### 2.4 Data Summarization

Here we try to find any obvious relationships in the data.

- Create univariate plots of each attribute.
- Create bivariate plots of each attribute with every other attribute.
- Create bivariate plots of each attribute with the class variable. 


### 3. Spot Check Algorithms

Now we can start building and evaluating models [4]][5].

#### 3.1 Create Test Harness

Here we define a method for model evaluation that we can use to compare results.

- Create a hold-out validation dataset for use later.

- Evaluate and select an appropriate test option.

- Select one (or a small set) performance measure used to evaluate models.

#### 3.2 Evaluate Candidate Algorithms

Here we try to quickly find how learnable the problem might be and what algorithms and views of the data may be good for further investigation.

- Select a diverse set of algorithms to evaluate (10-20).

- Use common or standard algorithm parameter configurations.

- Evaluate each algorithm on each prepared view of the data.

**Shortlist of Algorithms To Try on Classification Problems**

Here the list does not matter as much as the strategy of spot checking and not going with your favorite algorithm.

Try to evaluate a good mix of algorithms that model the problem quite differently [6]. 

- Instance-based like k-Nearest Neighbors and Learning Vector Quantization

- Simpler methods such as Naive Bayes, Logistic Regression and Linear Discriminant Analysis

- Decision Trees such as CART and C4.5/C5.0

- Complex non-linear approaches like Backpropagation and Support Vector Machines

- Always try random forest and gradient boosted machines


### 4. Improve Results

Here we will have a smaller pool of models that are known to be effective on the problem, so it is time to improve the results [7].

#### 4.1 Algorithm Tuning

Here we try to get the most from well-performing models.

- Use historically effective model parameters.
- Search the space of model parameters.
- Optimize well performing model parameters.

#### 4.2 Ensemble Methods

Here we try to combine the results from well-performing models obtain a further improvement in accuracy.

- Use Bagging on well performing models.
- Use Boosting on well performing models.
- Blend the results of well performing models.

#### 4.3 Model Selection

Here we try to verify that the process of model selection is well-rounded.

- Select a diverse subset (5-10) of well performing models or model configurations.
- Evaluate well performing models on a hold out validation dataset.
- Select a small pool (1-3) of well performing models.


### 5. Finalize Project

Now that we have some results, we can review the problem definition to make good use of the results [8]]. This includes training the chosen model on the entire dataset.

#### 5.1 Present Results

Here we try to ensure that we capture what we have done and learned so that we can make the best use of it later. 

- Write up project in a short report (1-5 pages).
- Convert write-up to a slide deck to share findings with others.
- Share code and results with interested parties.

#### 5.2 Operationalize Results

Here we try to make sure that we deliver on the solution promise that we made up front.

- Adapt the discovered procedure from raw data to result to an operational setting.
- Deliver and make use of the predictions.
- Deliver and make use of the predictive model.


### Next Step

Pick a problem that you can complete in 1-to-2 hours and use the checklist to complete a project.


## References

[1]: [How to Define Your Machine Learning Problem](https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/)

[2]: [How to Prepare Data For Machine Learnin](https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/)

[3]: [Quick and Dirty Data Analysis for your Machine Learning Problem](https://machinelearningmastery.com/quick-and-dirty-data-analysis-for-your-machine-learning-problem/)

[4]: [How to Evaluate Machine Learning Algorithms](https://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/)

[5]: [Why you should be Spot-Checking Algorithms on your Machine Learning Problems](https://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/)

[6]: [Tour of Machine Learning Algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

[7]: [How to Improve Machine Learning Results](https://machinelearningmastery.com/how-to-improve-machine-learning-results/)

[8]: [How to Use Machine Learning Results](https://machinelearningmastery.com/how-to-use-machine-learning-results/)

[9]: [Save And Finalize Your Machine Learning Model in R](https://machinelearningmastery.com/finalize-machine-learning-models-in-r/)


[^ml_checklist_howto]: <https://machinelearningmastery.com/machine-learning-checklist/> "How to Use a Machine Learning Checklist to Get Accurate Predictions, Reliably"
