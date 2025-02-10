# Machine Learning Algorithms

## Background

Machine learning is about machine learning algorithms [1].

You need to know what algorithms are available for a given problem, how they work, and how to get the most out of them.

Here is how to get started with machine learning algorithms:

**Step 1:** Discover the different types of machine learning algorithms.

**Step 2:** Discover the foundations of machine learning algorithms.

- How Machine Learning Algorithms Work
- Parametric and Nonparametric Algorithms
- Supervised and Unsupervised Algorithms
- The Bias-Variance Trade-Off
- Overfitting and Underfitting With Algorithms

**Step 3:** Discover how the top machine learning algorithms work.

- Machine Learning Algorithms Mini-Course
- Master Machine Learning Algorithms


----------


## Categories of ML Algorithms

There are two common approaches to categorizing ML algorithms [2]:

- Group algorithms by learning style
- Group algorithms by similarity in form or function


![Taxonomy of 63 Machine Learning Algorithms|600xauto {Figure: 63 Machine Learning Algorithms by Priyanshu Jain}](https://miro.medium.com/max/4056/1*A9QcGlMRQBAl6cfhxS-XeA.png)


## Overview of Machine Learning Algorithms

When crunching data to model business decisions, we will usually be using supervised and unsupervised learning methods [1] [2].

A hot topic at the moment is semi-supervised learning methods in areas such as image classification where there are large datasets with very few labeled examples.


## Machine Learning Techniques

Data set, algorithms, models, feature extraction, and training are 5 crucial components of Machine Learning [6]. 

Depending upon the scenario, data type, input, etc., there are different ways an ML model learns: supervised and unsupervised.

In general, there are 10 Machine Learning techniques that are commonly used for AI development [6]:

1. Regression (supervised)

2. Classification (supervised)

3. Transfer Learning (supervised)

4. Clustering (unsupervised)

5. Ensemble Methods (supervised)

6. Neural Networks and Deep Learning (unsupervised)

7. Dimensionality Reduction (supervised and unsupervised)

8. Word Embeddings (unsupervised)

9. Natural Language Processing (unsupervised)

10. Reinforcement Learning (unsupervised)



## Algorithms Grouped by Learning Style

There are four ways an algorithm can model a problem based on its interaction with the experience, environment, or whatever we want to call the input data [2].

It is popular in machine learning and AI textbooks to consider the learning styles that an algorithm can adopt.

This _taxonomy_ or way of organizing machine learning algorithms is useful because it forces you to think about the roles of the input data and model preparation process so that you can select one that is the most appropriate for your problem.


### Supervised Learning

In supervised learning, the input data is called training data and has a known label or result. 

The model is prepared through a training process in which it is required to make predictions and is corrected when those predictions are wrong. 

The training process continues until the model achieves a desired level of accuracy on the training data.

Examples problems: classification and regression.

Example algorithms: Logistic Regression and Back-Propagation Neural Network.

### Unsupervised Learning

In unsupervised learning, the input data is not labeled, so the data does not have a known result.

The model is prepared by deducing structures present in the input data which may be to extract general rules through a mathematical process to systematically reduce redundancy or to organize data by similarity.

Example problems: clustering, dimensionality reduction, and association rule learning.

Example algorithms: The Apriori algorithm, K-Means, and PCA. 

### Semi-Supervised Learning

In semi-supervised learning, the input data is a mixture of labeled and unlabelled examples.

There is a desired prediction problem but the model must learn the structures to organize the data as well as make predictions.

Example problems: classification and regression.

Example algorithms are extensions to other flexible methods that make assumptions about how to model the unlabeled data.

### Reinforcement Learning

In reinforcement learning (RL), an agent (a model) gets feedback as it goes about its task, and learns from that feedback. 

RL can be applied to situations where the agent has a goal and tries different actions within its environment until it reaches the goal.

There are two main types of reinforcement learning: policy and value function-based.

Policy-based RL is where an algorithm determines which action to take in a given situation by estimating which action is most likely to bring about a good result based on the agent’s current policy which is a state-action mapping that can be updated with each action taken.

Value function RL adjusts the agent’s behaviour according to how valuable   certain actions are within its environment by estimating expected future rewards when certain actions are performed in a given state.

With reinforcement learning, you do not have any labelled data for training purposes. You provide direct feedback from the environment to let the agent know what works and what does not. 

Once agents have been taught using reinforcement learning, they can  apply this knowledge elsewhere in their environment.



## The Elbow Method

The _elbow method_ is a common heuristic (not a formula) for finding the optimal k value for unsupervised models such as PCA or k-means [9]. 

The best approach to using the elbow method is to create the scree plot and compute a metric for comparison. 

Also, AutoML tools such as Orange are good for quickly visualizing the results for the elbow method. 


Here are some tutorials that may help:

[PCA Example in Python with scikit-learn](https://cmdlinetips.com/2018/03/pca-example-in-python-with-scikit-learn/amp/)

[How to do a Clustering project Step by Step](https://z-ai.medium.com/how-to-do-a-clustering-project-step-by-step-4b41e94fad1)

[Detailed Introduction To K-Means Clustering In Python](https://onepagecodewriter.medium.com/detailed-introduction-to-k-means-clustering-in-python-d8178d7eb123)



----------



## Algorithms Grouped By Similarity

Algorithms are often grouped by similarity in terms of their function (how they work). For example, tree-based methods and neural network inspired methods [2].

This is perhaps the most useful way to group algorithms and it is the approach we will use in [5].

This is a useful grouping method but it is not perfect. However, there are algorithms that could just as easily fit into multiple categories.


### Classification Algorithms

[Guide to Classification and Regression](./ml/regression.md)

Classification is when the output Y is categorical such as positive vs negative (sentiment analysis), dog vs cat (image classification), and disease vs no disease (medical diagnosis).

Linear: Logistic Regression, Linear Discriminant Analysis (LDA)

Non-linear: K-nearest neighbor, Support vector machines (SVM), Naive Bayes, Decision Tree

Not all classification predictive models support multi-class classification.

Algorithms such as the Perceptron, Logistic Regression, and Support Vector Machines were designed for binary classification and do not natively support classification tasks with more than two classes.

One approach for using binary classification algorithms for multi-classification problems is to split the multi-class classification dataset into multiple binary classification datasets and fit a binary classification model on each. 

Two different examples of this approach are: One-vs-Rest and One-vs-One.


### Regression Algorithms

**Regression** is concerned with modeling the relationship between variables that is iteratively refined using a measure of error in the predictions made by the model.

Regression methods are a workhorse of statistics and have been applied to statistical machine learning. 

Linear: Linear regression, Ridge, LASSO, Elastic-net

Non-linear: K-nearest neighbor, Support vector machines (SVM), Decision Tree

The most common regression algorithms are:

- Ordinary Least Squares Regression (OLSR)
- Linear Regression
- Logistic Regression
- Stepwise Regression
- Multivariate Adaptive Regression Splines (MARS)
- Locally Estimated Scatterplot Smoothing (LOESS)

The primary assumptions of linear regression are:

1. **Linearity:** There is a linear relationship between the outcome and predictor variable(s).

2. **Normality:** The residuals or errors follow a normal distribution. The error is calculated by subtracting the predicted value from the actual value.

3. **Homoscedasticity:** The variability in the dependent variable is equal for all values of the independent variable(s).

### Instance-based Algorithms

Instance-based learning model is a decision problem with instances or examples of training data that are deemed important or required to the model.

Such methods typically build up a database of example data and compare new data to the database using a similarity measure in order to find the best match and make a prediction. 

For this reason, instance-based methods are also called winner-take-all methods and memory-based learning. 

The focus is on the representation of the stored instances and similarity measures used between instances.

The most popular instance-based algorithms are:

- k-Nearest Neighbor (kNN)
- Learning Vector Quantization (LVQ)
- Self-Organizing Map (SOM)
- Locally Weighted Learning (LWL)
- Support Vector Machines (SVM)

### Decision Tree Algorithms

Decision tree methods construct a model of decisions made based on actual values of attributes in the data.

Decisions fork in tree structures until a prediction decision is made for a given record. 

Decision trees are trained on data for classification and regression problems. 

Decision trees are often fast and accurate and a big favorite in machine learning.

The most popular decision tree algorithms are:

- Classification and Regression Tree (CART)
- Iterative Dichotomiser 3 (ID3)
- C4.5 and C5.0 (different versions of a powerful approach)
- Chi-squared Automatic Interaction Detection (CHAID)
- Decision Stump
- M5
- Conditional Decision Trees


### Bayesian Algorithms

Bayesian methods are those that explicitly apply Bayes’ Theorem for problems such as classification and regression.

The most popular Bayesian algorithms are:

- Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Averaged One-Dependence Estimators (AODE)
- Bayesian Belief Network (BBN)
- Bayesian Network (BN)


#### Problems with Bayesian Networks

The main problem with Bayesian Networks (BN) is that they appear to be deceptively simple. However, in reality there are many issues with their practical use:

- Most of the software tools currently available for BN are GUI which makes them impractical for most real-world applications.

- There are few software libaries available for BN and those that are available are usually poorly documented. I completed an assignment using pyagrum which was a recommended library for BN and I encountered many problems with the API and documentation.

- My professor gave us what appeared to be a simple BN problem. However, none of the students in the class (including me) were able to get the problem correct even though we were given more than one chance (each time with more explanation on BN). However, I did get it correct the third time. Thus, I am not a fan of BN.


### Clustering Algorithms

Clustering (like regression) describes the class of problem and the class of methods.

Clustering methods are typically organized by the modeling approaches such as centroid-based and hierarchal. 

All clustering methods are concerned with using the inherent structures in the data to best organize the data into groups of maximum commonality.

The most popular clustering algorithms are:

- k-Means
- k-Medians
- Expectation Maximization (EM)
- Hierarchical Clustering


### Association Rule Learning Algorithms

Association rule learning methods extract rules that best explain observed relationships between variables in data.

These rules can discover important and commercially useful associations in large multidimensional datasets that can be exploited by an organization.

The most popular association rule learning algorithms are:

- Apriori algorithm
- Eclat algorithm


### Artificial Neural Network Algorithms

Artificial Neural Networks are models that are inspired by the structure and/or function of biological neural networks.

They are a class of _pattern matching_ that are commonly used for regression and classification problems but are really an enormous subfield comprised of hundreds of algorithms and variations for all manner of problem types.

The most popular artificial neural network algorithms are:

- Perceptron
- Multilayer Perceptrons (MLP)
- Back-Propagation
- Stochastic Gradient Descent
- Hopfield Network
- Radial Basis Function Network (RBFN)

### Deep Learning Algorithms

Deep Learning methods are a modern update to Artificial Neural Networks that exploit abundant cheap computation.

They are concerned with building much larger and more complex neural networks and many methods are concerned with very large datasets of labelled analog data such as image, text. audio, and video.

The most popular deep learning algorithms are:

- Convolutional Neural Network (CNN)
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory Networks (LSTMs)
- Stacked Auto-Encoders
- Deep Boltzmann Machine (DBM)
- Deep Belief Networks (DBN)


### Dimensionality Reduction Algorithms

Like clustering methods, dimensionality reduction seek and exploit the inherent structure in the data, but in an unsupervised manner or order to summarize or describe data using less information.

This can be useful to visualize dimensional data or to simplify data which can then be used in a supervised learning method. 

Many of these methods can be adapted for use in classification and regression.

- Principal Component Analysis (PCA)
- Principal Component Regression (PCR)
- Partial Least Squares Regression (PLSR)
- Sammon Mapping
- Multidimensional Scaling (MDS)
- Projection Pursuit
- Linear Discriminant Analysis (LDA)
- Mixture Discriminant Analysis (MDA)
- Quadratic Discriminant Analysis (QDA)
- Flexible Discriminant Analysis (FDA)


### Principal Component Analysis (PCA)

PCA is not really considered an ML model rather it is a technique (linear transform) used to reduce the dimensionality of the data (data preparation technique) in preparation for use with an ML model.

### Ensemble Algorithms

Ensemble methods are models composed of multiple weaker models that are independently trained and whose predictions are combined in some way to make the overall prediction.

Much effort is put into what types of weak learners to combine and the ways in which to combine them. 

Ensemble is a very powerful class of techniques that is very popular.

- Boosting
- Bootstrapped Aggregation (Bagging)
- AdaBoost
- Weighted Average (Blending)
- Stacked Generalization (Stacking)
- Gradient Boosting Machines (GBM)
- Gradient Boosted Regression Trees (GBRT)
- Random Forest


Other Lists of Machine Learning Algorithms
How to Study Machine Learning Algorithms
How to Run Machine Learning Algorithms


### Decision Trees

Decision Trees are an important type of algorithm for predictive modeling machine learning.

The classical decision tree algorithms have been around for decades and modern variations like random forest are among the most powerful techniques available.


Gradient boosting involves the creation and addition of decision trees sequentially, each attempting to correct the mistakes of the learners that came before it.

This raises the question as to how many trees (weak learners or estimators) to configure in your gradient boosting model and how big each tree should be. 


[How to Tune the Number and Size of Decision Trees with XGBoost in Python](https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/)

[Decision Trees vs Random Forest](https://medium.com/mlearning-ai/decision-trees-vs-random-forest-c0fcabb757f2)



## Regression Algorithms Pros and Cons

There are many regression algorithms you should know and try when working on a real-world problem. 

The article discusses some of the Pros and Cons of 9 common regression algorithms with code using Scikit-learn and XGBoost:

1. Linear Regression
2. Polynomial Regression
3. Simple Vector Regression (SVR)
4. Decision Tree Regression
5. Random Forest Regression
6. LASSO Regression
7. Ridge Regression
8. ElasticNet Regression
9. XGBoost Regression


----------


## Examples of Algorithm Lists To Create

Here are 10 examples of machine learning algorithm lists that we could create [8].

- Regression algorithms
- SVM algorithms
- Data projection algorithms
- Deep learning algorithms
- Time series forecasting algorithms
- Rating system algorithms
- Recommender system algorithms
- Feature selection algorithms
- Class imbalance algorithms
- Decision tree algorithms


## Tips for Creating Algorithm List

- Start with why we want the list and use that to define the type of list to create.

- Only capture the algorithm properties we actually need, keep it as simple as possible.




### Multimodal Algorithms

Multimodal learning is a subfield of artificial intelligence that seeks to effectively process and analyze data from multiple modalities [10]. 

- Fusion-Based Approach
- Alignment-Based Approach
- Late Fusion

The main idea is to combine information from different sources such as text, image, audio, and video to build a more complete and accurate understanding of the underlying data.

### Combining Models

Combining models is a technique in machine learning that involves using multiple models to improve the performance of a single model.

- Ensemble
- Stacking
- Bagging

The main idea is that one model's strengths can compensate for another's weakness, resulting in a more accurate and robust prediction. 

### Differences

In combining models, the models are trained independently and the final prediction is made by combining the outputs of these models using a combined model technique.

Combining models is useful when the models have complementary strengths and weaknesses such that the combination provides a more accurate prediction.

In multimodal learning, the goal is to combine information from different sources to make a prediction using one of the multimodal techniques  to create a high-dimensional representation that captures the semantic information of the data from each modality.

Multimodal learning is useful when different modalities provide complementary information that can improve the accuracy of the prediction.

The main difference between combining models and multimodal learning is that combining models involves using multiple models to improve the performance of a single model while multimodal learning involves learning from and combining information from multiple modalities such as image, text, and audio to perform a prediction.



----------



## Orange Algorithm List

### Regression

- Linear Regression

### Classification

- Logistic Regression
- Naive Bayes
- Correspondence Analysis (multivariate)

### Classification and Regression

- AdaBoost
- Gradient Boosting (xgboost)
- Random Forest
- kNN
- SGD
- SVM
- Tree

### Dimensionality Reduction

- PCA
- MDS
- t-SNE
- Manifold Learning

### Clustering

- k-Means
- DBSCAN
- Louvain Clustering
- Hierarchical Clustering


## References

[1]: [Understand Machine Learning Algorithms](https://machinelearningmastery.com/start-here/#algorithms)

[2]: [A Tour of Machine Learning Algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

[3]: [Clustering with K-Means: simple yet powerful](https://medium.com/@alexandre.hsd/everything-you-need-to-know-about-clustering-with-k-means-722f743ef1c4)

[4]: [A Practical Introduction to 9 Regression Algorithms](https://towardsdatascience.com/a-practical-introduction-to-9-regression-algorithms-389057f86eb9?source=rss----7f60cf5620c9---4)

[5]: [You Should Check Out This Effective Framework for Model Selection](https://pub.towardsai.net/a-framework-for-model-selection-ea4dcda2cb3a)

[6]: [10 Machine Learning Techniques for AI Development](https://daffodilsw.medium.com/10-machine-learning-techniques-for-ai-development-15df0953f05f)

[7]: [63 Machine Learning Algorithms](https://medium.com/swlh/63-machine-learning-algorithms-introduction-5e8ea4129644)

[8]: [Take Control By Creating Targeted Lists of Machine Learning Algorithms](https://machinelearningmastery.com/create-lists-of-machine-learning-algorithms/)

[9]: [Elbow Method for optimal value of k in KMeans](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)

[10]: [Multimodal Models Explained](https://www.kdnuggets.com/2023/03/multimodal-models-explained.html)


[ML Cheatsheet - Algorithms](https://mclguide.readthedocs.io/en/latest/sklearn/guide.html#algorithms)


[Comprehensive Guide to Decision Tree Learning for Classification](https://therised.medium.com/comprehensive-guide-to-decision-tree-learning-for-classification-61bf2c510c6b)

[Stop using the Elbow Method](https://medium.com/geekculture/stop-using-the-elbow-method-96bcfbbbe9fd)

[Stop Using One-vs-One or One-vs-Rest for Multi-Class Classification Tasks](https://towardsdatascience.com/stop-using-one-vs-one-or-one-vs-rest-for-multi-class-classification-tasks-31b3fd92cb5e)
