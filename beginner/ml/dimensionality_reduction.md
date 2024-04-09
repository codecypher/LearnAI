# Dimensionality Reduction

## What is Dimensionality Reduction

**Dimensionality reduction** is the process of reducing the dimension of your feature set.

Our feature set could be a dataset with a hundred columns (features) or it could be an array of points that make up a large sphere in the three-dimensional space.

Dimensionality reduction is the process of bringing the number of columns down to twenty or converting the sphere to a circle in thek two-dimensional space.

Why would we drop 80 columns off our dataset when we could straight up feed it to our machine learning algorithm and let it do the rest?


## High-dimensional data

In machine learning, the number (degree) of features in a dataset is referred to as its _dimensionality_.

Machine learning problems with high dimensionality have a variety of issues [8].

- As the number of features in a dataset increases, the time needed to train the model will also increase.

- High-dimensional datasets are extremely difficult (if not impossible) to visualize. The harder it is to visualize a dataset, the harder it can be to explain which contributes to the “black box” problem in machine learning.

- The Curse of Dimensionality: the amount of data needed to train a model grows exponentially with the number of features (dimensions) in a dataset.


## The Curse of Dimensionality

The _curse of dimensionality_ refers to the problems that arise when working with data in the higher dimensions which does not exist in the lower dimensions [3].

- As the number of features (dimensions) increase, the number of samples also increases proportionally. Thus, the more features we have, the more samples we will need to have all combinations of feature values well-represented.

- As the number of features increases, the model becomes more complex and the greater the chances of overfitting.

- A machine learning model that is trained on a large number of features becomes increasingly dependent on the data it was trained on and in turn overfitted which results in poor performance on new unseen data.


Avoiding overfitting is a major motivation for performing dimensionality reduction: the fewer features our training data has, the fewer assumptions our model makes and the simpler it will be.

In addition, dimensionality reduction has the following advantages [3]:

  1. Less misleading data means model accuracy improves.
  2. Less dimensions mean less computing.
  3. Less data means algorithms train faster.
  3. Less data means less storage space required.
  4. Less dimensions allow usage of algorithms unfit for a large number of dimensions
  5. Removes redundant features and noise.


The curse of dimensionality describes the phenomenon where the feature space becomes increasingly _sparse_ for an increasing number of dimensions of a fixed-size training dataset [1] [2].

We can think of even the closest neighbors being too far away in a high-dimensional space to give a good estimate.

> KNN is very susceptible to overfitting due to the curse of dimensionality.

We have discussed the concept of regularization in the section about logistic regression as one way to avoid overfitting. However, in models where regularization is not applicable such as decision trees and KNN, we can use feature selection and dimensionality reduction techniques to help us avoid the curse of dimensionality.


---------



## PCA Requirements

Compared to similar statistical analyses, Principal Component Analysis (PCA) has only a few requirements that must be met in order to obtain meaningful results.

The basic properties that the data set should have are:

- The correlation between the features should be linear.

- The data set should be free of outliers which means individual data points that deviate strongly from the mass.

- If possible, the variables should be continuous.

- The result of the PCA becomes better, the larger the sample is.

Not all data sets can be used for Principal Component Analysis. It must be ensured that the data are approximately normally distributed and interval-scaled such as an interval between two numerical values always has the same spacing.

For example, dates are interval scaled since from 01.01.1980 to 01.01.1981 the time interval is the same as from 01.01.2020 to 01.01.2021 (leap years excluded). In fact, interval scaling must be judged by the user himself and cannot be detected by standardized statistical tests.



---------



# Techniques for Dimensionality Reduction

## What is Data Mining

Data mining is the process of discovering trends and insights in high-dimensionality datasets (containing thousands of columns).

High-dimensionality datasets have enabled organizations to solve complex, real-world problems.

However, large datasets can contain columns with poor-quality data which can lower the performance of the model (more is not always better).

One way to preserve the structure of high-dimensional data in a low-dimensional space is to use a _dimensional reduction (DR)_ technique.

## What is Dimensional Reduction

**Dimensionality reduction (DR)** is the process of reducing the number of attributes in a dataset while keeping as much of the variation in the original dataset as possible.

Dimensionality reduction is a data preprocessing step which means that it is done before training the model.

## Benefits of Dimensional Reduction

1. DR improves the model accuracy due to less misleading data
2. The model trains faster since it has fewer dimensions
3. DR makes the model simpler for researchers to interpret.

There are three main dimensional reduction techniques:

1. Feature elimination and extraction
2. Linear algebra
3. Manifold.

We will look at a strategy for implementing dimensionality reduction into your AI workflow, explore the different dimensionality reductions techniques, and work through a dimensionality reduction example.

The dimensionality reduction technique is part of the cleansing stage of the process.

## Methods for Dimensional Reduction

There are two main methods for reducing dimensionality:

In **feature selection**, we keep the most important features in the dataset and remove the redundant features. There is no transformation applied to the set of features.

Examples: Backward elimination, Forward selection, and Random forests.

In **feature extraction**, we find a combination of new features and an appropriate transformation is applied to the set of features. The new set of features contains different values rather than the original values.

## Feature Extraction

The first stage in dimensionality reduction process is _feature extraction_ which is the process of selecting a subset of columns for use in the model.

Feature extraction can be further divided into _linear_ methods and _non-linear_ methods:

### Linear methods

Linear methods involve linearly projecting the original data onto a low-dimensional space.

These methods can be applied to linear data but do not perform well on non-linear data.

Principal Component Analysis (PCA), Factor Analysis (FA), Linear Discriminant Analysis (LDA) and Truncated Singular Value Decomposition (SVD) are examples of linear dimensionality reduction methods.

### Non-linear methods

If we are dealing with non-linear data which are frequently used in real-world applications, linear methods discussed so far do not perform well for dimensionality reduction.

Some non-linear methods are well known such as _Manifold learning_.

In this section, we discuss four non-linear dimensionality reduction methods that can be used with non-linear data:  Kernel PCA, t-distributed Stochastic Neighbor Embedding (t-SNE), Multidimensional Scaling (MDS) and Isometric mapping (Isomap).

### Other methods

Those methods only keep the most important features in the dataset and remove the redundant features. So, they are mainly used for feature selection. But, dimensionality reduction happens automatically while selecting the best features! Therefore, they can also be considered to be dimensionality reduction methods.

These methods will improve model accuracy scores and/or boost performance on very high-dimensional datasets.

- Forward Selection
- Backward Elimination
- Random forests


## Feature Extraction Techniques

The first stage in dimensionality reduction process is _feature extraction_ which is the process of selecting a subset of columns for use in the model.

A few of the common feature extraction techniques include:

- **Missing values ratio:** Columns with too many missing values are unlikely to add additional value to a machine learning model. Therefore, when a column exceed a given threshold for missing values it can be excluded for the training set.

- **Low-variance filter:** Columns that have a small variance are unlikely to add value to a machine learning model. Therefore, when a column goes below a given threshold for variance it can be excluded from the training set.

- **High-correlation filter:** If multiple columns contain similar trends then it we only need to use one of the columns. To identify these columns, we can use a Pearson’s Product Momentum Coefficient.

- **Random forest:** One way to eliminate features is to use a random forest technique which creates a decision tree using the target attributes and then leverage the usages statistics to identify the most informative subset of features.

- **Backwards-feature elimination:** The backwards feature elimination is a top down approach that starts with all the features within the dataset and progressively removes one feature at a time until the algorithm has reached the maximum tolerable error.

- **Forward-feature construction:** The forward feature construction takes a bottom up approach which starts with one feature and progressively adds the next feature with the highest increase in performance.


## Linear Algebra Methods

The most well-known dimensionality-reduction techniques are ones that implement linear transformation:

- **Principal component analysis (PCA)** is an unsupervised machine learning algorithm that reduces the dimensions of a dataset while retaining as much information as possible.

The algorithm creates a new set of features from an existing set of features. To avoid a feature with large values dominating the results, all variables should be on the same scale.

In Python’s scikit-learn, we can use the StandardScaler function to ensure all of the variables are on the same scale.

- **Linear Discriminatory Analysis (LDA)** is a supervised technique that seeks to retain as much as possible of the discriminatory power for the dependent variables.

First, the LDA algorithm computes the separability between classes. Second, it computes the distance between the sample of each class and the mean. Finally, it produces the dataset in a lower-dimensionality space.

- **Singular Value Composition (SVD)** extracts the most important features from the dataset which is popular because it is based on simple, interpretable linear algebra models.


## Manifold Learning

One approach to non-linear dimensionality reduction is manifold learning.

**Manifold learning** uses geometric properties to project points onto a lower dimensional space while preserving its structure.

A few of the common manifold learning techniques include:

- Isomap embedding attempts to preserve the relationships within the dataset by producing an embedded dataset.

Isomaps start with producing neighborhood network. Next, it estimates the geodesic distance (the shortest path between two points on a curved surface) between all pairs of points. Finally, it uses eigenvalue decomposition of the geodesic distance matrix to identify a low-dimensional embedding of the dataset.

- Locally linear embedding (LLE) attempts to preserve the relationship within the dataset by producing an embedded dataset.

First, it finds the k-nearest neighbours (kNN) of the points. Second, it estimates each data vector as a combination of it’s kNN. Finally, it creates low-dimensional vectors that best reproduce these weights.

There are two benefits of LLE algorithm:

1. LLE is able to detect more features that the linear algebra methods.

2. LLE is more efficient compared to other algorithms.

- t-Distributed Stochastic Neighbour. t-SNE is sensitive to local structures. It is one of the best for visualization purposes, and it is helpful in understanding theoretical properties of a dataset. However, it is one of the most computationally expensive approaches and other techniques such as missing values ratio should be used before applying this technique. Also, all the features should be scaled before applying this technique.

No single dimensionality reduction technique consistently provides the ‘best’ results. Therefore, the data analyst should explore a range of options and combinations of different dimensionality-reduction techniques, so they move their model closer to the optimal solution.




## References

[1]: E. Alpaydin, Introduction to Machine Learning, 3rd ed., MIT Press, ISBN: 978-0262028189, 2014.

[2]: S. Raschka. and V. Mirjalili, Python Machine Learning, 2nd ed. Packt, ISBN: 978-1787125933, 2017.

[3]: [A beginner’s guide to dimensionality reduction in Machine Learning](https://towardsdatascience.com/dimensionality-reduction-for-machine-learning-80a46c2ebb7e)

[4]: [Techniques for Dimensionality Reduction](https://towardsdatascience.com/techniques-for-dimensionality-reduction-927a10135356)

[5]: [An Intuitive Guide to PCA](https://towardsdatascience.com/an-intuitive-guide-to-pca-1174055fc800)

[6]: [11 Dimensionality reduction techniques you should know in 2021](https://towardsdatascience.com/11-dimensionality-reduction-techniques-you-should-know-in-2021-dcb9500d388b)

[7]: [A Guide to Dimensionality Reduction in Python](https://towardsdatascience.com/a-guide-to-dimensionality-reduction-in-python-ce0c6ab91986?source=rss----7f60cf5620c9---4)

[8]: [Major Problems of Machine Learning Datasets: Part 3](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-3-eae18ab40eda)


[The Similarity between t-SNE, UMAP, PCA, and Other Mappings](https://towardsdatascience.com/the-similarity-between-t-sne-umap-pca-and-other-mappings-c6453b80f303)
