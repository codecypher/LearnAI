# Factor Analsysis

Principal Component Analysis (PCA), Factor Analysis (FA), Linear Discriminant Analysis (LDA) and Truncated Singular Value Decomposition (SVD) are examples of linear dimensionality reduction methods. 


## Dimensionality Reduction

 Factor Analysis is a technique used to express data with a reduced number of variables. Reducing the number of variables in a data is helpful method to simplify large dataset by decreasing the variables without loosing the generality of it. 

The Scikit-learn API provides the `FactorAnalysis` model that performs a maximum likelihood estimate (MLE) of the loading matrix using the Singular Value Decomposition (SVD) approach.

In this tutorial, we learn how to use the FactorAnalysis model to reduce the data dimension and visualize the output in Python using the MNIST dataset.


----------


## Introduction to Factor Analysis

Factor Analysis (FA) is an exploratory data analysis method used to search influential underlying factors or latent variables from a set of observed variables which helps in data interpretations by reducing the number of variables. 

FA extracts maximum common variance from all variables and puts them into a common score.

Factor analysis is widely utilized in market research, advertising, psychology, finance, and operation research. 

Market researchers use factor analysis to identify price-sensitive customers, identify brand features that influence consumer choice, and helps in understanding channel selection criteria for the distribution channel.

Factor analysis is a linear statistical model that is used to explain the variance among the observed variable and condense a set of the observed variable into the unobserved variable called _factors_. 

- An observed variable in modeled as a linear combination of factors and error terms. 

- A factor or latent variable is associated with multiple observed variables which have common patterns of responses. Each factor explains a particular amount of variance in the observed variables which helps in data interpretations by reducing the number of variables.

Factor analysis is a method for investigating whether a number of variables of interest X1, X2, ..., Xl are linearly related to a smaller number of unobservable factors F1, F2, ..., Fk.

Assumptions:

1. There are no outliers in data.
2. Sample size should be greater than the factor.
3. There should not be perfect multicollinearity.
4. There should not be homoscedasticity between the variables.

### Types of Factor Analysis

- Exploratory Factor Analysis (EFA): the most popular factor analysis approach among social and management researchers. The basic assumption is that any observed variable is directly associated with any factor.

- Confirmatory Factor Analysis (CFA): the basic assumption is that each factor is associated with a particular set of observed variables. Thus, CFA confirms what is expected on the basic.

### How does factor analysis work?

The primary objective of factor analysis is to reduce the number of observed variables and **find unobservable variables**. This conversion of the observed variables to unobserved variables can be achieved in two steps:

- Factor Extraction: In this step, the number of factors and approach for extraction selected using variance partitioning methods such as principal components analysis (PCA) and common factor analysis (CFA).

- Factor Rotation: In this step, rotation tries to convert factors into uncorrelated factors — the main goal of this step to improve the overall interpretability. There are lots of rotation methods that are available such as: Varimax rotation method, Quartimax rotation method, and Promax rotation method.


### Terminology

#### What is a factor?

A _factor_ is a latent variable that describes the association among the number of observed variables. The maximum number of factors is equal to the number of observed variables. 

- Every factor explains a certain variance in observed variables. 
- The factors with the lowest amount of variance are dropped. 

Factors are also known as latent variables or hidden variables or unobserved variables or hypothetical variables.

#### What are the factor loadings?

The _factor loading_ is a matrix that shows the relationship of each variable to the underlying factor. 

The factor loading shows the correlation coefficient for observed variable and factor which is the variance explained by the observed variables.

#### What is Eigenvalues?

Eigenvalues represent variance explained each factor from the total variance which are also known as characteristic roots.

#### What are Communalities?

_Commonalities_ are the sum of the squared loadings for each variable which represent the common variance. It ranges from 0-1 and values close to 1 represent more variance.

#### What is Factor Rotation?

_Rotation_ is a tool for better interpretation of factor analysis which can be orthogonal or oblique. It re-distributes the commonalities with a clear pattern of loadings.

### Choosing the Number of Factors

**Kaiser criterion** is an analytical approach that is based on the principal that the more significant proportion of variance explained by a factor will be selected. 

The eigenvalue is a good criterion for determining the number of factors. 

Generally, an eigenvalue greater than 1 will be considered as selection criteria for the feature.

The _graphical approach_ is based on the visual representation of the factors' eigenvalues also called _scree plot_ which helps us to determine the number of factors where the curve makes an elbow.


### Factor Analysis vs PCA

- PCA components explain the maximum amount of variance while factor analysis explains the covariance in data.

- PCA components are fully orthogonal to each other whereas factor analysis does not require factors to be orthogonal.

- PCA component is a linear combination of the observed variable while in FA the observed variables are linear combinations of the unobserved variable or factor.

- PCA components are uninterpretable. In FA, underlying factors are can be labeled and interpreted.

- PCA is a kind of dimensionality reduction method whereas factor analysis is the latent variable method.

- PCA is a type of factor analysis. PCA is observational whereas FA is a modeling technique.


### Factor Analysis in python using factor_analyzer package


## FA vs PCA vs DRO

**Factor analysis (FA)** is a well-known methods of classical multivariate analysis which is useful when a large number of variables are believed to be determined by a relatively few common causes or factors. 

FA involves the calculation of a variable-by-variable correlation matrix that is used to extracts new variables which are a linear combination of the original variables. 

The coefficients in each linear combination are known as factor loadings which can be used to identify the variables that are most closely related to a factor.  Thus, the factors extracted are the maximum likelihood estimates of the factor loadings. 

Perhaps the most widely used method for determining the number of factors is using eigenvalues greater than one.

FA is often confused with **principal components analysis (PCA)**. The two methods are related since factor analysis is essentially equivalent to principal components analysis if the errors in the factor analysis model are assumed to all have the same variance. 

**Dimensionality reduction operation (DRO)** is primarily used to deal with the high-dimensionality of stock data but there have been no studies comparing the performance of DNN models based on different DRO techniques. 


Lv, Wang, Li, and Xiang [54] evaluated 424 S&P 500 index component stocks (SPICS) and 185 CSI 300 index component stocks (CSICS) and concluded the following:

1. There was no significant difference in performance of the algorithms. 

2. DRO does not significantly improve the execution speed of any of the DNN models.



## References

[Dimensionality Reduction Example with Factor Analysis in Python](https://www.datatechnotes.com/2021/02/dimension-reducing-with-factor-analysis-in-python.html)

[Introduction to Factor Analysis in Python](https://www.datacamp.com/community/tutorials/introduction-factor-analysis)

