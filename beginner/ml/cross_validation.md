# Cross Validation

## What is model validation?

**Model validation** is the process by which we ensure that our models can perform acceptably in "the real world". In more technical terms, model validation allows you to predict how your model will perform on datasets not used in the training.

Model validation is important because we do not actually care how well the model predicts data we trained it on. 

We already know the target values for the data we used to train a model, so it is more important to consider how robust and capable a model is when tasked to model new datasets of the same distribution and characteristics, but with different individual values from our training set.

## Holdout validation

The first form of model validation introduced is usually what is known as _holdout validation_ which is often considered to be the simplest form of cross validation, so it is the easiest to implement.

In holdout validation, we split the data into a training and testing set. 

- The training set will be what the model is created on and the testing data will be used to validate the generated model. 

- Although there are (fairly easy) ways to do this using pandas methods, we can make use of scikit-learns `train_test_split` method to accomplish this.

We use `train_test_split` with three parameters: the input (X) data, the target (y) data, and the percentage of data we would like to put into the test dataset. 

In this case 25% (common split is usually 70–30, depending on a multitude of factors about your data). 

Then we assign the split X and y data to a set of new variables to work with later.

```py
    import pandas as pd
    import numpy as np

    # import scikit learn databases
    from sklearn import datasets
    from sklearn import train_test_split

    # import california housing data from sklearn and store data into a variable
    calihouses = datasets.fetch_california_housing()
    calidata = calihouses.data

    # define the columns names of the data then convert to dataframe
    headers = calihouses.feature_names
    df = pd.DataFrame(calidata, columns=headers)

    # print the df and shape to get a better understanding of the data
    print(df.shape)
    print(df)

    # first store all target data to a variable
    y = calihouses.target

    # create testing and training sets for hold-out verification using scikit learn method
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.25)

    # validate set shapes
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

```

Now that we have created our test/train split,  we can create a model and generate some predictions based on the train data. 

```py
    # time function using .time methods for later comparison
    from timeit import default_timer as timer

    start_ho = timer()

    # fit a model using linear model method from sklearn
    from sklearn import linear_model

    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)

    # generate predictions
    predictions = lm.predict(X_test)
    end_ho = timer()

    # calcualte function runtime
    time_ho = (end_ho - start_ho)

    # show predictions
    print(predictions)
```

We start by graphing our given target data vs our predicted target data to give us a visualization of how our model performs.

```py
    # import seaborn and plotly
    import matplotlib
    from matplotlib import pyplot as plt
    import seaborn as sns

    # set viz style
    sns.set_style('dark')

    # plot the model
    plot = sns.scatterplot(y_test, predictions)
    plot.set(xlabel='Given', ylabel='Prediction')

    # generate and graph y = x line
    x_plot = np.linspace(0,5,100)
    y_plot = x_plot
    plt.plot(x_plot, y_plot, color='r')
```


## What are bias and variance in the context of model validation?

To understand bias and variance, let us first address overfitiing and underfitting.

An overfit model is generated when the model is so tightly fit to the training data that it may account for random noise or unwanted trends which will not be present or useful in predicting targets for subsequent datasets.

An underfit model occurs when the model is not complex enough to account for general trends in the data which would be useful in predicting targets in subsequent datasets.

[Example Graph](https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html)

When creating a model, we account for a few types of error: validation error, testing error, error due to bias, and error due to variance in a relationship known as the **bias variance trade-off**.

[Example Graph](http://www.luigifreda.com/2017/03/22/bias-variance-tradeoff/)

As mentioned earlier, we want to know how the model will perform on new unseen data including _validation error_ which is comprised of error due to bias and error due to variance (training error does not provide information on how the model will perform on future datasets).

**Minimizing validation error** requires finding the point of model complexity where the combination of bias and variance error is minimized, as shown in the linked visual. 

As model complexity increases, error due to bias decreases while error due to variance increases, creating the bias-variance trade-off which we will seek to address later with various methods of cross validation.

### Bias

Bias is the error resulting from the difference between the expected value of a model and the actual (or correct) value for which we want to predict over multiple iterations. 

In the scientific concepts of accuracy and precision, bias is very similar to _accuracy_.

### Variance

Variance is defined as the error resulting from the variability between different data predictions in a model. 

In variance, the correct value does not matter as much as the range of differences in value between the predictions. 

Variance also comes into play more when we run multiple model creation trials.

[Bias/Variance Visual](http://scott.fortmann-roe.com/docs/BiasVariance.html)

In machine learning, bias and variance are often discussed together as a **bias-variance tradeoff** which means that minimizing one error effectively makes the other more likely to be present when creating and assessing a model. 

Ideally, we would seek a model whose tradeoff results in both low bias and low variance and we would look to achieve this by using _cross validation_. 

Depending on characteristics of the dataset, one method of cross validation is likely to be more ideal to achieving the bias-variance tradeoff when creating and assessing a model.


## What is cross validation?

What if the split we made just happened to be very conducive to this model? 

What if the split we made introduced a large skew into the data? 

Did we significantly reduce the size of our training dataset by splitting it like that?

**Cross Validation** is a method that splits the data in creative ways in order to obtain the better estimates of "real world" model performance and minimize validation error.

Remember those questions we asked about hold out validation? Cross validation is our answer.

### K-Fold Cross Validation

**K-fold validation** is a popular method of cross validation which shuffles the data and splits it into _k_ number of folds (groups). 

In general, K-fold validation is performed by taking one group as the test data set and the other k-1 groups as the training data, fitting and evaluating a model, and recording the chosen score. 

This process is repeated with each fold (group) as the test data and all the scores averaged to obtain a more comprehensive model validation score.

[K-fold Visual](http://www.ebc.cat/2017/01/31/cross-validation-strategies/#k-fold)

When choosing a value for _k_ each fold (group) should be large enough to be **representative** of the model (usually k=10 or k=5) and small enough to be computed in a reasonable amount of time. Depending on the dataset size, different _k_ values can sometimes be experimented with.

As a general rule, as _k_ increases, bias decreases and variance increases.

Let us work though an example with our dataset from earlier.

We will make use of a linear model again, but this time do model validation with scikit learn’s `cross_val_predict` method which will do most of the heavy lifting in generating K-Fold predictions. Here we set k=10.

```py
    # store data as an array
    X = np.array(df)

    # again, timing the function for comparison
    start_kfold = timer()

    # use cross_val_predict to generate K-Fold predictions
    lm_k = linear_model.LinearRegression()
    k_predictions = cross_val_predict(lm_k, X, y, cv=10)
    print(k_predictions)

    end_kfold = timer()
    kfold_time = (end_kfold - start_kfold)
```

The `cross_val_predict` method takes the model used on the data, the input and target data, as well as a `cv` argument (essentially our k value) and returns the predicted values for each input. Now we can plot the predictions as we did with the hold out method.

```py
    # plot k-fold predictions against actual
    plot_k = sns.scatterplot(y, k_predictions)
    plot_k.set(xlabel='Given', ylabel='Prediction')

    # generate and graph y = x line
    x_plot = np.linspace(0,5,100)
    y_plot = x_plot
    plt.plot(x_plot, y_plot, color='r')
```

Now we can get the scores of the 10 generated models and plot them in a graph.

```py
    kfold_score_start = timer()

    # find the mean score from the k-fold models usinf cross_val_score
    kfold_scores = cross_val_score(lm_k, X, y, cv=10, scoring='neg_mean_squared_error')
    print(kfold_scores.mean())
    kfold_score_end = timer()
    kfold_score_time = (kfold_score_end - kfold_score_start)

    # plot scores
    sns.distplot(kfold_scores, bins=5)
```

Notice that the score is a little farther from zero than the holdout method (not good). We will discuss that later.

### Leave One Out Cross Validation

Leave One Out Cross Validation (LOOCV) can be considered a type of K-Fold validation where `k=n` is the number of rows in the dataset. 

The methods are quire similar but you will notice that running the following code will take much longer than previous methods.

### Stratified Cross-validation

Cross-validation implemented using _stratified_ sampling ensures that the proportion of the feature of interest is the same across the original data, training set, and the test set so that no value is over/under-represented in the training and test sets which gives a more accurate estimate of performance/error.


## Where and when should different methods be implemented?

As we noticed in the results of our comparison, we can see that the LOOCV method takes a lot longer to complete than the other two methods. This is because that method creates and evaluates a model for each row in the dataset (in this case over 20,000). Even though our MSE is a little lower, this may not be worth it given the additional computational requirements.

Here are some heuristics which can help in choosing a method.

### Hold out method

The hold out method can be effective and computationally inexpensive on very large datasets, or on limited computational resources. It is also often easier to implement and understand for beginners. However, it is very rarely good to apply to small datasets since it can significantly reduce the training data available and hurt model performance.

### K-Fold Cross Validation

K-Fold can be very effective on medium sized datasets, though by adjusting the K value we can significantly alter the results of the validation. Let us add to our rule from earlier: as k increases, bias decreases, and variance and computational requirements increase. K-Fold cross validation is likely the most common of the three methods due to the versatility of adjusting K-values.

### LOOCV

LOOCV is most useful in small datasets as it allows for the smallest amount of data to be removed from the training data in each iteration. However, in large datasets the process of generating a model for each row in the dataset can be incredibly computationally expensive and thus prohibitive for larger datasets.


## What are some advantages and disadvantages of the different cross validation techniques?

## Holdout Validation

In holdout validation, we are doing nothing more than performing a simple train/test split in which we fit our model to our training data and apply it to our testing data to generate predicted values. We "hold out" the testing data to be strictly used for prediction purposes only. Holdout validation is NOT a cross validation technique. But we must discuss the standard method of model evaluation so that we can compare its attributes with the actual cross validation techniques.

When it comes to code, holdout validation is easy to use. The implementation is simple and does not require large dedications to computational power and time complexity. Moreover, we can interpret and understand the results of holdout validation better as they do not require us to figure out how the iterations are performing in the grand scheme of things.

However, holdout validation does not preserve the statistical integrity of the dataset in many cases. For instance, a holdout validation that splits the data into training and testing segments causes bias by not incorporating the testing data into the model. The testing data could contain some important observations. This would result in a detriment to the accuracy of the model. Furthermore, this will cause an underfitting and overfitting of the data in addition to an introduction of validation and/or training error.

## K-fold

In K-fold cross validation, we answer many of the problems inherent in holdout validation such as underfitting/overfitting and validation and training error. This is done by using all of the observations in our validation set at some iteration. We compute an average accuracy score of all the accuracy scores that are calculated in each _k_ iteration. By doing so, we minimize bias and variation that may be present in our initial model evaluation technique, holdout validation.

However, in terms of computational power, k-fold cross validation is very costly. The computer has to perform several iterations to generate a proper accuracy score. The accuracy score of the model will in theory increase with each added _k_ iteration. This will decrease bias while increasing variation.

We will see an example of this later in this article when we attempt to apply k-fold validation to a very large dataset that contains about 580,000 observations.

### LOOCV

LOOCV is very similar to K-fold, with a special case in which _k_ is equal to the length (or number of samples/rows) of the whole dataset. Thus, the training set will be of length k-1, and the testing set will be a single sample of the data. LOOCV is particularly useful in the case that our data set is not large enough to sensibly do Kfold. LOOCV is also less computationally expensive in general, although it is usually due to the inherently smaller datasets that tend utilize it.

However, LOOCV tends to yield high variance due to the fact that the method would pick up on all of the possible noise and outlier values in the data through the individual testing values. LOOCV would be very computationally expensive for very large data sets; in this case, it would be better to use regular k-fold.


## When would you not want to use cross validation?

Cross validation becomes a computationally expensive and taxing method of model evaluation when dealing with large datasets. Generating prediction values ends up taking a very long time because the validation method has to run _k_ times in K-Fold strategy, iterating through the entire dataset. Thus, cross validation becomes a very costly model evaluation strategy in terms of time complexity.

We will examine this phenomenon by performing a normal holdout validation and a K-Fold cross validation on a very large dataset with approximately 580,000 rows. See if you can figure it out, why it works the way it does (and the new data visualizations). Good luck!


## Key Terminology

**Model Validation:** Any process by which a generated model is verified against additional data not used in the process of generating the model. E.g. cross validation, K-Fold validation, hold out validation, etc.

**Cross Validation:** A type of model validation where multiple subsets of a given dataset are created and verified against each-other, usually in an iterative approach requiring the generation of a number of separate models equivalent to the number of groups generated.

**K-Fold Cross Validation:** A type of cross validation where a given dataset is split into k number of groups and k number of models are generated. One of the groups is chosen as the test data, and the other k-1 groups are used as training data, and model generated and scored. This process is repeated k times such that each k-fold (group) serves as the testing group once.

**LOOCV:** A type of cross validation similar to K-Fold validation, where k is equal to the length of the dataset which the validation is being performed on.

**Bias:** The error resulting from the difference between the expected value(s) of a model and the actual (or “correct”) value(s) for which we want to predict over multiple iterations. In the scientific concepts of accuracy and precision, bias is very similar to accuracy.

**Variance:** The error resulting from the variability between different data predictions in a model. In variance, the correct value(s) don’t matter as much as the range of differences in value between the predictions. Variance also comes into play more when we run multiple model creation trials.

**Underfit:** Occurs when the model is so tightly fit to the training data that it may account for random noise or unwanted trends which will not be present or useful in predicting targets for subsequent datasets.

**Overfit:** Occurs when the model is not complex enough to account for general trends in the data which would be useful in predicting targets in subsequent datasets, such as using a linear fit on a polynomial trend.

**Bias-Variance Trade-off:** The idea that as error due to bias decreases error due to variance increases, creating a trade-off which should be minimized in model validation, and other circumstances.



## Guide to Cross-Validation Techniques

Cross-validation (CV) is a resampling technique in machine learning which ensures the performance and efficiency of the machine learning model by evaluating it over a subset of the dataset.

CV validates the model performance by training it on a subset of the input data and then testing it on the subset of the input data on which it has not been trained earlier. 

CV evaluates how a model will perform on an independent/new test dataset by resampling the same training data you already have.

### Why do we need Cross-Validation?

The main purpose of CV is to validate a machine learning model’s accuracy and performance on different subsets of data.

Sometimes we achieve high accuracy in our machine learning model, but does no ensure it will generalize to new data which is known as overfitting.

- To avoid overfitting, we can use cross-validation.

- Cross-validation can help determine optimal hyperparameter values during hyperparameter tuning.

### Hold-out cross-validation

Hold-out cross-validation is one of the simplest cross-validation techniques. In this technique, we randomly divide the total dataset into two parts: training data and testing data.

The model is trained on the training dataset and then its performance is evaluated on the test dataset.

### Leave P out cross-validation

If we have n samples in our dataset, then p samples are used for validation of the model and n-p samples are used for training the model.

What makes leave p out differ from k-fold validation is that it creates overlapping test sets as it tries all possible combinations to divide the dataset into training and validation sets which it requires high computation for a large p-value.

The main drawback of this technique is that this method is an exhaustive method which learns and tests all the possible ways to divide the original dataset into training and validation sets. Due to this the number of iterations grows higher and is not computationally feasible. Therefore, it should not be used with large datasets. 

This technique can also quickly become unusable for even smaller datasets.

### K-fold cross-validation

In this cross-validation technique, we divide the dataset into k subsets of equal size which is known as a fold. 

In each iteration, out of k subsets, 1 subset is used for the model validation and the remaining k-1 subsets are used for the training of our machine learning model.

This process is repeated k times until all the data is used for the validation and the remaining for the training model. Then the final accuracy is calculated by taking the average of the accuracy score in each iteration.

In this technique, each data point gets to be a part of the test data once only and gets to be a part of the training data for k-1 times.

The main disadvantage of this technique is the high computational cost as the model needs to be trained k times.

### Stratified k-fold cross-validation

Stratified k-fold cross-validation is an extension of k-fold cross-validation, which uses stratified sampling instead of just random sampling (in the case of k-fold). Like k-fold cross-validation, it also splits the data into k subsets.

The k-fold cross-validation technique does not perform very well when there is an imbalance in our dataset and due to the random splitting of our data, so there is a possibility that data might be imbalanced. 

In stratified k-fold, data is split in a stratified manner.

Here, the data gets split in such a way that it represents all classes from the population data. It splits the data into k folds in such a way that each fold has the same ratio of instances of the target variable that are in the complete dataset.


## Five Cross-validation Techniques

The article [3] discusses five different ways to split your data for validation along with the code from scikit-learn:

1. Basic KFold
2. Stratified KFold
3. Group K-fold
4. StratifiedGroupKFold
5. TimeSeriesSplit



## References

[1] [Cross Validation: A Beginner’s Guide](https://towardsdatascience.com/cross-validation-a-beginners-guide-5b8ca04962cd)

[2] [Guide to Cross-Validation Techniques and Its Types in Machine Learning](https://heartbeat.comet.ml/guide-to-cross-validation-techniques-and-its-types-in-machine-learning-10c1dc0f7a09)

[3] [5 different ways to cross-validate your data](https://medium.com/the-techlife/5-different-ways-to-cross-validate-your-data-376a79b7f205)
