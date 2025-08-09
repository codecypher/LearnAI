# Train-Test Split


## ML Datasets

### Training Dataset

**Training Dataset:** The sample of data used to fit the model.

The actual dataset that we use to train the model (weights and biases in the case of a Neural Network). The model sees and learns from this data.

### Validation Dataset

**Validation Dataset:** The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.

The validation set is used to evaluate a given model, but this is for frequent evaluation. We use this data to _fine-tune_ the model hyperparameters. Therefore, the model occasionally sees this data, but never does it "Learn" from this dataset. We use the validation set results, and update higher level hyperparameters. So the validation set affects a model, but only indirectly.

### Test Dataset

**Test Dataset:** The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.

The Test dataset provides the _gold standard_ used to evaluate the model. It is only used once a model is completely trained (using the train and validation sets). The test set is generally what is used to evaluate competing models (on many Kaggle competitions, the validation set is released initially along with the training set and the actual test set is only released when the competition is about to close, and it is the result of the the model on the Test set that decides the winner). Many times the validation set is used as the test set, but it is not good practice. The test set is generally well curated. It contains carefully sampled data that spans the various classes that the model would face, when used in the real world.

### About the dataset split ratio

Now you might be looking for recommendations on how to split your dataset into Train, Validation and Test sets.

This mainly depends on two things. 1) the total number of samples in your data and 2) the actual model you are training.

Some models need substantial data to train on, so in this case you would optimize for the larger training sets. Models with very few hyperparameters will be easy to validate and tune, so you can probably reduce the size of your validation set, but if your model has many hyperparameters, you would want to have a large validation set as well (although you should also consider cross validation). Also, if you happen to have a model with no hyperparameters or ones that cannot be easily tuned, you probably do not need a validation set too!

### Note on Cross Validation

Many a times, people first split their dataset in 2 — Train and Test. Then, they set aside the Test set, and randomly choose X% of their Train dataset to be the actual Train set and the remaining (100-X)% to be the Validation set, where X is a fixed number (say 80%). The model is then iteratively trained and validated on these different sets. There are multiple ways to do this, and it is commonly known as **Cross Validation**. Basically, you use your training set to generate multiple splits of the Train and Validation sets. Cross validation avoids overfitting and is getting more and more popular, with K-fold Cross Validation being the most popular method of cross validation. Check this out for more.



## Train-Test Split

A key step in ML is the choice of model.  

> Split first, normalize later.

A train-test split conists of the following:

1. Split the dataset into training, validation and test set

2. We normalize the training set only (fit_transform). 

3. We normalize the validation and test sets using the normalization factors from train set (transform).

Suppose we fit the model with the train set while evaluating with the test set, we would obtain only a _single_ sample point of evaluation with one test set. 

If we have two models and found that one model is better than another based on the evaluation, how can we know this is not by chance?

**Solution:** the train-validation-test split


## Train-Validation-Test Split

Here are the steps for a train-validation-test split:

1. The model is fit on the train data set. 

2. The fitted model is used to predict the responses for the observations on the validation set. 

3. The test set is used to provide an unbiased evaluation of the final model that has been fit on the train dataset. 

If the data in the test set has never been used in training (such as cross-validation), the test set is also called a _holdout_ data set.


The reason for such practice is the concept of preventing _data leakage_ which is discussed below. 


What we should care about is the evaluation metric on the _unseen data_. 

Therefore, we need to keep a slice of data from the entire model selection and training process and save it for the final evaluation called the test set. 

The process of _cross-validation_ is the following:

1. The train set is used to train a few candidate models. 

2. The validation set is used to evaluate the candidate models. 

3. One of the candidates is chosen. 

4. The chosen model is trained with a new train set.

5. The final trained model is evaluated using the test set. 

The dataset for evaluation in step 5 and the one we used in steps 2 are different because we do not want _data leakage_. 

If the test and validation sets were the same, we would see the same score that we have already seen from cross validation or the test score would be good because it was part of the data we used to train the model and we  adapted the model for that test dataset.

Thus, we make use of the test dataset that was never used in previous steps (holdout set) to evaluate the performance on unseen data which is called _generalization_.  


## Code Samples

```py
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

The above example randomly splits the data observations into a training set containing 80% of the original observations and a test set housing the remaining 20% instances.

```py
  # split the dataset into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
```    


## References

[1]: [About Train, Validation, and Test Sets in Machine Learning](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)

