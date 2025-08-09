# Machine Learning Crash Course

## Framing: Key ML Terminology

What is (supervised) machine learning?

ML systems learn how to combine input to produce useful predictions on never-before-seen data.

A **label** is the target varisvle the tgat we are predicting — the y variable in simple linear regression. 

A **feature** is an input variable — the X variable in simple linear regression. 

A simple machine learning project might use a single feature, while a more sophisticated machine learning project could use millions of features. 

An **example** is a particular instance of data X. 

We break examples into two categories:

- labeled examples
- unlabeled examples

A **labeled example** includes both feature(s) and the label which are used to **train** the model. 

An **unlabeled example** contains features but not the label. 

After we have trained our model with labeled examples, we use that model to predict the label on unlabeled examples. 

### Models

A **model** defines the relationship between features and label. 

For example, a spam detection model might associate certain features strongly with "spam". 

Let us highlight two phases of a model's life:

- **Training** means creating or **learning** the model which means we show the model labeled examples and enable the model to gradually learn the relationships between features and label.

- **Inference** means applying the trained model to unlabeled examples which means we use the trained model to make useful predictions (y').

### Regression vs. Classification

A **regression** model predicts continuous values. 

A **classification** model predicts discrete values.


## Linear Regression

**Linear regression** is a method for finding the straight line or hyperplane that best fits a set of points. 

By convention in machine learning, ee write the equation for a model slightly differently:

```
  y' = w1 x1 + b
```

where

  -  y' is the predicted label (a desired output).

  - b is the bias (the y-intercept), sometimes referred to as w0. 

  - w1 is the weight of feature 1. Weight is the same concept as the "slope" in the traditional equation of a line.

  - x1 is a feature (a known input).
  
To **infer** (predict) the temperature for y' a new chirps-per-minute value x1 just substitute the x1 value into this model.

Although this model uses only one feature, a more sophisticated model might rely on multiple features, each having a separate weight (w1, w2, ...). 

For example, a model that relies on three features might look as follows:


## Training and Loss

**Training** a model simply means learning (determining) good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

Loss is the penalty for a bad prediction, so **loss** is a number indicating how bad the model's prediction was on a single example. 

If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater than zero. 

The goal of training a model is to find a set of weights and biases that have _low_ loss (on average) across all examples. 

For example, Figure 3 shows a high loss model on the left and a low loss model on the right. 

### Squared loss: a popular loss function

The linear regression models we examine here use a loss function called **squared loss** or **L2 loss**. 

The squared loss for a single example is: (y - y')^2

**Mean square error (MSE)** is the average squared loss per example over the whole dataset. 

To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples:

Although MSE is commonly-used in machine learning, it is neither the only practical loss function nor the best loss function for all circumstances.

### Reducing Loss

To train a model, we need a good way to reduce the model’s loss. 

An iterative approach is one widely used method for reducing loss, and is as easy and efficient as walking down a hill.


## An Iterative Approach

The previous module introduced the concept of loss. 

Here, we learn how a machine learning model iteratively reduces loss.

Figure 1. An iterative approach to training a model.

We use this same iterative approach throughout the Machine Learning Crash Course, detailing various complications, particularly within that stormy cloud labeled "Model (Prediction Function)." 

Iterative strategies are prevalent in machine learning, primarily because they scale so well to large data sets.

The "model" takes one or more features as input and returns one prediction (y') as output. 

To simplify, consider a model that takes one feature and returns one prediction:

What initial values should we set for b and w1? 
 
 For linear regression problems, it turns out that the starting values are not important, so b = 0 = w1. 
 
 The "Compute Loss" part of the diagram is the loss function that the model will use. 
 
 Suppose we use the squared loss function then the loss function takes in two input values:

  - y': The model's prediction for features x

  - y: The correct label corresponding to features x.

Finally, we reach the "Compute parameter updates" part of the diagram where the machine learning system examines the value of the loss function and generates new values for b and w1. 
 
 For now, just assume that this mysterious box devises new values and then the machine learning system re-evaluates all those features against all those labels, yielding a new value for the loss function which yields new parameter values. 
 
 And the learning continues iterating until the algorithm discovers the model parameters with the lowest possible loss. 
 
 Usually, you iterate until overall loss stops changing or at least changes extremely slowly so that the model has **converged**.
 
 **Key Point:** A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.
 
## Gradient Descent

The iterative approach diagram (Figure 1) contained a green hand-wavy box entitled "Compute parameter updates." 

We now replace that algorithmic fairy dust with something more substantial.

Suppose we had the time and the computing resources to calculate the loss for all possible values of w1. 

For the kind of regression problems we have been examining, the resulting plot of loss vs w1 will always be convex which means the plot will always be bowl-shaped. 

Figure 2. Regression problems yield convex loss vs. weight plots.

Convex problems have only one minimum (one place where the slope is exactly 0) where the loss function converges.

Calculating the loss function for every conceivable value of w1 over the entire data set would be an inefficient way of finding the convergence point. 

Let us examine a better mechanism called **gradient descent**.

The first stage in gradient descent is to pick a starting value (a starting point) for w1. 

The starting point does not matter much, so many algorithms simply set w1 to 0 or pick a random value. 

The gradient descent algorithm calculates the gradient of the loss curve at the starting point. 

Here in Figure 3, the gradient of the loss is equal to the derivative (slope) of the curve, and tells us which way is "warmer" or "colder." 

When there are multiple weights, the gradient is a vector of partial derivatives with respect to the weights.

Note that a gradient is a vector, so it has both direction and magnitude. 

The gradient always points in the direction of greatest (steepest) increase in the loss function. In contrast, the negative of the gradient points in the direction of greatest decrease of the function.

The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.

To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point as shown in the following figure:

The gradient descent then repeats this process, edging ever closer to the minimum.

**NOTE:** When performing gradient descent, we generalize the above process to tune all the model parameters simultaneously. 

For example, to find the optimal values of both w1 and the bias b, we calculate the gradients with respect to both w1 and b. 

Next, we modify the values of w1 and b based on their respective gradients. 

Then we repeat these steps until we reach minimum loss.


## Learning Rate

Recall thta the gradient vector has both a direction and a magnitude. 

Gradient descent algorithms multiply the gradient by a scalar known as the **learning rate** or **step size**p to determine the next point. 

For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.

**Hyperparameters** are the knobs that programmers tweak in machine learning algorithms. 

Most machine learning programmers spend a fair amount of time tuning the learning rate. 

If you pick a learning rate that is too small, learning will take too long. 

If you specify a learning rate that is too large, the next point will perpetually bounce haphazardly across the bottom of the well. 

There is an ideal learning rate for every regression problem that is related to how flat the loss function is. 

If you know the gradient of the loss function is small then you can safely try a larger learning rate which compensates for the small gradient and results in a larger step size.

The ideal learning rate in one-dimension  the inverse of the second derivative of f(x) at x. 

The ideal learning rate for 2 or more dimensions is the inverse of the Hessian (the matrix of second partial derivatives).

The story for general convex functions is more complex.

**NOTE:**  In practice, finding a "perfect" (or near-perfect) learning rate is not essential for successful model training. The goal is to find a learning rate large enough that gradient descent converges efficiently, but not so large that it never converges.


## Stochastic Gradient Descent

In gradient descent, a **batch** is the total number of examples you use to calculate the gradient in a single iteration. 

So far we hav assumed that the batch has been the entire data set. 

In reality, data sets often contain billions or even hundreds of billions of examples with huge numbers of features. Thus, a batch can be enormous. 

A very large batch may cause even a single iteration to take a long time to compute.

A large data set with randomly sampled examples probably contains redundant data. In fact, redundancy becomes more likely as the batch size grows. 

Some redundancy can be useful to smooth out noisy gradients, but enormous batches tend not to carry much more predictive value than large batches.

What if we could get the right gradient on average for much less computation? 

By choosing examples at random from our data set, we could estimate a big average from a much smaller one. 

**Stochastic gradient descent (SGD)** takes this idea to the extreme. 

SGD uses a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example comprising each batch is chosen at _random_.

**Mini-batch stochastic gradient descent (mini-batch SGD)** is a compromise between full-batch iteration and SGD. 

A mini-batch is typically between 10 and 1,000 examples, chosen at random. 

Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

To simplify the explanation, we focused on gradient descent for a single feature, but gradient descent also works on feature sets that contain multiple features.


## Generalization

**Generalization** refers to your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.

### Peril of Overfitting

The model shown in Figures 2 and 3 overfits the peculiarities of the data it trained on. An overfit model gets a low loss during training but does a poor job predicting new data. If a model fits the current sample well, how can we trust that it will make good predictions on new data? As you'll see later on, overfitting is caused by making a model more complex than necessary. The fundamental tension of machine learning is between fitting our data well, but also fitting the data as simply as possible.

Machine learning's goal is to predict well on new data drawn from a (hidden) true probability distribution. Unfortunately, the model can't see the whole truth; the model can only sample from a training data set. If a model fits the current examples well, how can you trust the model will also make good predictions on never-before-seen examples?

Ockham's Razor: The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.

In modern times, we have formalized Ockham's razor into the fields of statistical learning theory and computational learning theory. 

These fields have developed **generalization bounds** which is a statistical description of a model's ability to generalize to new data based on factors such as:

  - the complexity of the model
  - the model's performance on training data

While the theoretical analysis provides formal guarantees under idealized assumptions, they can be difficult to apply in practice. 

A machine learning model aims to make good predictions on new, previously unseen data. 

How would you get the previously unseen data? One way is to divide your data set into two subsets:

  - training set: a subset to train a model.
  - test set: a subset to test the model.

Good performance on the test set is a useful indicator of good performance on the new data in general, assuming that:

  - The test set is large enough.
  - You do not reuse the same test set. 


### The ML fine print

The following three basic assumptions guide generalization:

  1. We draw examples **independently and identically (iid)** at random from the distribution. 
  
  i.i.d. refers to the randomness of variables
  i.i.i.d. means examples do not influence each other
  
  2. The distribution is stationary which means the distribution does not change within the data set.
  
  3. We draw examples from partitions from the same distribution.


In practice, we sometimes violate these assumptions:

  - Consider a model that chooses ads to display. The i.i.d. assumption would be violated if the model bases its choice of ads on what ads the user has previously seen.
  
  - Consider a data set that contains retail sales information for a year. User's purchases change seasonally which would violate stationarity.

When we know that any of the preceding three basic assumptions are violated, we must pay careful attention to metrics.


### Summary

Overfitting occurs when a model tries to fit the training data so closely that it does not generalize well to new data.

If the key assumptions of supervised ML are not met, we lose important theoretical guarantees on our ability to predict on new data.


## Training and Test Sets

A **test set** is a data set used to evaluate the model developed from a training set.

### Training and Test Sets: Splitting Data

The previous module introduced the idea of dividing your data set into two subsets:

  - training set: a subset to train a model.
  - test set: a subset to test the trained model.

Make sure that your test set meets the following two conditions:

  - Is large enough to yield statistically meaningful results.

  - Is representative of the data set as a whole -- do not pick a test set with different characteristics than the training set.

Assuming that your test set meets the preceding two conditions, your goal is to create a model that generalizes well to new data. 

Our test set serves as a proxy for new data. 

**Never train on test data.** If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. 

For example, high accuracy might indicate that test data has leaked into the training set.

Q: We looked at a process of using a test set and a training set to drive iterations of model development. On each iteration, we train on the training data and evaluate on the test data using the evaluation results on test data to guide choices of and changes to various model hyperparameters such as learning rate and features:

A: Doing many rounds of this procedure might cause us to implicitly fit to the peculiarities of our specific test set.


## Validation Set

Partitioning a data set into a training set and test set lets you judge whether a given model will generalize well to new data. However, using only two partitions may be insufficient when doing many rounds of hyperparameter tuning.

You can greatly reduce your chances of overfitting by partitioning the data set into the three subsets (train/validation/test). 

Use the validation set to evaluate results from the training set. Then, use the test set to double-check your evaluation after the model has "passed" the validation set. 

The following figure shows this new workflow:

In this improved workflow:

  1. Pick the model that does best on the validation set.

  2. Double-check that model against the test set. 
  
This is a better workflow because it creates fewer exposures to the test set.

**TIP:** Test sets and validation sets "wear out" with repeated use. That is, the more you use the same data to make decisions about hyperparameter settings or other model improvements, the less confidence you will have that these results actually generalize to new, unseen data.

If possible, it is a good idea to collect more data to "refresh" the test set and validation set. Starting anew is a great reset.


## Representation

We must create a **representation** of the data to provide to a machine learning model with a useful vantage point into the data's key qualities. 

Thus, you must choose the set of features that best represent the data in order to train a model. 

### Feature Engineering

In traditional programming, the focus is on code. 

In machine learning projects, the focus shifts to representation -- one way developers hone a model is by adding and improving its features.

#### Mapping Raw Data to Features

The left side of Figure 1 illustrates raw data from an input data source; the right side illustrates a **feature vector** which is the set of floating-point values comprising the examples in your data set. 

**Feature engineering** means transforming raw data into a feature vector. 

Many machine learning models must represent the features as real-numbered vectors because the feature values must be multiplied by the model weights.

Mapping numeric values

Integer and floating-point data do not need a special encoding because they can be multiplied by a numeric weight.

Mapping categorical values

Categorical features have a discrete set of possible values. 

Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values.

We can accomplish this by defining a mapping from the feature values which we refer to as the **vocabulary** of possible values, to integers. 

Since not every street in the world will appear in our dataset, we can group all other streets into a catch-all "other" category known as an **OOV (out-of-vocabulary) bucket**.

To remove both these constraints, we can  create a binary vector for each categorical feature in our model that represents values as follows:

  - For values that apply to the example, set corresponding vector elements to 1.
  - Set all other elements to 0.

The length of this vector is equal to the number of elements in the vocabulary. 

This representation is called a **one-hot encoding** when a single value is 1, and a **multi-hot encoding** when multiple values are 1.

**TIP:** One-hot encoding extends to numeric data that you do not want to directly multiply by a weight, such as a postal code.

### Sparse Representation

Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a very inefficient representation in terms of both storage and computation time when processing these vectors. 

In this situation, a common approach is to use a **sparse representation** in which only nonzero values are stored. 

In sparse representations, an independent model weight is still learned for each feature value as described above.


## Qualities of Good Features

Now we explore what kinds of values actually make good features within those feature vectors.

### Avoid rarely used discrete feature values

Good feature values should appear more than 5 or so times in a data set which enables a model to learn how the feature value relates to the label. 

Thus, having many examples with the same discrete value gives the model a chance to see the feature in different settings and determine when it is a good predictor for the label.

Conversely, if a feature's value appears only once or very rarely, the model cannot make predictions based on that feature.

### Prefer clear and obvious meanings

Each feature should have a clear and obvious meaning. 

In some cases, noisy data (rather than bad engineering choices) causes unclear values. 

For example, the following user_age_years came from a source that did not check for appropriate values:

### Do not mix "magic" values with actual data

Good floating-point features do not contain peculiar out-of-range discontinuities or "magic" values. 

For example, suppose a feature holds a floating-point value between 0 and 1.

If a user did not enter a quality_rating, perhaps the data set represented its absence with a magic value of -1. 

To explicitly mark magic values, create a Boolean feature that indicates whether or not a quality_rating was supplied. 

Give this Boolean feature a name like is_quality_rating_defined.

In the original feature, replace the magic values as follows:

  - For variables that take a finite set of values (discrete variables), add a new value to the set and use it to signify that the feature value is missing.

  - For continuous variables, ensure missing values do not affect the model by using the mean value of the feature's data.
  
### Account for upstream instability

The definition of a feature should not change over time. 


## Cleaning Data

### Scaling feature values

**Scaling** means converting floating-point feature values from their natural range into a standard range (say 0 to 1 or -1 to +1). 

If a feature set consists of only a single feature, scaling provides little to no practical benefit. 

If a feature set consists of multiple features, feature scaling provides the following benefits:

- Helps gradient descent converge more quickly.

- Helps avoid the "NaN trap" where one number in the model becomes a NaN (when a value exceeds the floating-point precision limit during training) and (due to math operations) every other number in the model also eventually becomes a NaN.

- Helps the model learn appropriate weights for each feature. Otherwise, the model will pay too much attention to the features having a wider range.

You do not have to give every floating-point feature exactly the same scale. Nothing terrible will happen if Feature A is scaled from -1 to +1 while Feature B is scaled from -3 to +3. However, your model will react poorly if Feature B is scaled from 5000 to 100000.

### Outliers

How could we minimize the influence of those extreme outliers? One way would be to take the log of every value:

Clipping the feature value at 4.0 does not mean that we ignore all values greater than 4.0 -- it means that all values that were greater than 4.0 now become 4.0. 

### Binning

In the data set, latitude is a floating-point value. 

It does not make sense to represent latitude as a floating-point feature in our model since there is no linear relationship exists between latitude and housing values.

To make latitude a helpful predictor, we can divide latitudes into "bins". 

### Scrubbing

So far we have assumed that all the data used for training and testing was trustworthy. 

In real-life, many examples in data sets are unreliable due to one or more of the following:

  - Omitted values
  - Duplicate examples
  - Bad labels
  - Bad feature values

We typically "fix" bad examples by removing them from the data set. 

To detect omitted values or duplicated examples, we can write a simple program. 

Detecting bad feature values or labels can be far trickier.

In addition to detecting bad individual examples, you must also detect bad data in the aggregate. 

Histograms are a great mechanism for visualizing your data in the aggregate. 

In addition, getting statistics such as the following can help:

  - Maximum and minimum
  - Mean and median
  - Standard deviation

Consider generating lists of the most common values for discrete features. 


## Feature Crosses

A **feature cross** is a synthetic feature formed by multiplying (crossing) two or more features. 

Crossing combinations of features can provide predictive abilities beyond what those features can provide individually.

### Encoding Nonlinearity

Can you draw a single straight line that neatly separates the sick trees from the healthy trees? No. 

This is a nonlinear problem. 

Any line you draw will be a poor predictor of tree health.

To solve the nonlinear problem shown in Figure 2, create a feature cross. 


A feature cross is a synthetic feature that encodes nonlinearity in the feature space by multiplying two or more input features together (the term cross comes from cross product). 

We can create a feature cross named x3
 by crossing x1 and x2: x3 = x1 x2
 
We treat this newly minted 
 feature cross just like any other feature. 
 
The linear formula becomes:

y = b + w1 x1 + w2 x2 + w3 x3

Although x3 encodes nonlinear information, we do not need to change how the linear model trains to determine the value of w3.

Thanks to stochastic gradient descent, linear models can be trained efficiently. 

Thus, supplementing scaled linear models with feature crosses has traditionally been an efficient way to train on massive-scale data sets.

### Crossing One-Hot Vectors

In practice, machine learning models seldom cross continuous features. 

However, machine learning models do frequently cross one-hot feature vectors. 

Think of feature crosses of one-hot feature vectors as logical conjunctions. 

For example, suppose we have two features: country and language. 

A one-hot encoding of each generates vectors with binary features that can be interpreted as country=USA, country=France, language=English, or language=Spanish. 

Then, if you do a feature cross of these one-hot encodings, you get binary features that can be interpreted as logical conjunctions:

```
  country:usa AND language:spanish
```

As another example, suppose you bin latitude and longitude, producing separate one-hot five-element feature vectors.

For example, a given latitude and longitude could be represented as follows:

```
  binned_latitude = [0, 0, 0, 1, 0]
  binned_longitude = [0, 1, 0, 0, 0]
```

Suppose you create a feature cross of these two feature vectors:

```
  binned_latitude X binned_longitude
```

This feature cross is a 25-element one-hot vector (24 zeroes and 1 one). 

The single 1 in the cross identifies a particular conjunction of latitude and longitude. 

Your model can then learn particular associations about that conjunction.

Creating a feature cross of those coarse bins leads to synthetic feature having the following meanings:

Now suppose our model needs to predict how satisfied dog owners will be with dogs based on two features:

- Behavior type (barking, crying, snuggling, etc.) 
- Time of day


If we build a feature cross from both these features:

```
  [behavior type X time of day]
```
  
Then we end up with vastly more predictive ability than either feature on its own. 

For example, if a dog cries (happily) at 5:00 pm when the owner returns from work will likely be a great positive predictor of owner satisfaction. 

Crying (miserably, perhaps) at 3:00 am when the owner was sleeping soundly will likely be a strong negative predictor of owner satisfaction.

Linear learners scale well to massive data. 

Using feature crosses on massive data sets is one efficient strategy for learning highly complex models. 

Neural networks provide another strategy.

### Check Your Understanding

Q: Different cities in California have markedly different housing prices. Suppose you must create a model to predict housing prices. Which of the following sets of features or feature crosses could learn city-specific relationships between roomsPerPerson and housing price?

A: One feature cross: [binned latitude X binned longitude X binned roomsPerPerson]

Crossing binned latitude with binned longitude enables the model to learn city-specific effects of roomsPerPerson. 

Binning prevents a change in latitude producing the same result as a change in longitude. 

Depending on the granularity of the bins, this feature cross could learn city-specific or neighborhood-specific or even block-specific effects.


## Regularization for Simplicity: L₂ Regularization

Consider the following generalization curve which shows the loss for both the training set and validation set against the number of training iterations.

Figure 1. Loss on training set and validation set.

Figure 1 shows a model in which training loss gradually decreases, but validation loss eventually goes up. 

In other words, this generalization curve shows that the model is overfitting to the data in the training set. 

Channeling our inner Ockham, perhaps we could prevent overfitting by penalizing complex models, a principle called **regularization**.

In other words, instead of simply aiming to minimize loss (empirical risk minimization):

we will minimize loss+complexity which is called **structural risk minimization**:

Our training optimization algorithm is now a function of two terms: the _loss term_ which measures how well the model fits the data, and the _regularization term_ which measures model complexity.

The Machine Learning Crash Course focuses on two common (and somewhat related) ways to think of model complexity:

- Model complexity as a function of the _weights_ of all the features in the model.

- Model complexity as a function of the total _number of features_ with nonzero weights. 

If model complexity is a function of weights, a feature weight with a high absolute value is more complex than a feature weight with a low absolute value.

We can quantify complexity using the L2 regularization formula which defines the regularization term as the sum of the squares of all the feature weights:

In this formula, weights close to zero have little effect on model complexity while outlier weights can have a huge impact.

### Regularization for Simplicity: Lambda

Model developers tune the overall impact of the regularization term by multiplying its value by a scalar known as **lambda** (the regularization rate). 

Thus, model developers aim to do the following:

Performing L2 regularization has the following effect on a model:

- Encourages weight values toward 0 (but not exactly 0)

- Encourages the mean of the weights toward 0 with a normal (bell-shaped or Gaussian) distribution. 

Increasing the lambda value strengthens the regularization effect. 

For example, the histogram of weights for a high value of lambda might look as shown in Figure 2.

Lowering the value of lambda tends to yield a flatter histogram, as shown in Figure 3.

When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:

- If your lambda value is too high, your model will be simple, but you run the risk of _underfitting_ your data. 

  Your model will not learn enough about the training data to make useful predictions.

- If your lambda value is too low, your model will be more complex, and you run the risk of _overfitting_ your data. 

  Your model will learn too much about the particularities of the training data and will to be able to generalize to new data.

NOTE: Setting lambda to zero removes regularization completely. In this case, training focuses exclusively on minimizing loss which poses the highest possible overfitting risk.

The ideal value of lambda produces a model that generalizes well to new, previously unseen data. Unfortunately, that ideal value of lambda is data-dependent, so you will need to do some tuning.

### More about L2 regularization

There is a close connection between learning rate and lambda. 

Strong L2 regularization values tend to drive feature weights closer to 0. 

Lower learning rates (with early stopping) often produce the same effect because the steps away from 0 are not as large. 

This, tweaking learning rate and lambda simultaneously may have confounding effects.

**Early stopping** means ending training before the model fully reaches convergence. 

In practice, we often end up with some amount of implicit early stopping when training in an online (continuous) fashion. Thus, some new trends just have not had enough data yet to converge.

As noted, the effects from changes to regularization parameters can be confounded with the effects from changes in learning rate or number of iterations. 

One useful practice (when training across a fixed batch of data) is to give yourself a high enough number of iterations so that early stopping does not play into things.

### Check Your Understanding

L2 Regularization

Imagine a linear model with 100 input features:

- 10 are highly informative.
- 90 are non-informative.

Assume that all features have values between -1 and 1. 

Which of the following statements are true?

Answer:

1. L2 regularization will encourage many of the non-informative weights to be nearly (but not exactly) 0.0.

Yes, L2 regularization encourages weights to be near 0.0 but not exactly 0.0.

2. L2 regularization may cause the model to learn a moderate weight for some non-informative features.

Surprisingly, this can happen when a non-informative feature happens to be correlated with the label. 

In this case, the model incorrectly gives such non-informative features some of the "credit" that should have gone to informative features.

L2 Regularization and Correlated Features


Imagine a linear model with two strongly correlated features. Thus, these two features are nearly identical copies of one another but one feature contains a small amount of random noise. 

If we train this model with L2 regularization, what will happen to the weights for these two features?

A: Both features will have roughly equal, moderate weights.

L2 regularization will force the features towards roughly equivalent weights that are approximately half of what they would have been had only one of the two features been in the model.


## Logistic Regression

Instead of predicting exactly 0 or 1, logistic regression generates a probability — a value between 0 and 1, exclusive. 

For example, consider a logistic regression model for spam detection. 

If the model infers a value of 0.932 on a particular email message, it implies a 93.2% probability that the email message is spam. 

More precisely, it means that in the limit of infinite training examples, the set of examples for which the model predicts 0.932 will actually be spam 93.2% of the time and the remaining 6.8% will not.


### Calculating a Probability

Many problems require a probability estimate as output. 

Logistic regression is an extremely efficient mechanism for calculating probabilities. 

Practically speaking, you can use the returned probability in either of the following two ways:

- As is
- Converted to a binary category.

Let us consider how we might use the probability "as is." 

Suppose we create a logistic regression model to predict the probability that a dog will bark during the middle of the night. 

We will call that probability: p(bark | night)

If the logistic regression model predicts  p(bark | night) = 0.05 then over a year, the dog's owners should be startled awake approximately 18 times: 0.05 * 365 = 18. 
 

In many cases, you wim map the logistic regression output into the solution to a binary classification problem in which the goal is to correctly predict one of two possible labels (such as "spam" or "not spam").

You might be wondering how a logistic regression model can ensure output that always falls between 0 and 1. 

A sigmoid function defined as follows, produces output having those same characteristics:

The sigmoid function yields the following plot:

If z represents the output of the linear layer of a model trained with logistic regression, then sigmoid(z) will yield a value (a probability) between 0 and 1. 

In mathematical terms:

Note that z is also referred to as the _log-odds_ because the inverse of the sigmoid states that z can be defined as the log of the probability of the 1 label ("dog barks") divided by the probability of the label 0 ("dog does not bark"):

Here is the sigmoid function with ML labels:
 
### Sample logistic regression inference calculation


## Loss and Regularization

### Loss function for Logistic Regression

The loss function for linear regression is squared loss. 

The loss function for logistic regression is **Log Loss** which is defined as follows:

where:

- (x, y) in D is the data set containing many labeled examples which are (x, y) pairs. 

- y is the label in a labeled example. Since this is logistic regression, every value of 
must either be 0 or 1.

- y' is the predicted value (somewhere between 0 and 1), given the set of features in x.

### Regularization in Logistic Regression

Regularization is extremely important in logistic regression modeling. 

Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions. 

Thus, most logistic regression models use one of the following two strategies to dampen model complexity:

- L2 regularization.
- Early stopping, that is, limiting the number of training steps or the learning rate.

We will discuss a third strategy (L1 regularization) in a later module.

Imagine that you assign a unique id to each example and map each id to its own feature. 

If you do not specify a regularization function, the model will become completely overfit. 

This is because the model would try to drive loss to zero on all examples and never get there, driving the weights for each indicator feature to +infinity or -infinity. 

This can happen in high dimensional data with feature crosses when there is a huge mass of rare crosses that happen only on one example each.

Fortunately, using L2 or early stopping will prevent this problem.


## Classification

This module shows how logistic regression can be used for classification tasks, and explores how to evaluate the effectiveness of classification models.

### Thresholding

Logistic regression returns a probability. 

You can use the returned probability "as is" (the probability that the user will click on this ad is 0.00023) or convert the returned probability to a binary value (this email is spam).

A logistic regression model that returns 0.9995 for a particular email message is predicting that it is very likely to be spam. 

Conversely, another email message with a prediction score of 0.0003 on that same logistic regression model is very likely not spam. 

What about an email message with a prediction score of 0.6? 

In order to map a logistic regression value to a binary category, you must define a **classification threshold** (**decision threshold**). 

A value above that threshold indicates "spam"; a value below indicates "not spam." 

It is tempting to assume that the classification threshold should always be 0.5 but thresholds are problem-dependent, so they values that you must tune.

The following sections take a closer look at metrics that you can use to evaluate a classification model's predictions, as well as the impact of changing the classification threshold on these predictions.

NOTE: "Tuning" a threshold for logistic regression is different from tuning hyperparameters such as learning rate. 

Part of choosing a threshold is assessing how much your model will suffer for making a mistake. 

For example, mistakenly labeling a non-spam message as spam is very bad. However, mistakenly labeling a spam message as non-spam is unpleasant but hardly the end of your job.

### True vs. False and Positive vs. Negative

In this section, we define the primary building blocks of the metrics we will use to evaluate classification models. 

Let us make the following definitions:

- "Wolf" is a positive class.
- "No wolf" is a negative class.

We can summarize our "wolf-prediction" model using a 2x2 confusion matrix that depicts all four possible outcomes:

A **true positive** is an outcome where the model correctly predicts the positive class. 

A **true negative** is an outcome where the model correctly predicts the negative class.

A **false positive** is an outcome where the model incorrectly predicts the positive class.

A **false negative** is an outcome where the model incorrectly predicts the negative class.

In the following sections, we look at how to evaluate classification models using metrics derived from these four outcomes.

### Accuracy

Accuracy is one metric for evaluating classification models. 

Informally, **accuracy** is the fraction of predictions our model got right. 

Formally, accuracy has the following definition:

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

wherw TP = True Positives, TN = True Negatives, FP = False Positives, and FN = False Negatives.

Let us try calculating accuracy for the following model that classified 100 tumors as malignant (the positive class) or benign (the negative class):

Accuracy comes out to 0.91, or 91% (91 correct predictions out of 100 total examples) which means our tumor classifier is doing a great job of identifying malignancies, right?

Actually, let us do a closer analysis of positives and negatives to gain more insight into our model's performance.

While 91% accuracy may seem good at first glance, another tumor-classifier model that always predicts benign would achieve the exact same accuracy (91/100 correct predictions) on our examples. Thus, our model is no better than one that has zero predictive ability to distinguish malignant tumors from benign tumors.

Accuracy alone does not tell the full story when you are working with a **class-imbalanced data set** such as this one where there is a significant disparity between the number of positive and negative labels.

In the next section, we look at two better metrics for evaluating class-imbalanced problems: precision and recall.

### Precision and Recall

#### Precision

Precision attempts to answer the following question:

What proportion of positive identifications was actually correct?

NOTE: A model that produces no false positives has a precision of 1.0.

Let us calculate precision for our ML model from the previous section that analyzes tumors:

True Positives (TPs): 1	
False Positives (FPs): 1
False Negatives (FNs): 8	
True Negatives (TNs): 90

Our model has a precision of 0.5—in other words, when it predicts a tumor is malignant, it is correct 50% of the time.

#### Recall

Recall: What proportion of actual positives was identified correctly?

NOTE: A model that produces no false negatives has a recall of 1.0.

Let us calculate recall for our tumor classifier:

Our model has a recall of 0.11 which means it correctly identifies 11% of all malignant tumors.

### Precision and Recall: A Tug of War

To fully evaluate the effectiveness of a model, you must examine _both_ precision and recall. 

Unfortunately, precision and recall are often in tension. 

Thus, improving precision typically reduces recall and vice versa. 

Explore this notion by looking at the following figure which shows 30 predictions made by an email classification model. 

Those to the right of the classification threshold are classified as "spam" while those to the left are classified as "not spam."

Figure 1. Classifying email messages as spam or not spam.

Let us calculate precision and recall based on the results shown in Figure 1:

True Positives (TP): 8	
False Positives (FP): 2
False Negatives (FN): 3	
True Negatives (TN): 17

Precision measures the percentage of emails flagged as spam that were correctly classified — the percentage of dots to the right of the threshold line that are green in Figure 1:

Recall measures the percentage of actual spam emails that were correctly classified - the percentage of green dots that are to the right of the threshold line in Figure 1:

Figure 2. Increasing classification threshold.

The number of false positives decreases but false negatives increase. As a result, precision increases while recall decreases:

Figure 3 illustrates the effect of decreasing the classification threshold (from its original position in Figure 1).

Figure 3. Decreasing classification threshold.

False positives increase and false negatives decrease. As a result, this time, precision decreases and recall increases:

True Positives (TP): 9	False Positives (FP): 3
False Negatives (FN): 2	True Negatives (TN): 16

Various metrics have been developed that rely on both precision and recall such as F1 score.

### Check Your Understanding

#### Accuracy

Q: In which of the following scenarios would a high accuracy value suggest that the ML model is doing a good job?

A: In the game of roulette, a ball is dropped on a spinning wheel and eventually lands in one of 38 slots. Using visual features (the spin of the ball, the position of the wheel when the ball was dropped, the height of the ball over the wheel), an ML model can predict the slot that the ball will land in with an accuracy of 4%.

This ML model is making predictions far better than chance; a random guess would be correct 1/38 of the time—yielding an accuracy of 2.6%. Although the model's accuracy is "only" 4%, the benefits of success far outweigh the disadvantages of failure.

#### Precision

Q: Consider a classification model that separates email into two categories: "spam" or "not spam." If you raise the classification threshold, what will happen to precision?

A: Probably increase.

In general, raising the classification threshold reduces false positives, thus raising precision.

#### Recall

Q: Consider a classification model that separates email into two categories: "spam" or "not spam." If you raise the classification threshold, what will happen to recall?

A: Always decrease or stay the same.

Raising our classification threshold will cause the number of true positives to decrease or stay the same and will cause the number of false negatives to increase or stay the same. Thus, recall will either stay constant or decrease.

#### Precision and Recall

Q: Consider two models (A and B) that each evaluate the same dataset. Which one of the following statements is true?

A: If model A has better precision and better recall than model B, then model A is probably better.

In general, a model that outperforms another model on both precision and recall is likely the better model. 

Obviously, we will need to make sure that comparison is being done at a precision / recall point that is useful in practice for this to be meaningful. 

For example, suppose our spam detection model needs to have at least 90% precision to be useful and avoid unnecessary false alarms. In this case, comparing one model at {20% precision, 99% recall} to another at {15% precision, 98% recall} is not particularly instructive since neither model meets the 90% precision requirement. 

However, this is still a good way to think about comparing models when using precision and recall.


## ROC Curve and AUC

### ROC curve

An **ROC curve (receiver operating characteristic curve)** is a graph showing the performance of a classification model at all classification thresholds. 

This curve plots two parameters:

- True Positive Rate
- False Positive Rate

True Positive Rate (TPR) is a synonym for recall which defined as follows:


False Positive Rate (FPR) is defined as follows:

An ROC curve plots TPR vs. FPR at different classification thresholds. 

Lowering the classification threshold classifies more items as positive which increases both False Positives and True Positives. 

The following figure shows a typical ROC curve.

Figure 4. TP vs. FP rate at different classification thresholds.

To compute the points in an ROC curve, we could evaluate a logistic regression model many times with different classification thresholds, but this would be inefficient. 

Fortunately, there is an efficient, sorting-based algorithm called AUC.

### AUC: Area Under the ROC Curve

AUC stands for "Area under the ROC Curve." 

AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).

Figure 5. AUC (Area under the ROC Curve).

AUC provides an aggregate measure of performance across all possible classification thresholds. 

One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. 

For example, given the following examples which are arranged from left to right in ascending order of logistic regression predictions:

Figure 6. Predictions ranked in ascending order of logistic regression score.

AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.

AUC ranges in value from 0 to 1. 

A model whose predictions are 100% wrong has an AUC of 0.0 while a model whose predictions are 100% correct has an AUC of 1.0.

AUC is desirable for the following two reasons:

- AUC is scale-invariant. 

  AUC  measures how well predictions are ranked rather than their absolute values.
  
- AUC is classification-threshold-invariant. 

  AUC measures the quality of the model's predictions irrespective of what classification threshold is chosen.

However, both these reasons come with caveats which may limit the usefulness of AUC in certain use cases:

Scale invariance is not always desirable. 

For example, sometimes we really do need well calibrated probability outputs and AUC will not tell us about that.

Classification-threshold invariance is not always desirable. 

In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. 

For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). Thus, AUC is not a useful metric for this type of optimization.

### Check Your Understanding

#### ROC and AUC

Which of the following ROC curves produce AUC values greater than 0.5?

NOTE: In practice, if you have a "perfect" classifier with an AUC of 1.0, you should be suspicious, as it likely indicates a bug in your model. For example, you may have overfit to your training data, or the label data may be replicated in one of your features.

#### AUC and Scaling Predictions

How would multiplying all of the predictions from a given model by 2.0 (for example, if the model predicts 0.4, we multiply by 2.0 to get a prediction of 0.8) change the model's performance as measured by AUC?

No change. AUC only cares about relative prediction scores.

Yes, AUC is based on the relative predictions, so any transformation of the predictions that preserves the relative ranking has no effect on AUC. 

However, this is clearly not the case for other metrics such as squared error, log loss, or prediction bias (discussed later).


## Prediction Bias

Logistic regression predictions should be unbiased.

"average of predictions" should ≈ "average of observations"

Prediction bias is a quantity that measures how far apart those two averages are:

prediction bias = average of prefictions + average of labels


NOTE: "Prediction bias" is a different quantity than bias (the b in wx + b).

A significant nonzero prediction bias tells you there is a bug somewhere in the model - it indicates that the model is wrong about how frequently positive labels occur.

For example, suppose we know that on average, 1% of all emails are spam. If we do not know anything at all about a given email, we should predict that it is 1% likely to be spam. 

Similarly, a good spam model should predict on average that emails are 1% likely to be spam. This, if we average the predicted likelihoods of each individual email being spam, the result should be 1%. 

If instead, the model's average prediction is 20% likelihood of being spam, we can conclude that it exhibits prediction bias.

Possible root causes of prediction bias are:

- Incomplete feature set
- Noisy data set
- Buggy pipeline
- Biased training sample
- Overly strong regularization

You might be tempted to correct prediction bias by post-processing the learned model by adding a calibration layer that adjusts your model's output to reduce the prediction bias. 

For example, if your model has +3% bias, you could add a calibration layer that lowers the mean prediction by 3%. However, adding a calibration layer is a bad idea for the following reasons:

- You are fixing the symptom rather than the cause.

- You have built a more brittle system that you must now keep up to date.

If possible, avoid calibration layers. Projects that use calibration layers tend to become reliant on them — using calibration layers to fix all their model's sins. Ultimately, maintaining the calibration layers can become a nightmare.

NOTE: A good model will usually have near-zero bias. However, a low prediction bias does not prove that your model is good. A really terrible model could have a zero prediction bias. 

For example, a model that just predicts the mean value for all examples would be a bad model, despite having zero bias.

### Bucketing and Prediction Bias

Logistic regression predicts a value _between_ 0 and 1. However, all labeled examples are either exactly 0 ("not spam") or exactly 1 ("spam"). 

Therefore, when examining prediction bias, you cannot accurately determine the prediction bias based on only one example - you must examine the prediction bias on a "bucket" of examples. 

Thus, prediction bias for logistic regression only makes sense when grouping enough examples together to be able to compare a predicted value (for example, 0.392) to observed values (for example, 0.394).

You can form buckets in the following ways:

- Linearly breaking up the target predictions. 

- Forming quantiles.

Consider the following calibration plot from a particular model where each dot represents a bucket of 1,000 values. 

The axes have the following meanings:

- The x-axis represents the average of values the model predicted for that bucket.

- The y-axis represents the actual average of values in the data set for that bucket.

Both axes are logarithmic scales.

Figure 8. Prediction bias curve (logarithmic scales)

Why are the predictions so poor for only part of the model? 

Here are a few possibilities:

- The training set does not adequately represent certain subsets of the data space.

- Some subsets of the data set are noisier than others.

- The model is overly regularized. (Consider reducing the value of lambda.)


## References

[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)
 