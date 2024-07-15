# Data Science Primer

## 1 Bird's Eye View

### Key Terminology

This data science primer will cover exploratory analysis, data cleaning, feature engineering, algorithm selection, and model training.

We will focus on developing practical intuition instead of diving into technicalities (which we save for later).

Therefore, it is important to be clear and concise with our terminology.

Let us make sure we have a shared language for discussing these topics:

  - **Model:** a set of patterns learned from data.

  - **Algorithm:** a specific ML process used to train a model.

  - **Training data:** the dataset from which the algorithm learns the model.

  - **Test data:** a new dataset for evaluating model performance.

  - **Features:** the variables (columns) in the dataset used to train the model.

  - **Target variable:** a specific variable we are trying to predict.

  - **Observations:** the data points (rows) in the dataset.

Suppose we have a dataset of 150 primary school students, and we want to predict their Height based on their Age, Gender, and Weight:

  - You have 150 observations.
  - 3 features (Age, Gender, Weight).
  - 1 target variable (Height).

  - You might then separate your dataset into two subsets:

    1. Set of 120 used to train several models (training set)
    2. Set of 30 used to pick the best model (test set)

We will explain why separate training and test sets are important during **Model Training**.


### Machine Learning Tasks

In applied machine learning, you should pick the right machine learning task for the job.

- A **task** is a specific objective for your algorithms.

- Algorithms can be swapped in and out as long as you pick the right task.

- You should **always try multiple algorithms** because you most likely will not know which one will perform best for your dataset.

The two most common categories of tasks are _supervised learning_ and _unsupervised learning._

#### Supervised Learning

Supervised learning includes tasks for _labeled_ data (you have a target variable).

- In practice, it is often used as an advanced form of _predictive modeling_.

- Each observation must be labeled with a _correct answer_.

- Then you build a predictive model by telling the algorithm what is "correct" while training (supervising) it.

- **Regression** is the task for modeling _continuous_ target variables.

- **Classification** is the task for modeling _categorical_ (class) target variables.


- Multinomial Logistic Regression

  Multinomial Logistic Regression is a classification algorithm used to do multiclass classification.

  sklearn: We can fit a multinomial logistic regression with L1 penalty on a subset of the MNIST digits classification task.


#### Unsupervised Learning

Unsupervised learning includes tasks for _unlabeled_ data (you do not have a target variable).

- In practice, it is often used either as a form of _automated data analysis_ or _automated signal extraction_.

- Unlabeled data have no predetermined correct answer.

- You allow the algorithm to learn patterns from the data (without supervision).

- **Clustering** is the most common unsupervised learning task for finding groups within your data.


### The Blueprint

The first essential element is the quality of your data.

- Garbage In = Garbage Out, no matter which algorithm(s) you use.

One of the pitfalls in machine learning is _overfitting_.

An overfit model has _memorized_ the noise in the training set rather than learning the true underlying patterns.

- For most applications, the stakes will not be that high, but overfitting is still the single largest mistake you must avoid.

- We will teach you strategies for preventing overfitting by 1) choosing the right algorithms and 2) tuning them correctly.

Our machine learning blueprint is designed around those two elements.

#### There are five core steps

  1. Exploratory Analysis: get to know the data. This step should be quick, efficient, and decisive.

  2. Data Cleaning: clean your data to avoid many common pitfalls. Better data beats fancier algorithms.

  3. Feature Engineering: help your algorithms "focus" on what is important by creating new features.

  4. Algorithm Selection: choose the best, most appropriate algorithm(s).

  5. Model Training: train your models. This step is pretty formulaic after you have done the first 4.

There are other steps as well:

  - S: Project Scoping - Sometimes you will need to roadmap the project and anticipate data needs.

  - W: Data Wrangling - You may need to restructure your dataset into a format that algorithms can handle.

  - P: Preprocessing - transforming your features first can further improve performance.

  - E: Ensembling - You can squeeze out even more performance by combining multiple models.


## Encoding and Embedding

All Machine Learning (ML) methods work with input feature vectors and almost all of them require input features to be numerical [2].

All Machine Learning (ML) methods work with input feature vectors and almost all of them require input features to be numerical.

There are four types of ML features [2]:

1. Numerical (continuous or discrete): numerical data can be characterized by continuous or discrete data.

Continuous data can assume any value within a range whereas discrete data has distinct values.

- An example of continues numerical variable is `height`.

- An example of discrete numerical variable is `age`.

2. Categorical (ordinal or nominal): categorical data represents characteristics such as eye color, and hometown.

  Categorical data can be ordinal or nominal.

  In ordinal variable, the data falls into ordered categories that are ranked in some particular way.

  An example is `skill level` that takes values of [`beginner`, `intermediate`, `advanced`].

  A nominal variable has no order among its values.

  An example is `eye color` that takes values of [`black`, `brown’, `blue`, `green`].

3. Time series: Time series is a sequence of numbers collected at regular intervals over some period of time.

- This data is ordered in time unlike previous variables.

- An example of this is `average of home sale price over years in USA`.

4. Text: Any document is a text data, that we often represent them as a ‘bag of words’.

  To feed any variables to an ML model, we have to convert them into numerical. Both encoding and embedding techniques can ne used to make the conversion.

### Encoding

Encoding is the process of converting raw data, such as text, images, or audio, into a structured numerical format that can be easily processed by computers [2].

There are two ways to encode a categorical variable [2]:

1. Integer encoding

2. One-hot encoding

3. Multi-hot encoding (an extension of one-hot encoding)

The conclusion is that one-hot and multi-hot encodings are not practical for features with large value sets [2].

### Embedding

To address the shortcomings mentioned, we can translate high dimensional sparse vector to short dense vectors called _embeddings_.

An embedding is a translation of a high-dimensional vector into a low-dimensional space and captures semantic similarity.

#### Singular Value Decomposition (SVD)

The simplest embedding method is perhaps Singular Value Decomposition (SVD) that takes an input matrix A and decompose it into three matrices as shown below:

#### Neural Networks as Embedder

State of the art embedders are among Neural Networks (NN).

There are many NN techniques to compute word embeddings: Word2Vec, Glove, BERT, fastText, etc.



## 2 Exploratory Analysis

The purpose of exploratory analysis is to _get to know_ the dataset.

Doing so upfront will make the rest of the project much smoother, in three main ways:

  1. You will gain valuable hints for Data Cleaning (which can make or break your models).

  2. You will think of ideas for Feature Engineering (which can take your models from good to great).

  3. You will get a "feel" for the dataset (which will help you communicate results and deliver greater impact).

### Start with Basics

First, you want to answer a set of basic questions about the dataset:

- How many observations do I have?
- How many features?
- What are the data types of my features? Are they numeric? Categorical?
- Do I have a target variable?

The purpose of displaying examples from the dataset is to get a **qualitative** "feel" for the dataset.

- Do the columns make sense?
- Do the values in those columns make sense?
- Are the values on the right scale?
- Is missing data going to be a big problem based on a quick eyeball test?

### Plot Numerical Distributions

Next, it can be helpful to plot the distributions of your numeric features.

Often, a quick and dirty grid of histograms is enough to understand the distributions.

Here are a few things to look out for:

- Distributions that are unexpected
- Potential outliers that do not make sense
- Features that should be binary (wannabe indicator variables)
- Boundaries that do not make sense
- Potential measurement errors

At this point, you should start making notes about potential fixes you would like to make. If something looks out of place (such as a potential outlier in one of your features) now is a good time to ask the client/key stakeholder or to dig a bit deeper.

However, we will wait until **Data Cleaning** to make fixes so that we can keep our steps organized.

### Plot Categorical Distributions

Categorical features cannot be visualized through histograms. Instead, you can use **bar plots**.

In particular, you want to look out for **sparse classes** which are classes that have a very small number of observations.

By the way, a **class** is simply a unique value for a categorical feature.

Example: The following bar plot shows the distribution for a feature called 'exterior_walls', so "Wood Siding", "Brick", and "Stucco" are each classes for that feature.

As you can see, some of the classes for 'exterior_walls' have very short bars (sparse classes).

They tend to be problematic when building models.

- In the best case, they do not influence the model much.
- In the worse case, they can cause the model to be **overfit**.

Therefore, we recommend making a note to combine or reassign some of these classes later. We prefer saving this until Feature Engineering (Lesson 4).

### Plot Segmentations

Segmentations are a powerful way to observe the _relationship between categorical features and numeric features_.

*Box plots* allow you to observe the segmentations.

Here are a few insights you could draw from the following chart:

- The median transaction price (middle vertical bar in the box) for Single-Family homes was much higher than that for Apartments / Condos / Townhomes.

- The min and max transaction prices are comparable between the two classes.

- The round-number min ($200k) and max ($800k) suggest possible data truncation which is very important to remember when assessing the **generalizability** of your models later!

### Study Correlations

Finally, correlations allow you to look at the relationships _between numeric features_.

A _correlation_ is a value between -1 and 1 that represents how closely two features move in unison. You do not need to remember the math to calculate them, just know the intuition:

- **Positive** correlation means that as one feature increases, the other increases. Example: a child’s age and her height.

- **Negative** correlation means that as one feature increases, the other decreases. Example: hours spent studying and number of parties attended.

- Correlations near -1 or 1 indicate a **strong relationship**.

- Those closer to 0 indicate a **weak relationship**.

- 0 indicates **no relationship**.

Correlation **heatmaps** help you visualize this information.

Here is an example (all correlations were multiplied by 100):


In general, you should look out for:

- Which features are strongly correlated with the target variable?
- Are there interesting or unexpected strong correlations between other features?

Again, your aim is to gain intuition about the data which will help you later.

By the end of your Exploratory Analysis step, you will have a better understanding of the dataset, some notes for data cleaning, and possibly some ideas for feature engineering.



## 3 Data Cleaning

Better data beats fancier algorithms ...

In the previous overview, you learned about essential data visualizations for "getting to know" the data where we explained the types of insights to look for.

Based on those insights, it is time to get our dataset into tip-top shape through **data cleaning**.

The steps and techniques for data cleaning will vary from dataset to dataset. As a result, it is impossible for a single guide to cover everything you might run into.

However, this guide provides a **reliable starting framework** that can be used every time.

We cover the most common steps such as fixing structural errors, handling missing data, and filtering observations.

### Remove Unwanted observations

The first step to data cleaning is removing unwanted observations from your dataset.

This includes **duplicate** or **irrelevant** observations.

#### Duplicate observations

Duplicate observations most frequently arise during **data collection** such as when you:

- Combine datasets from multiple places
- Scrape data
- Receive data from clients/other departments

#### Irrelevant observations

Irrelevant observations are those that do not actually fit the **specific problem** that you are trying to solve.

  - If you were building a model for Single-Family homes only, you would not want observations for Apartments in there.

  - This is also a great time to review your charts from Exploratory Analysis.
    You can look at the distribution charts for categorical features to see if there are any classes that should not be there.

  - Checking for irrelevant observations before engineering features can save you many headaches down the road.

### Fix Structural Errors

The next step involves fixing _structural errors_ which are errors that arise during measurement, data transfer, or other types of "poor housekeeping".

Example: You can check for **typos** or **inconsistent capitalization**. This is mostly a concern for categorical features and you can use bar plots to check.

Here is an example:

As you can see:

  - 'composition' is the same as 'Composition'

  - 'asphalt' should be 'Asphalt'

  - 'shake-shingle' should be 'Shake Shingle'

  - 'asphalt,shake-shingle' could probably just be 'Shake Shingle' as well

After we replace the typos and inconsistent capitalization, the class distribution becomes much cleaner:

Finally, check for **mislabeled classes** which are separate classes that should really be the same.

Examples:

  - If ’N/A’ and ’Not Applicable’ appear as two separate classes, you should combine them.

  - ’IT’ and ’information_technology’ should be a single class.

### Filter Unwanted Outliers

Outliers can cause problems with certain types of models.

For example, linear regression models are less robust to outliers than decision tree models.

In general, if you have a **legitimate** reason to remove an outlier, it will help your model’s performance.

However, outliers are **innocent until proven guilty**.

You should never remove an outlier just because it is a "big number." That big number could be very informative for your model.

**We cannot stress this enough:** you must have a good reason for removing an outlier such as suspicious measurements that are unlikely to be real data.

### Handle Missing Data

Missing data is a deceptively tricky issue in applied machine learning.

First, **you cannot simply ignore missing values in your dataset**. You must handle them in some way for the very practical reason that most algorithms do not accept missing values.

There are two commonly used methods of dealing with missing data:

  1. **Dropping** observations that have missing values

  2. **Imputing** the missing values based on other observations

Dropping missing values is sub-optimal because when you drop observations, you **drop information**.

- The fact that the value was missing may be informative in itself.

- In the real world, you often need to make predictions on new data even if some of the features are missing.

Imputing missing values is sub-optimal because the value was originally missing but you filled it in, which always leads to a loss in information, no matter how sophisticated your imputation method is.

- **Missingness** is almost always informative in itself so you should **tell your algorithm** if a value was missing.

- Even if you build a model to impute your values, you are not adding any real information. You are just reinforcing the patterns already provided by other features.

In short, you should always tell your algorithm that a value was missing because **missingness is informative**.

#### Missing categorical data

The best way to handle missing data for _categorical_ features is to simply label them as "Missing".

- You are essentially adding a new class for the feature.
- This tells the algorithm that the value was missing.
- This also gets around the technical requirement for no missing values.

#### Missing numeric data

For missing _numeric_ data, you should **flag and fill** the values.

  1. Flag the observation with an indicator variable of missingness.
  2. Fill the original missing value with 0 just to meet the technical requirement of no missing values.

By using this technique of flagging and filling, you are essentially **allowing the algorithm to estimate the optimal constant for missingness** rather than just filling it in with the mean.



## 4 Feature Engineering

Applied machine learning is basically feature engineering.

In the previous chapter, you learned a reliable framework for cleaning your dataset. We fixed structural errors, handled missing data, and filtered observations.

In this guide, we see how we can perform **feature engineering** to help out our algorithms and improve model performance.

Remember, data scientists usually spend the most time on feature engineering.

This is a skill that you can develop with time and practice, but **heuristics** can help you know where to start looking, spark ideas, and get unstuck.

### What is Feature Engineering?

Feature engineering is about **creating new input features** from existing ones.

In general, you can think of data cleaning as a process of _subtraction_ and feature engineering as a process of _addition_.

This is often one of the most valuable tasks a data scientist can do to improve model performance:

  1. You can isolate and highlight key information which helps your algorithms "focus" on what is important.

  2. You can bring in your own domain expertise.

  3. Once you understand the "vocabulary" of feature engineering, you can bring in other people’s domain expertise!

In this lesson, we will introduce several _heuristics_ to help spark new ideas.

Before moving on, we just want to note that this is not an exhaustive list of all feature engineering because there are limitless possibilities for this step.

The good news is that this skill will improve as you gain more experience.

### Infuse Domain Knowledge

You can often engineer informative features by tapping into domain expertise.

Try to think of specific information you might want to **isolate**. Here, you have a lot of creative freedom."

Going back to our example with the real-estate dataset, suppose you remembered that the housing crisis occurred in the same timeframe...

If you suspect that prices would be affected, you could create an **indicator variable** for transactions during that period.​

Indicator variables are binary variables that can be either 0 or 1.

They _indicate_ if an observation meets a certain condition which helps in isolating key properties.

As you might suspect, domain knowledge is very broad and open-ended. At some point, you will get stuck or exhaust your ideas.

These are a few specific heuristics that can help spark more ideas.

### Create Interaction Features

The first heuristic is checking to see if you can create any **interaction features** that make sense which are combinations of two or more features.

In some contexts, interaction terms must be products between two variables. Here, interaction features can be products, sums, or differences between two features.

In general, look at each pair of features and ask yourself, "could I combine this information in any way that might be more useful?"

Example (real-estate):

- Suppose we had a feature called 'num_schools' (the number of schools within 5 miles of a property).

- We also have the feature 'median_school' (the median quality score of those schools).

- We might suspect that what is really important is having many school options but only if they are good.

- To capture that interaction, we could simple create a new feature 'school_score' = 'num_schools' x 'median_school'

### Combine Sparse Classes

This heuristic involves grouping sparse classes.

**Sparse classes** (in categorical features) are those that have very few total observations.

They can be problematic for certain ML algorithms, causing models to be overfit.

- There is no formal rule of how many each class needs.

- It also depends on the size of your dataset and the number of other features you have.

- As a _rule of thumb_, we recommend combining classes until each one has at least `~50` observations. As with any rule of thumb, use this as a guideline rather than a rule.

Let us take a look at the real-estate example:

First, we can group **similar classes**.

In the chart above, the 'exterior_walls' feature has several classes that are quite similar.

- We might want to group 'Wood Siding', 'Wood Shingle', and 'Wood' into a single class called 'Wood'.

Next, we can group the remaining sparse classes into a single 'Other' class, even if there is already an 'Other' class.

We can group 'Concrete Block', 'Stucco', 'Masonry', 'Other', and 'Asbestos shingle' into 'Other'.

Here is how the class distributions look after combining similar and other classes:

After combining sparse classes, we have fewer unique classes but each one has more observations.

Often, an **eyeball test** is enough to decide if you want to group certain classes together.

### Add Dummy Variables

Most ML algorithms cannot directly handle categorical features that are _text values_.

Therefore, we need to create dummy variables for our categorical features which is called _one-hot encoding_.

A **dummy variable** is a binary (0 or 1) variable that represents a single class from a categorical feature.

The information we represent is exactly the same but this numeric representation allows you to pass the technical requirements for algorithms.

In the example above, we were left with 8 classes (after grouping sparse classes) which translate to 8 dummy variables:

### Remove Unused Features

We can remove unused or redundant features from the dataset.

Unused features are those that do not make sense to pass into our machine learning algorithms.

Examples include:

- ID columns

- Features that would not be available at the time of prediction

- Other text descriptions

**Redundant** features would typically be those that have been replaced by other features that you have added during feature engineering.

After completing Data Cleaning and Feature Engineering, you have transformed your raw dataset into an **analytical base table (ABT)**.

We call it an ABT because it is what you will use for building your models.

**Tip:** Not all of the features you engineer need to be winners. In fact, you will often find that many of them do not improve your model.

The key is choosing machine learning algorithms that can **automatically select the best features** among many options called **built-in feature selection**.

This will allow you to **avoid overfitting** your model despite providing many input features.



## 5 Algorithm Selection

In the previous overview, you learned several different heuristics for effective feature engineering including tapping into domain knowledge and grouping sparse classes.

This guide will explain **algorithm selection** for machine learning.

Rather than bombarding you with options, we are going to jump straight to best practices.

We will introduce two powerful mechanisms in modern algorithms: **regularization** and **ensembles**.

As you will see, these mechanisms "fix" some fatal flaws in older methods which has lead to their popularity.

### How to Pick ML Algorithms

In this lesson, we introduce five very effective machine learning algorithms for **regression** tasks which each have classification counterparts as well.

Instead of giving you a long list of algorithms, our goal is to explain a few essential concepts (regularization, ensembling, automatic feature selection) that will teach you why some algorithms tend to perform better than others.

In applied machine learning, individual algorithms should be swapped in and out depending on which performs best for the problem and the dataset. Therefore, we will focus on **intuition** and **practical benefits** over math and theory.

### Why Linear Regression is Flawed

To introduce the reasoning for some of the advanced algorithms, we start by discussing basic linear regression.

Linear regression models are very common but deeply flawed.

Simple linear regression models fit a "straight line" (technically a _hyperplane_ depending on the number of features).

In practice, they rarely perform well. We actually recommend skipping them for most machine learning problems.

Their main advantage is that they are easy to interpret and understand.

However, our goal is to build a model that can make accurate predictions (not to write a research report).

In this regard, simple linear regression suffers from **two major flaws**:

  1. It is prone to overfit with many input features.
  2. It cannot easily express non-linear relationships.

### Regularization in Machine Learning

This is the first "advanced" tactic for improving model performance.

It is considered pretty "advanced" in many ML courses, but it is really pretty easy to understand and implement.

The first flaw of linear models is that they are prone to overfit with many input features.

Let us take an extreme example to illustrate why this happens:

- Say you have 100 observations in your training dataset.

- Say you also have 100 features.

- If you fit a linear regression model with all of those 100 features, you can perfectly "memorize" the training set.

- Each **coefficient** would simply **memorize one observation**.
  This model would have perfect accuracy on the training data but perform poorly on unseen data.

- It has not learned the true underlying patterns; it has only _memorized the noise_ in the training data.

**Regularization** is a technique used to prevent overfitting by artificially penalizing model coefficients.

- It can discourage large coefficients (by dampening them).

- It can also remove features entirely (by setting their coefficients to 0).

- The strength of the penalty is **tunable**.


### Regularized Regression Algorithms

There are three common types of regularized linear regression algorithms.

#### Lasso Regression

Lasso or LASSO stands for Least Absolute Shrinkage and Selection Operator.

- Lasso regression penalizes the _absolute size_ of coefficients.

- Practically, this leads to coefficients that can be exactly 0.

- Thus, Lasso offers **automatic feature selection** because it can completely remove some features.

- Remember: the "strength" of the penalty should be tuned.

- A stronger penalty leads to _more_ coefficients pushed to zero.

#### Ridge Regression

Ridge stands Really Intense Dangerous Grapefruit Eating (just kidding... it's just ridge).

- Ridge regression penalizes the _squared size_ of coefficients.

- Practically, this leads to smaller coefficients, but it does not force them to 0.

- In other words, Ridge offers **feature shrinkage**.

- Remember: the "strength" of the penalty should be tuned.

- A stronger penalty leads to coefficients pushed closer to zero.

#### Elastic-Net

Elastic-Net is a compromise between Lasso and Ridge.

- Elastic-Net penalizes a _mix_ of both absolute and squared size.

- The _ratio_ of the two penalty types should be tuned.

- The overall strength should also be tuned.

There is no "best" type of penalty. It really depends on the dataset and the problem.

We recommend trying different algorithms that use a range of penalty strengths as part of the tuning process, which we will cover later.


### Decision Tree Algorithms

We have seen three algorithms that can protect linear regression from overfitting, but linear regression suffers from two main flaws:

  1. It is prone to overfit with many input features.
  2. It cannot easily express non-linear relationships.

How can we address the second flaw?

We need to move away from linear models: we need to bring in a new category of algorithms.

Decision trees model data as a "tree" of hierarchical branches.

They make branches until they reach "leaves" that represent predictions.

Due to their **branching structure**, decision trees can easily model **nonlinear** relationships.

- Suppose for Single Family homes, larger lots command higher prices.

- Suppose that for Apartments, smaller lots command higher prices (it is a proxy for urban/rural).

- This _reversal of correlation_ is difficult for linear models to capture unless you explicitly add an interaction term (you can anticipate it ahead of time).

- On the other hand, decision trees can capture this relationship naturally.

Unfortunately, decision trees suffer from a major flaw as well. If you allow them to grow limitlessly, they can completely "memorize" the training data, just from creating more and more and more branches.

**As a result, individual unconstrained decision trees are very prone to being overfit.**​

How can we take advantage of the flexibility of decision trees while preventing them from overfitting the training data?


### Tree Ensembles

Ensembles are machine learning methods for combining predictions from multiple separate models.

There are a few different methods for ensembling, but the two most common are:

#### Bagging

**Bagging** attempts to reduce the chance of overfitting _complex models_.

- It trains a large number of _strong learners_ in parallel.

- A **strong learner** is a model that is relatively unconstrained.

- Bagging combines all the strong learners together in order to "smooth out" their predictions.

#### Boosting

**Boosting** attempts to improve the predictive flexibility of _simple models_.

- It trains a large number of _weak learners_ in sequence.

- A **weak learner** is a constrained model (you could limit the max depth of each decision tree).

- Each one in the sequence focuses on learning from the mistakes of the one before it.

- Boosting combines all the weak learners into a single strong learner.

While bagging and boosting are both ensemble methods, they approach the problem from opposite directions:

- Bagging uses complex base models and tries to "smooth out" their predictions.
- Boosting uses simple base models and tries to "boost" their aggregate complexity.

Ensembling is a general term but winhen the **base models** are decision trees, they have special names: random forests and boosted trees!

#### Gradient Boosting

**Gradient boosting** is a machine learning technique for regression and classification problems which produces a prediction model in the form of an _ensemble_ of weak prediction models, typically decision trees.

When a decision tree is the weak learner, the resulting algorithm is called gradient boosted trees which usually outperforms random forest.

Gradient boosting builds the model in a stage-wise fashion like other boosting methods, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

#### Random forests

Random forests train a **large number of strong decision trees** and combine their predictions through bagging.

In addition, there are two sources of "randomness" for random forests:

  1. Each tree is only allowed to choose from a random subset of features to split on (leading to **feature selection**).

  2. Each tree is only trained on a random subset of observations (a process called **resampling**).

In practice, random forests tend to perform very well right out of the box.

- They often beat many other models that take up to weeks to develop.

- They are the perfect **swiss-army-knife** algorithm that almost always gets good results.

- They do not have many complicated parameters to tune.

#### Boosted trees

Boosted trees train a **sequence of weak, constrained decision trees** and combine their predictions through boosting.

- Each tree is allowed a **maximum depth** which should be tuned.

- Each tree in the sequence tries to correct the prediction errors of the one before it.

In practice, boosted trees tend to have the highest performance ceilings.

- They often beat many other types of models **after proper tuning**.

- They are more complicated to tune than random forests.

### Summary

**Key idea:** The most effective algorithms usually offer a combination of regularization, automatic feature selection, ability to express nonlinear relationships, and/or ensembling.

Those algorithms include:

- Lasso regression
- Ridge regression
- Elastic-Net
- Random forest
- Boosted tree



## 6 Model Training

In the previous overview, we introduced five effective ML algorithms that tap into the powerful mechanisms of regularization and ensembles.

In this guide, we will take you step-by-step through the **model training** process.

Since we have already done the hard part, actually fitting (training) our model will be fairly straightforward.

There are a few key techniques that we will discuss, and these have become widely-accepted **best practices** in the field.

Again, this mini-course is meant to be a gentle introduction to data science and machine learning, so we will not get into the nitty gritty yet.

### How to Train ML Models

Professional data scientists actually spend the bulk of their time on the steps leading up to this one:

  1. Exploring the data.
  2. Cleaning the data.
  3. Engineering new features.

This is because **better data beats fancier algorithms**.

In this lesson, you will learn how to setup the entire modeling process to maximize performance while **safeguarding against overfitting**.

We will swap algorithms in and out and automatically find the best parameters for each one.

### Split Dataset

We start with a crucial but sometimes overlooked step: _spending your data_.

Think of your data as a limited resource.

- You can spend some of it to train your model (i.e. feed it to the algorithm).
- You can spend some of it to evaluate (test) your model.
- But you cannot reuse the same data for both!

If you evaluate your model on the same data you used to train it, your model could be very overfit and you wouldn’t even know!

A model should be judged on its ability to predict _new, unseen data_.

Therefore, you should have separate training and test subsets of your dataset.

#### Train Test Split

**Training sets** are used to fit and tune your models.
**Test sets** are put aside as unseen data to evaluate your models.

- You should always split your data _before_ doing anything else.

- This is the best way to get reliable estimates of your models’ performance.

- After splitting your data, **do not touch your test set** until you are ready to choose your final model!

Comparing test vs. training performance allows us to avoid overfitting:

**If the model performs very well on the training data but poorly on the test data, then it is overfit.**


### What are Hyperparameters?

When we talk of tuning models, we mean tuning the **hyperparameters**.

There are two types of parameters in machine learning algorithms:

**The key distinction is:** model parameters can be learned directly from the training data while hyperparameters cannot.

#### Model parameters

Model parameters are learned attributes that define individual models.

- regression coefficients
- decision tree split locations
- They can be **learned directly** from the training data.

#### Hyperparameters

Hyperparameters express "higher-level" structural settings for algorithms.

- strength of the penalty used in regularized regression
- the number of trees to include in a random forest
- They are **decided** before fitting the model because they cannot be learned from the data.


### What is Cross-Validation?

Cross-validation is a method for getting a _reliable estimate of model performance_ using only the training data.

It help us tune our models.

There are several ways to cross-validate, but the most common method is **10-fold cross-validation** which breaks the training data into 10 equal parts (folds), essentially creating 10 miniature train/test splits.

These are the steps for 10-fold cross-validation:

  1. Split the data into 10 equal parts or _folds_.
  2. Train the model on 9 folds (such as the first 9 folds).
  3. Evaluate it on the 1 remaining "hold-out" fold.
  4. Perform steps (2) and (3) 10 times, each time holding out a different fold.
  5. Average the performance across all 10 hold-out folds.

The average performance across the 10 hold-out folds is the final performance estimate called the **cross-validated score**. Because we created 10 mini train/test splits, this score is usually pretty reliable.


### Fit and Tune Models

Now that we have split our dataset into training and test sets, and we have learned about hyperparameters and cross-validation, we are ready _fit and tune_ our models.

Basically, we just perform the entire cross-validation loop detailed above on each **set of hyperparameter values** we would like to try.
​
The high-level pseudo-code looks like this:

```
    For each algorithm (i.e. regularized regression, random forest, etc.):
      For each set of hyperparameter values to try:
        Perform cross-validation using the training set.
        Calculate cross-validated score.
```

At the end of this process, you will have a cross-validated score for each set of hyperparameter values (for each algorithm).

Then, we pick the best set of hyperparameters _within each algorithm_.

```
    For each algorithm:
      Keep the set of hyperparameter values with best cross-validated score.
      Re-train the algorithm on the entire training set (without cross-validation).
```

Each algorithm sends its own representatives (model trained on the best set of hyperparameter values) to the final selection which is coming up next...

### Select the Winning Model

By now, you will have a single "best" model _for each algorithm_ that has been tuned through cross-validation.

Most importantly, you have only used the training data.

Now it is time to evaluate each model and pick the best one.

Because you saved your **test set** as a truly unseen dataset, you can now use it get a reliable estimate of each models' performance.

There are a variety of **performance metrics** you can choose from.

We will not spend too much time on them here, but in general:

- For regression tasks, we recommend Mean Squared Error (MSE) or Mean Absolute Error (MAE) where lower values are better

- For classification tasks, we recommend Area Under ROC Curve (AUROC) where higher values are better.

The process is very straightforward:

1. For each of your models, make predictions on your test set.

2. Calculate performance metrics using those predictions and the "ground truth" target variable from the test set.

Finally, use these questions to help you pick the winning model:

- Which model had the best performance on the test set? (performance)
- Does it perform well across various performance metrics? (robustness)
- Did it also have (one of) the best cross-validated scores from the training set? (consistency)
- Does it solve the original business problem? (win condition)


----------


# Machine Learning Metrics

Evaluating your machine learning algorithm is an essential part of any project.

A model may give you satisfying results when evaluated using a metric say accuracy_score but may give poor results when evaluated against other metrics such as logarithmic_loss or any other such metric [3]

Most of the times we use classification accuracy to measure the performance of our model, but accuRracy alone is not enough to truly judge our model.

It is also worth mentioning that _metric_ is different from _loss function_ [3].

- Loss functions show a measure of the model performance and are used to train a machine learning model (using some kind of optimization) and are usually differentiable in model’s parameters.

- Metrics are used to monitor and measure the performance of a model (during training and test) and do not need to be differentiable.

If for some tasks the performance metric is differentiable, it can be used both as a loss function (perhaps with some regularizations added to it) and a metric such as MSE [3].

The article [3] has two parts:

- The first part covers 10 metrics that are widely used for evaluating classification and regression models.

- The second part covers 10 metrics that are used to evaluate ranking, computer vision, NLP, and deep learning models.

The author covers the popular metrics used in the following problems [3]:

- Classification Metrics (accuracy, precision, recall, F1-score, ROC, AUC, …)

- Regression Metrics (MSE, MAE)

- Ranking Metrics (MRR, DCG, NDCG)

- Statistical Metrics (Correlation)

- Computer Vision Metrics (PSNR, SSIM, IoU)

- NLP Metrics (Perplexity, BLEU score)

- Deep Learning Related Metrics (Inception score, Frechet Inception distance)


## Performance Metrics For Classification

We need some way to tell how well our classification model is doing [5].

- Accuracy
- Precision and Recall
- F1 Score

### Accuracy

Classification Accuracy is the ratio of number of correct predictions to the total number of input samples.

  Accuracy = Number of correct predictions / Total number of predictions

### Confusion Matrix

Confusion Matrix gives us a matrix as output and describes the complete performance of the model.

- True Positives: The cases in which we predicted YES and the actual output was also YES.

- True Negatives: The cases in which we predicted NO and the actual output was NO.

- False Positives: The cases in which we predicted YES and the actual output was NO.

- False Negatives: The cases in which we predicted NO and the actual output was YES.

Accuracy for the matrix can be calculated by taking average of the values lying across the main diagonal.

The Confusion Matrix forms the basis for the other types of metrics.

- True Positive Rate (Sensitivity) : True Positive Rate is defined as TP/ (FN+TP)

True Positive Rate corresponds to the proportion of positive data points that are correctly considered as positive, with respect to all positive data points.

- True Negative Rate (Specificity) : True Negative Rate is defined as TN / (FP+TN).

True Negative Rate corresponds to the proportion of negative data points that are correctly considered as negative, with respect to all negative data points.

- False Positive Rate : False Positive Rate is defined as FP / (FP+TN).

False Positive Rate corresponds to the proportion of negative data points that are mistakenly considered as positive with respect to all negative data points.

### Logarithmic Loss

Logarithmic Loss (Log Loss) works by penalising the false classifications which works well for multi-class classification.

### Area Under Curve (AUC)

Area Under Curve (AUC) is used for binary classification problem. AUC of a classifier is equal to the probability that the classifier will rank a randomly chosen positive example higher than a randomly chosen negative example.

  AUC has a range of [0, 1]. The greater the value, the better the performance of our model.

### F1 Score

F1 Score is used to measure a test’s accuracy.

F1 Score tries to find the balance between precision and recall.

F1 Score is the Harmonic Mean between precision and recall. The range for F1 Score is [0, 1]. It tells you how precise your classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances).

High precision but lower recall, gives you an extremely accurate, but it then misses a large number of instances that are difficult to classify. The greater the F1 Score, the better is the performance of our model.

- Precision: the number of correct positive results divided by the number of positive results predicted by the classifier. Precision indicates how precise or correct our predictions are.

  Precision = TP / (TP + FP)

- Recall: the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive).

  Recall will tell us how many patients were identified to be sick out of the patients who were actually sick. Thus, recall is a very important metric in the medical domain.

  Recall = TP / (TP + FN)


## Performance Metrics For Regression

The metrics like accuracy, precision, recall and F1 score are only applicable only if the problem is classification [5].

- MAE
- MSE
- RMSE

### Mean Absolute Error

Mean Absolute Error (MAE) is the average of the difference between the Original Values and the Predicted Values. MAE measures how far the predictions were from the actual output. However, they do not give us any idea of the direction of the error -- whether we are under or over predicting the data.

MAE has the advantage of not penalizing the large errors which is helpful in some cases. But one  disadvantage of MAE is that it uses the **absolute value** which is undesirable in many mathematical calculations.

### Mean Squared Error

Mean Squared Error (MSE) is similar to Mean Absolute Error, but MSE takes the average of the square of the difference between the original values and the predicted values.

The advantage of MSE is that it is easier to compute the gradient whereas MAE requires complicated linear programming tools to compute the gradient. Since we take the square of the error, the effect of larger errors become more pronounced than smaller error, so the model can now focus more on the larger errors.

### Root Mean Squared Error

Root Mean Squared Error (RMSE) is the most popular and widely used metric in regression problems. RMSE will take an assumption that the errors are normally distributed and they are unbiased.

Many Machine learning algorithms use RMSE, because it is faster and easier to compute.

Since the errors are squared and due to which RMSE will not be in the same scale as the errors. RMSE will also penalize large errors which is not good in some cases. RMSE is also highly affected by outliers.


## References

[1]: [Data Science Primer](https://elitedatascience.com/primer)

[2]: [From Encodings to Embeddings](https://towardsdatascience.com/from-encodings-to-embeddings-5b59bceef094)

[3]: [20 Popular Machine Learning Metrics](https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce)

[4]: [Metrics to Evaluate your Machine Learning Algorithm](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)

[5]: [Error Metrics in Machine learning](https://medium.com/analytics-vidhya/error-metrics-in-machine-learning-f9eed7b139f)


[Datasets for Data Science and Machine Learning](https://elitedatascience.com/datasets)
