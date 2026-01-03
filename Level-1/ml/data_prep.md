# Data Preparation

[Feature Engineering](./feature_engineering.md)

[Dimensionality Reduction](./dimensionality_reduction.md)

## Background

> Data quality is important to creating a successful machine learning model.

> If you torture the data, it will confess to anything - Ronald Coase

There is usually more than one way to sample and interogate data which means that deciding the best approach is subjective and open to bias.

Thus, _data torture_ is the practice of repeatedly interpreting source data until it reveals a desired result.

## Definitions

The key terms for discussing the machine learning workflow:

- Model: a set of patterns learned from data.

- Algorithm: a specific ML process used to train a model.

- Training data: the dataset from which the algorithm learns the model.

- Test data: a new dataset for reliably evaluating model performance.

- Features: Variables (columns) in the dataset used to train the model.

- Target variable: A specific variable you’re trying to predict.

- Observations: Data points (rows) in the dataset.

## ETL vs ELT

ETL and ELT (extract, transform, load or extract, load, and transform).

ETL transforms the data before loading it into a data warehouse while ELT loads the data and allows the transformation to be handled within the data warehouse [12].

- **Extract:** This refers to pulling the source data from the original database or data source.

  With ETL, the data goes into a temporary staging area.

  With ELT, it goes immediately into a data lake storage system.

- **Transform:** This refers to the process of changing the format/structure of the information so that it can integrate with the target system and the rest of the data within that system.

- **Load:** This refers to the process of inserting the information into a data storage system.

  In ELT scenarios, the raw unstructured data is loaded then transformed within the target system.

  In ETL, the raw data is transformed into structured data prior to reaching the target.

## Overview

The process of applied machine learning consists of a sequence of steps [6], [18]:

1. Define Problem
2. Prepare Data
3. Evaluate Models
4. Finalize Model

Here we are concerned with the data preparation step (step 2), and the common tasks used during the data preparation step in a machine learning project.

On a predictive modeling project, there are four main reasons the raw data usually cannot be used directly [2]:

- Data Types: Implementations of machine learning algorithms require data to be numeric.

- Data Requirements: Some machine learning algorithms impose requirements on the data.

- Data Errors: Statistical noise and errors in the data may need to be corrected.

- Data Complexity: Complex nonlinear relationships may be extracted from the data.

Therefore, the raw data must be pre-processed before being used to fit and evaluate a machine learning model which is referred to as _data preparation_.

Here are some things to keep in mind about data preparation:

- The approaches to data preparation depend on the dataset as well as the data types.

- We try to handle the worst cases and not necessarily every case.

- Exploratory Data Analysis (EDA) is crucial: summary stats and making plots of the data.

- It is estimated that 80% of AI project development time is spent on preparing the data [4].

## Data Preparation

The types of data preparation performed usually involve the following tasks [6], depending on the data:

- Exploratory Data Analysis (EDA): Get to know the data. Check if data is normally distributed or heavy-tailed; check for outliers; check if clustering of the data will help; check for imbalanced data.

- Data Preprocessing: Organize the selected data by formatting, cleaning, and sampling from it.

- Feature Selection: Identify the input variables that are most relevant to the task.

- Feature Engineering: Derive new variables from the available data.

- Data Transforms: Change the scale or distribution of variables.

- Dimensionality Reduction: Create compact projections of the data.

This provides a basic framework for exploring different data preparation algorithms that we may consider on a given project with structured or tabular data.

Depending on the project, there may be other steps as well [4]:

- Project Scoping: Sometimes we need to roadmap the project and anticipate data needs.

- Data Wrangling: We may need to restructure the dataset into a format that can be used by algorithms.

----------

The data preparation stage usually involves three steps that may overlap [5]:

1. Data Selection: Consider what data is available, what data is missing, and what data can be removed.

2. Data Preprocessing: Organize the selected data by formatting, cleaning, and sampling from it.

3. Data Transformation: Transform preprocessed data ready for machine learning by engineering features using scaling, attribute decomposition, and attribute aggregation.

  Split first and normalize later which means that we should perform the train-test split first then normalize the datasets.

Data preparation is a large topic that can involve a lot of iterations, exploration, and analysis.

## 6 Steps for Data Preparation

Data preparation in machine learning is a complex, multi-step process [8].

The final performance of an ML model depends on the dataset used for training.

Not having proper data is probably a top reason why an ML model is hard to build.

The possible consequences of skipping on data preparation include:

- Reduced model accuracy
- Overfitting or underfitting
- Increased computation costs
- Model bias
- Scalability issues
- Misleading insights

Here are the 6 basic steps for data preparation [8]:

1. Data Collection
2. Data Cleaning
3. Data Transformation
4. Data Reduction
5. Data Splitting
6. Feature Engineering

### 1. Data Collection

_Data Collection_ implies gathering relevant data from various sources such as databases, APIs, files, and online repositories [8].

The first step in data preparation for machine learning encompasses matters of the type, volume, and quality of data.

Here are some common sources of data [8]:

- Internal sources: Enterprise data warehouses, sales transactions, customer interactions.

- External sources: Public data sources like Kaggle, UCI Machine Learning Repository, and Google Dataset Search.

- Web scraping: Automated tools for extracting data from websites.

- Surveys: Collecting specific data points from target audiences.

Strategies to compensate for lack of data [8]:

- Data augmentation: Generating more data from existing samples.

- Active learning: Selecting informative data samples for labeling.

- Transfer learning: Using pre-trained models for related tasks.

- Collaborative data sharing: Working with other entities to share data.

### 2. Data Cleaning

The data obtained is usually not suitable for ML which ia why data cleaning or simply preparing raw data is required [8], [9], [19]:

- Handling missing data: Imputation, interpolation, deletion.

- Handling outliers: Remove, transform, winsorize, or treat them as a separate class.

- Removing duplicates: Use exact matching, fuzzy matching, and other techniques.

- Handling irrelevant data: Identify and remove irrelevant data points.

- Handling incorrect data: Transform or remove erroneous data points.

- Handling imbalanced data: Resampling, synthetic data generation, cost-sensitive learning, ensemble learning.

### 3. Data Transformation

Converting raw data into a suitable format for machine learning algorithms which enhances algorithmic performance and accuracy [8], [9].

- Scaling: Transform features to a specified range (such as 0 to 1) to balance feature importance.

- Normalization: Adjust data distribution for a more balanced feature comparison.

- Encoding: Convert categorical data into numerical format using techniques such as one-hot encoding, ordinal encoding, and label encoding.

- Discretization: Transform continuous variables into discrete categories.

- Dimensionality reduction: Limit the number of features to reduce complexity using techniques such as Principal Component Analysis (PCA) or  Linear Discriminant Analysis (LDA).

- Log transformation: Apply a logarithmic function to skewed data for a more symmetric distribution.

### 4. Data Reduction

Sometimes the number of parameters in the data gathered is higher than necessary.

For example, a survey may include responses that are not valid for training the intended model.

Data reduction helps preserve essential information while reducing complexity:

- Dimensionality reduction: Techniques such as Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) reduce the number of features while retaining significant information.

- Feature selection: Methods such as recursive feature elimination, mutual information, and chi-square tests select the most relevant features.

- Sampling: Reduce the dataset size by selecting a representative subset of data points.

### 5. Data Splitting

Three types of datasets are necessary for training every single ML model:

- Training dataset: Used to train the model to learn patterns and relationships between input features and target variables.

- Validation dataset: Used to tune model hyperparameters and evaluate performance during training to prevent overfitting.

- Testing dataset: Used to assess the performance of the final model on unseen data to make sure it generalizes well to new data.

Dividing data into subsets is another necessary step.

For training datasets, engineers use data of varying difficulty from least to most complex.

Validation datasets include specific cases, suitable for model fine tuning.

Testing dataset is a suite of real-life numbers / images / documents, helping to actually measure the model’s accuracy.

Recommended approaches:

- Random sampling: Randomly splitting data, useful for large datasets.

- Stratified sampling: Ensuring subsets maintain the same distribution of class labels or characteristics, ideal for imbalanced datasets.

- Time-based sampling: Using historical data for training and future data for testing which applies to time-series data.

- Cross-validation: Dividing data into multiple folds for training and testing to get a more accurate performance estimate (e.g. k-fold cross-validation).

### 6. Feature Engineering

After training of a model, there is usually a need to scale it and reinforce it with additional functionalities.

Creating new features or modifying existing ones to enhance model performance is known as feature engineering.

Recommended techniques:

- Interaction terms: Creating new features by combining existing ones.

- Polynomial features: Adding polynomial terms to capture nonlinear relationships.

- Domain knowledge: Leveraging expertise to create meaningful features.


## Notes on Data Preparation

The articles in [12] cover exploratory analysis, data cleaning, feature engineering, algorithm selection, and model training.

How to Make Your Data Models Modular to Avoid highly coupled systems and unexpected production bugs [14].

Here are some relevant topics disucssed in [1]:

7.1 Handling Missing Data
7.2 Data Transformation
7.3 String Manipulation

8.2 Combining and Merging Datasets
8.3 Reshaping and Pivoting

10. Data Aggregation and Group Operations

### Import data

For ML projects, it can be confusing to determine which library to choose to read and manipulate datasets, especially image and text [1], [2].

- Split data along delimiters (CSV)

- Extract parts from data entries (Do you only need part of a certain attribute?)

### Format adjustments

- Remove leading and trailing spaces

- Standardize types (decimal separators, date formats, or measurement units)

- Replace unrecognizable or corrupted characters

- Check for truncated entries (data entries that are cut off at a certain position)

### Correct inconsistencies

- Check for invalid values (age is 200 or negative)

- Check for extreme values in numerical data

- Check for wrong categories in categorical data (similar products should not be put into different categories)

### Check for extreme values

- High Cardinality: the number of different labels in categorical data is very high which causes problems to model to learn.

- Outliers: the extreme cases that may be due to error, but not in every case.

### Avoid Data Leakage

In general, we must answer two key questions to prevent data leakage [16]:

1. Am I exposing information from the test set to the training process?
2. Am I using future data that won’t be available when making predictions?

These two questions will help avoid overly optimistic performance metrics and build models that generalize well to new data.

We should also apply the same principles to the cross-validation process to ensure each fold is free from leakage.

### Imbalanced Datasets

Imbalanced data occurs when there is an uneven distribution of classes or labels [10].

Models trained with imbalanced data usually have high precision and recall scores for the majority class, whereas these scores will likely drop significantly for the minority class.

In a credit card detection task, the number of non-fraudulent transactions will likely be much greater than the number of fraudulent credit card transactions.

Need to upsample but categories with only 1 entry when oversampled will give a 100% accuracy and artificially inflate the total accuracy/precision.

- We can use `UpSample` in Keras/PyTorch and `pd.resample()` in Pandas


## Data Cleaning

Data cleaning includes the following tasks [8], [9], [15], [19]:

- Delete Unnecessary Columns
- Remove irrelevant data
- Check data types
- Fix structural errors
- Remove duplicates
- Handle incorrect data
- Handle missing data
- Handle outliers
- Handle categorical data
- Handle imbalanced data

- Encode class labels
- Parsing dates
- Character encodings
- Inconsistent data entry

Structural Errors are typos or inconsistent capitalization which is mostly a concern for categorical features.

Data cleaning refers to identifying and correcting errors in the dataset that may negatively impact a predictive model [7]:

- Identify Columns That Contain a Single Value
- Delete Columns That Contain a Single Value
- Consider Columns That Have Very Few Values
- Remove Columns That Have A Low Variance
- Identify Rows that Contain Duplicate Data
- Delete Rows that Contain Duplicate Data

The article [11] provides some Tips and Tricks to Deal with a Messy Date String Column in Pandas Dataframe.

### Delete Unnecessary Columns

There can be columns in the dataset that we do not need in our data analysis, so we can remove them using the `drop()` method with the specified column name [15].

```py
  df.drop('last_name', axis = 1, inplace = True)
```

We set the axis to 1 to specify that we want wasto delete a column and the inplace argument is set to True so that we modify the existing DataFrame to avoid creating a new DataFrame without the removed column.

### Check Data Types

```py
  df.info()
```

```py
  # List of numeric columns
  num_cols = ['age', 'bp', 'sg', 'al', 'su',
              'bgr', 'bu', 'sc', 'sod', 'pot',
              'hemo', 'pcv', 'wbcc', 'rbcc']

  for column in df.columns:
      if column in num_cols:
          # Replace ‘?’ with ‘NaN’
          # df[column] = df[column].replace('?', np.nan)

          # Convert to numeric type
          df[column] = pd.to_numeric(df[column])
```

### Data Type Conversion

Sometimes, data types might not be correct. For example, a date column might be interpreted as strings [15].

We need to convert these columns to the appropriate types.

The easiest way to convert object type to date type is using the `to_datetime()` method.

We may need set the `dayfirst` argument to True because some dates start with the day first (DD-MM-YYYY).

```py
# Converting advertisement_date column to datetime
df['advertisement_date'] = pd.to_datetime(df['advertisement_date'], dayfirst = True)

# Converting sale_date column to datetime
df['sale_date'] = pd.to_datetime(df['sale_date'], dayfirst = True)
```

We can also convert both columns at the same time by using the `apply()` method with `to_datetime()`.

```py
# Converting advertisement_date and sale_date columns to datetime
df[['advertisement_date', 'sale_date']] = df[['advertisement_date', 'sale_date']].apply(pd.to_datetime, dayfirst =  True)
```

Both approaches give you the same result.

### Inconsistent Data Entry

TODO: This will most likely vary

### Inconsistent Formatting

The data you have selected may not be in the format that is needed which includes: format adjustments, correct inconsistencies, and handle errors in variables.

### Handle Missing Values

There are various ways of dealing with missing values and it is likely that we will need to determine which method is right for a task at hand on a case-by-case basis [9], [10], [13], [17].

Missing data can happen due to forgotten to store, inappropriate data handling, inefficient data entry at the ground level, etc.

The removal of samples or dropping of feature columns may not feasible because we might lose too much valuable data.

We can use interpolation techniques to estimate the missing values from the other training samples in the dataset.

There are various methods [17]:

1. Fill with constant values ​​(0, 1, 2, etc.)

2. Use statistical methods such as mean / median.

3. Using values ​​from other data (such as values ​​from previous or subsequent data)

4. Creating a predictive model to estimate the missing values.

One of the most common interpolation techniques is _mean imputation_ where we simply replace the missing value by the mean value of the entire feature column.

- Numerical Imputation
- Categorical Imputation

Check for null values.
We can drop or fill the `NaN` values.

```py
  # return the number of missing values (NaN) per column
  df.isnull().sum()

  # remove all rows that contain a missing value
  df.dropna()

  # remove all columns with at least one missing value
  df.dropna(axis=1)

  # Drop the NaN
  df['col_name'] = df['col_name'].dropna(axis=0, how="any")

  # check NaN again
  df['col_name'].isnull().sum()

  # remove rows with None in column "date"
  # notna is much faster
  df.dropna(subset=['date'])
  df = df[df["date"].notna()]
```

```py
  # check for nan/null
  df.isnull().values.any()

  # count of nulls per column
  df.isnull().sum()

  # Drop NULL values
  df.dropna(inplace=True)


  # Find and verify missing values
  np.where(pd.isnull(df))
  df.iloc[296, 12]

  # replace missing values
  df.replace(np.nan, 0)

  # count of unique values
  df.nunique()

  # change null to 0
  df5.loc[df5['column1'].isnull(),   'column1'] = 0

  # change nan to 0
  df['column1'] = df['column1'].fillna(0)

  # drop rows where all columns are missing/NaN
  df.dropna(axis=0, how="any", inplace=True)
```

```py
  # We can delete specific columns by passing a list
  df.dropna(subset=['City', 'Shape Reported'], how='all')

  # Replace NaN by a specific value using fillna() method
  df['Shape Reported'].isna().sum()

  df['Shape Reported'].fillna(value='VARIOUS', inplace=True)
  df['Shape Reported'].isna().sum()
  df['Shape Reported'].value_counts(dropna=False)
```

### Handle Duplicate Values

```py
  # We can show if there are duplicates in specific column
  # by calling 'duplicated' on a Series.
  df.zip_code.duplicated().sum()

  # Check if an entire row is duplicated
  df.duplicated().sum()

  # find duplicate rows across all columns
  dup_rows = df[df.duplicated()]

  # find duplicate rows across specific columns
  dup_rows = df[df.duplicated(['col1', 'col2'])]

  # Return DataFrame with duplicate
  # rows removed, optionally only considering certain columns.
  # 'keep' controls the rows to keep.
  df.drop_duplicates(keep='first').shape

  # extract date column and remove None values
  # drop_duplicates is faster on larger dataframes
  date = df[df["date"].notna()]
  date_set = date.drop_duplicates(subset=['date'])['date'].values

  # extract date column and remove None values
  date = df[df["date"].notna()]['date'].values
  date_set = np.unique(date)
```

```py
  # check for duplicate values
  df.duplicated()

  # Remove duplicates
  df.drop_duplicates(subset=['PersonId', 'RecordDate'], keep='last')

  # Drop duplicate column
  df_X.drop(['TEST1', 'TEST2'], axis=1)
```

### Handle Outliers

- Remove: Outlier entries are deleted from the distribution

- Replace: The outliers could be handled as missing values and replaced with suitable imputation.

- Cap: Using an arbitrary value or a value from a variable distribution to replace the maximum and minimum values.

- Discretize: Converting continuous variables into discrete values.

```py
  # making boolean series for a team name
  filter1 = data["Team"] == "Atlanta Hawks"

  # making boolean series for age
  filter2 = data["Age"] > 24

  # filtering data on basis of both filters
  df.where(filter1 & filter2, inplace=True)

  df.loc[filter1 & filter2]

  # display
  print(df.head(20))
```

```py
  def get_outliers(df):
      """
      Identify the number of outliers +/- 3 standard deviations.
      Pass this function a data frame and it returns a dictionary.
      The 68–95–99.7 rule states that 99.7% of all data in a normal
      distribution lies within three standard deviations of the mean.
      When your data is highly left or right-skewed, this will not be true.
      """
      outs = {}

      df = df.select_dtypes(include=['int64'])

      for col in df.columns:
          # calculate summary statistics
          data_mean, data_std = np.mean(df[col]), np.std(df[col])

          # identify outliers
          cut_off = data_std * 3
          lower, upper = data_mean - cut_off, data_mean + cut_off

          # identify outliers
          outliers = [x for x in df[col] if x < lower or x > upper]

          outs[col] = len(outliers)

          return outs
```


### Remove Outliers

```py
  from scipy import stats

  # build a list of columns that you wish to remove ouliers from
  out_list = ['balance', 'pdays', 'duration']

  # overwrite the dataframe with outlier rows removed.
  df = df[((np.abs(stats.zscore(df[out_list])) < 3)).all(axis=1)]
```

## Data Preprocessing

The three most common data preprocessing steps are: formatting, cleaning, and sampling [5].

1. Formatting: The data you have selected may not be in the format that is needed which includes: format adjustments, correct inconsistencies, and handle errors in variables.

2. Cleaning: Identify and correct mistakes or errors in the data which includes: check data types; handle missing or invalid values; handle outliers; handle categorical values; encode class labels; parse dates; character encodings; handle imbalanced data.

  Check for extreme values:

- High Cardinality: the number of different labels in categorical data is very high which causes problems for the model to learn.

- Outliers: the extreme cases that may be due to error but not in every case.

  There may be data instances that are incomplete and do not contain the data needed to address the problem; these instances may need to be removed.

  There may be sensitive information in some of the attributes and these attributes may need to be anonymized or removed from the data entirely.

3. Sampling: There may be far more selected data available than we need.

  More data can result in much longer running times for algorithms and larger computational and memory requirements.

  We can use a smaller representative sample of the selected data that may be faster for exploring and prototyping solutions before considering the whole dataset.

----------

Data preparation: This step includes the following tasks: data preprocessing, data cleaning, and exploratory data analysis (EDA).

For image data, we would resize images to a lower dimension, such as (299 x 299), to allow mini-batch learning and also to keep up the computing limitations.

For text data, we would Remove newlines and tabs; Strip HTML Tags; Remove Links; Remove Whitespaces, and other possible steps listed in NLP Text Preprocessing on my GitHub repo.

Feature engineering: This step includes the following tasks: quantization or binning; mathematical transforms; scaling and normalization; modifying and/or creating new features.

For image data, we would perform image augmentation, which is described in Image Augmentation on my GitHub repo.

For text data, we would convert text data features into vectors and perform Tokenization, Stemming, and Lemmatization, as well as other possible steps described in Natural Language Processing on my GitHub repo.

----------

Design: This step includes the following tasks: data preparation, decomposing the problem, and building and evaluating models.

We can use AutoML or create a custom test harness to build and evaluate many models to determine what algorithms and views of the data should be chosen for further study.

### Parsing dates (Python)

Method 1: Parse date columns using `read_csv`

```py
    def parser(x):
        return dt.datetime.strptime(x, "%Y-%m-%d")

    def load_data(name):
        df_data = pd.read_csv(
            file_path,
            header=0,
            index_col=0,
            parse_dates=["day"],
            date_parser=parser    # optional
        )

        return df_data
```

Method 2: Parse dates using `to_datetime`

```py
    def load_data(name):
        df_data = pd.read_csv(name, header=3, index_col=0)

        # Replace index with DateTime
        df_data.index = pd.to_datetime(df_data.index)

        return df_data
```

### Encoding Categorical Features (Python)

Most machine learning algorithms and deep learning neural networks require that input and output variables are numbers [16] which means that categorical data must be encoded to numbers before we can use it to fit and evaluate a model.

_Binning_ is a technique for transforming variables whose values ​​are numeric into categorical ones [17].

For categorical data, we need to distinguish between _nominal_ and _ordinal_ features.

- Ordinal features can be understood as categorical values that can be sorted or ordered.

  Example: T-shirt size would be an ordinal feature because we can define an order M < L < XL.

- Nominal features do not imply any order.

  Example: T-shirt color is a nominal feature since it typically does not make sense to say that red is larger than blue.

There are several ways to encode categorical variables [9], [16]:

  1. Integer (Ordinal) Encoding: each unique label or category is mapped to an integer.

  2. One Hot Encoding: each label is mapped to a binary vector.

  3. Dummy Variable Encoding

  4. Learned Embedding: a distributed representation of the categories is learned.

#### Integer (Ordinal) Encoding

To make sure that the ML algorithm interprets the ordinal features correctly, we need to convert the categorical string values to integers.

Unfortunately, there is no convenient function that can automatically derive the correct order of the labels of our size feature.

Therefore, we need to define the mapping manually.

```py
    size_mapping = { 'XL': 3, 'L': 2, 'M': 1}
    df['size'] = df['size'].map(size_mapping)
```

#### Encoding Class Labels

Many machine learning libraries require that class labels are encoded as _integer_ values.

It is considered good practice to provide class labels as integer arrays to avoid technical glitches.

```py
    # Handle categorical features
    df['is_white_wine'] = [1 if typ == 'white' else 0 for typ in df['type']]

    # Convert to a binary classification task
    df['is_good_wine'] = [1 if quality >= 6 else 0 for quality in df['quality']]

    df.drop(['type', 'quality'], axis=1, inplace=True)
```

To encode the class labels, we can use an approach similar to the mapping of ordinal features above.

We need to remember that class labels are not ordinal so it does not matter which integer number we assign to a particular string-label.

There is a convenient `LabelEncoder` class in scikit-learn to achieve the same results as _map_.

```py
    from sklearn.preprocessing import LabelEncoder

    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    # array([0, 1, 0])

    class_le.inverse_transform(y)
    # array(['class1', 'class2', 'class1'], dtype=object)
```

```py

```

#### One-Hot Encoding of Nominal Features

A one-hot encoding is a type of encoding in which an element of a finite set is represented by the index in that set where only one element has its index set to “1” and all other elements are assigned indices within the range [0, n-1].

In contrast to binary encoding schemes where each bit can represent 2 values (0 and 1), one-hot encoding assigns a unique value to each possible value.

In the previous section, we used a simple dictionary-mapping approach to convert the ordinal size feature into integers.

Since scikit-learn's estimators treat class labels without any order, we can use the convenient `LabelEncoder` class to encode the string labels to integers.

```py
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
```

After executing the code above, the first column of the NumPy array X now holds the new color values which are encoded as follows: blue = 0, green = 1, red = 2 n

However, we will make one of the most common mistakes in dealing with categorical data:

Although the color values are not ordered, a ML algorithm will assume that green is larger than blue, and red is larger than green. Therefore, the results would not be optimal.

A common workaround is to use a technique called _one-hot encoding_ to create a new dummy feature for each unique value in the nominal feature column.

Here, we would convert the color feature into three new features: blue, green, and red.

Binary values can then be used to indicate the particular color of a sample; for example, a blue sample can be encodedas blue=1, green=0, red=0.

We can use the `OneHotEncoder` that is implemented in the `scikit-learn.preprocessing` module.

An even more convenient way to create the dummy features via one-hot encoding is to use the `get_dummies` method in pandas.

#### Dummy Variable Encoding

Most machine learning algorithms cannot directly handle categorical features that are _text values_.

Therefore, we need to create dummy variables for our categorical features which is called _one-hot encoding_.

The one-hot encoding creates one binary variable for each category which includes redundancy.

In contrast, a dummy variable encoding represents N categories with N-1 binary variables.

```py
    pd.get_dummies(df, columns=['Color'], prefix=['Color'])
```

Applied to a DataFrame, the `get_dummies` method will only convert string columns and leave all other columns unchanged:

```py
    pd.get_dummies(df[['price', 'color', 'size']])

    # drops the header row
    df_encoded = pd.get_dummies(df, drop_first=True)
```

A dummy variable representation is required for some models such as linear regression model (and other regression models that have a bias term) since a one hot encoding will cause the matrix of input data to become singular which means it cannot be inverted, so the linear regression coefficients cannot be calculated using linear algebra. Therefore, a dummy variable encoding must be used.

However, we rarely encounter this problem in practice when evaluating machine learning algorithms other than linear regression.

#### One-Hot Encoding Example

A one-hot encoding is appropriate for categorical data where no relationship exists between categories.

The scikit-learn library provides the `OneHotEncoder` class to automatically one hot encode one or more variables.

By default the `OneHotEncoder` class will output data with a sparse representation which is efficient because most values are 0 in the encoded representation. However, we can disable this feature by setting the `sparse=False` so that we can review the effect of the encoding.

```py
    import numpy as np
    import pandas as pd

    from numpy import mean
    from numpy import std
    from pandas import read_csv

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.metrics import accuracy_score

    # define the location of the dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"

    # load the dataset
    dataset = read_csv(url, header=None)

    # retrieve the array of data
    data = dataset.values

    # separate into input and output columns
    X = data[:, :-1].astype(str)
    y = data[:, -1].astype(str)

    # split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # one-hot encode input variables
    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit(X_train)
    X_train = onehot_encoder.transform(X_train)
    X_test = onehot_encoder.transform(X_test)

    # ordinal encode target variable
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    # define the model
    model = LogisticRegression()

    # fit on the training set
    model.fit(X_train, y_train)

    # predict on test set
    yhat = model.predict(X_test)

    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.2f' % (accuracy * 100))
```


## Data Splitting

[Train-Test Split](./ml/train_test_split.md)

A key step in ML is the choice of model.

> Split first, normalize later.

A train-test split conists of the following:

1. Split the dataset into training, validation, and test sets.

2. We normalize the training set only (fit_transform).

3. We normalize the validation and test sets using the normalization factors from train set (transform).

> Instead of discarding the allocated test data after model training and evaluation, it is a good idea to retrain a classifier on the entire dataset for optimal performance.


## Data Pipelines

[Data Pipelines](./pipelines.md)

There are multiple stages to running machine learning algorithms since it involves a sequence of tasks including pre-processing, feature extraction, model fitting, performance, and validation.


## Bootstrapping

The goal of _bootstrap_ is to create an estimate (sample mean x̄) for a population parameter (population mean θ) based on multiple data samples obtained from the original sample.

Bootstrapping is done by repeatedly sampling (with replacement) the sample dataset to create many simulated samples.

Each simulated bootstrap sample is used to calculate an estimate of the parameter and the estimates are then combined to form a sampling distribution.

The bootstrap sampling distribution then allows us to draw statistical inferences such as estimating the standard error of the parameter.



## Examples and Tutorials

Here are some code samples and tutorials:

### Data Preparation

[How to Fully Automate Data Cleaning with Python in 5 Steps](https://www.kdnuggets.com/how-to-fully-automate-data-cleaning-with-python-in-5-steps)

[Data Preparation for Machine Learning (Python)](https://machinelearningmastery.com/start-here/#dataprep)

[Tour of Data Preparation Techniques for Machine Learning](https://machinelearningmastery.com/data-preparation-techniques-for-machine-learning/)

[How to Perform Data Cleaning for Machine Learning with Python](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)

[Essential Data Cleaning Techniques for Accurate Machine Learning Models](https://www.kdnuggets.com/essential-data-cleaning-techniques-accurate-machine-learning-models)

----------

[Preprocessing of the data using Pandas and SciKit](https://mclguide.readthedocs.io/en/latest/sklearn/preprocessing.html)

[Missing Values Be Gone](https://towardsdatascience.com/missing-values-be-gone-a135c31f87c1?source=rss----7f60cf5620c9---4&gi=d11a8ff041dd)

[ML Guide Quick Reference](https://mclguide.readthedocs.io/en/latest/sklearn/guide.html)

[The Lazy Data Scientist’s Guide to AI/ML Troubleshooting](https://medium.com/@ODSC/the-lazy-data-scientists-guide-to-ai-ml-troubleshooting-abaf20479317?source=linkShare-d5796c2c39d5-1638394993&_branch_referrer=H4sIAAAAAAAAA8soKSkottLXz8nMy9bLTU3JLM3VS87P1Xcxy8xID4gMc8lJAgCSs4wwIwAAAA%3D%3D&_branch_match_id=994707642716437243)

[How to Select a Data Splitting Method](https://towardsdatascience.com/how-to-select-a-data-splitting-method-4cf6bc6991da)

### Categorical Data

[Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)

[3 Ways to Encode Categorical Variables for Deep Learning](https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/)

----------

[4 Categorical Encoding Concepts to Know for Data Scientists](https://towardsdatascience.com/4-categorical-encoding-concepts-to-know-for-data-scientists-e144851c6383)

[Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)

[Stop One-Hot Encoding Your Categorical Variables](https://towardsdatascience.com/stop-one-hot-encoding-your-categorical-variables-bbb0fba89809)

### Scaling

[Scaling vs Normalizing Data](https://towardsai.net/p/data-science/scaling-vs-normalizing-data-5c3514887a84)

[The Mystery of Feature Scaling is Finally Solved](https://towardsdatascience.com/the-mystery-of-feature-scaling-is-finally-solved-29a7bb58efc2)

[Scaling and Normalization: Standardizing Numerical Data](https://letsdatascience.com/scaling-and-normalization/)

-----

[How to Selectively Scale Numerical Input Variables for Machine Learning](https://machinelearningmastery.com/selectively-scale-numerical-input-variables-for-machine-learning/)

[How to use Data Scaling to Improve Deep Learning Model Stability and Performance](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)

[How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)

[The Mystery of Feature Scaling is Finally Solved](https://towardsdatascience.com/the-mystery-of-feature-scaling-is-finally-solved-29a7bb58efc2?source=rss----7f60cf5620c9---4)

### Normalization

[How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)

[How to Use Power Transforms for Machine Learning](https://machinelearningmastery.com/power-transforms-with-scikit-learn/)

### Data Splitting

[Training-validation-test split and cross-validation done right](https://machinelearningmastery.com/training-validation-test-split-and-cross-validation-done-right/)

[A Gentle Introduction to k-fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)

[How to Configure k-Fold Cross-Validation](https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/)

### Data Sampling

[Data Sampling Methods in Python](https://towardsdatascience.com/data-sampling-methods-in-python-a4400628ea1b)

[Common Data Problems (and Solutions)](https://www.kdnuggets.com/2022/02/common-data-problems-solutions.html)


## Data Preprocessing Tools

### Joblib

**Joblib** is an open-source Python library that helps to save pipelines to a file that can be used later.

[Using joblib to speed up your Python pipelines](https://medium.com/data-science/using-joblib-to-speed-up-your-python-pipelines-dd97440c653d)

[Lightweight Pipelining In Python](https://medium.com/data-science/lightweight-pipelining-in-python-1c7a874794f4)

### Kedro

`Kedro` is an open-source Python framework for creating reproducible, maintainable, and modular data science code.

### Kestra

**Kestra** is an infinitely scalable orchestration and scheduling platform, creating, running, scheduling, and monitoring millions of complex pipelines.

Kestra can manage ETL and ELT in the same solution, handling even the most complex workflows.

ETL processes can be used to scrub sensitive data, ensuring compliance, loading the transformed data within a temporary table.

With Kestra’s capacity for parallel flows, the rest of the data can be handled by ELT.

Kestra is able to perform ELT workloads on its own or with integrations to many popular solutions.

Kestra can handle loading data from BigQuery, CopyIn, Postgres, and more.

A simple query can be performed to move the data, for example, SQL INSERT INTO SELECT statements.

Dependencies between flows can be handled with Kestra’s trigger mechanisms to transform the data within the database (cloud or physical).

ETL is just as easily managed by Kestra’s flexible workflows.

FileTransform plugins are one possible method, but you can write a simple Python/Javascript/Groovy script to transform an extracted dataset data row per row.

For example, you can remove columns with personal data, clean columns by removing dates, and more. Integrating a custom docker image into your workflow is another method that can be used to transform the data.

Not only can you transform data row per row, you can potentially handle conversion of data between formats, for example, transforming AVRO data to JSON or CSV, or vice versa.

This is not usually possible with most solutions. Most ELT tools often prevent ETL processes by design because they cannot handle heavy transform operations.

Kestra is able to handle both because all transformations are considered to be row per row, and therefore do not use any memory to perform the function, only CPU.


## References

[1]: W. McKinney, Python for Data Analysis, 2nd ed., Oreilly, ISBN: 978-1-491-95766-0, 2018

[2]: J. Brownlee, Data Preparation for Machine Learning, Machine Learning Mastery, v1.1, 2020.

[3]: [Data Cleaning and Preprocessing for data science beginners](https://datasciencehorizons.com/data-cleaning-preprocessing-data-science-beginners-ebook/)

[4]:[Data Science Primer](https://elitedatascience.com/primer)

[5]: [How to Prepare Data For Machine Learning](https://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/)

[6]: [Tour of Data Preparation Techniques for Machine Learning](https://machinelearningmastery.com/data-preparation-techniques-for-machine-learning/)

[7]: [How to Perform Data Cleaning for Machine Learning with Python](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)

[8]: [Preparing Data for ML Development in 6 Steps](https://intelliarts.com/blog/data-preparation-in-machine-learning/)

[9]: [5 Essential Machine Learning Techniques to Master Your Data Preprocessing](https://pub.towardsai.net/5-machine-learning-data-preprocessing-techniques-e888f6d220e1)

[10]: [Read datasets with URL](https://towardsdatascience.com/dont-download-read-datasets-with-url-in-python-8245a5eaa919)

[11]: [Clean a Messy Date Column with Mixed Formats in Pandas](https://towardsdatascience.com/clean-a-messy-date-column-with-mixed-formats-in-pandas-1a88808edbf7)

[12]: [ELT vs ETL: Why not both?](https://medium.com/geekculture/elt-vs-etl-why-not-both-d0c4a0d30fc0)

[13]: [How to Detect Missing Values and Dealing with Them: Explained](https://medium.com/geekculture/ow-to-detect-missing-values-and-dealing-with-them-explained-13232230cb64)

[14]: [How to Make Your Data Models Modular](https://towardsdatascience.com/how-to-make-your-data-models-modular-71b21cdf5208)

[15]: [Mastering the Art of Data Cleaning in Python](https://www.kdnuggets.com/mastering-the-art-of-data-cleaning-in-python)

[16]: [Seven Common Causes of Data Leakage in Machine Learning](https://towardsdatascience.com/seven-common-causes-of-data-leakage-in-machine-learning-75f8a6243ea5)

[17]: [10 Basic Feature Engineering Techniques to Prepare Your Data](https://pub.towardsai.net/10-basic-feature-engineering-techniques-to-prepare-your-data-a43e99a0bf00)

[18]: [Data4ML Preparation Guidelines (Beyond The Basics)](https://pub.towardsai.net/data4ml-preparation-guidelines-beyond-the-basics-7613ff4282ff)

[19]: [Understanding Outliers in Machine Learning](https://blog.gopenai.com/understanding-outliers-in-machine-learning-732e43566763)

----------

[13 ways to access data in Python](https://towardsdatascience.com/13-ways-to-access-data-in-python-bac5683e0063)

[Refactoring for Scalable Python Code With Pandas](https://betterprogramming.pub/refactoring-for-scalable-python-code-with-pandas-727d15f14852)

[Concatenating CSV files using Pandas module](https://www.geeksforgeeks.org)

[Data Preprocessing for Effective Machine Learning Models](https://pub.towardsai.net/data-preprocessing-for-effective-machine-learning-models-e6ed0e83e888)


[Major Problems of Machine Learning Datasets: Part 3](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-3-eae18ab40eda)

[6 Tips for Dealing With Null Values](https://towardsdatascience.com/6-tips-for-dealing-with-null-values-e16d1d1a1b33)


[INFOGRAPHIC: Data prep and Labeling](https://www.cognilytica.com/2019/04/19/infographic-data-prep-and-labeling/)

[Kaggle Data Cleaning Challenge: Missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)

[Deduplicate and clean up millions of location records](https://towardsdatascience.com/deduplicate-and-clean-up-millions-of-location-records-abcffb308ebf)


[7 Cognitive Biases That Affect Your Data Analysis (and How to Overcome Them)](https://www.kdnuggets.com/7-cognitive-biases-that-affect-your-data-analysis-and-how-to-overcome-them)

