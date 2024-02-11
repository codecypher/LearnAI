# Data Preparation

[Feature Engineering](./feature_engineering.md)

[Dimensionality Reduction](./dimensionality_reduction.md)


## Overview

> Data quality is important to creating a successful machine learning model

[Feature Engineering](./feature_engineering.md)

> If you torture the data, it will confess to anything - Ronald Coase

There is often more than one way to sample and interogate data which means that deciding the best approach is subjective and open to bias. Thus, _data torture_ is the practice of repeatedly interpreting source data until it reveals a desired result. 


The approaches to data preparation will depend on the dataset as well as the data types, so there is no perfect list of steps.

[Concatenating CSV files using Pandas module](https://www.geeksforgeeks.org)

You try to handle the "worst" cases and not necessarily every case.

Exploratory Data Analysis (EDA) is crucial: summary stats and making plots of the data.

NOTE: It is estimated that 80% of AI project development time is spent on preparing the data [6]. 



## ETL vs ELT
 
ETL and ELT (extract, transform, load, or extract, load, and transform). 

ETL transforms the data before loading it into a data warehouse while ELT loads the data and allows the transformation to be handled within the data warehouse [13]. 

- **Extract:** This refers to pulling the source data from the original database or data source. 

  With ETL, the data goes into a temporary staging area. 

  With ELT, it goes immediately into a data lake storage system.

- **Transform:** This refers to the process of changing the format/structure of the information so that it can integrate with the target system and the rest of the data within that system.

- **Load:** This refers to the process of inserting the information into a data storage system. 

  In ELT scenarios, raw/unstructured data is loaded, then transformed within the target system. 

  In ETL, the raw data is transformed into structured data prior to reaching the target.

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



## Data Preparation

The [Data Science Primer](https://elitedatascience.com/primer) covers exploratory analysis, data cleaning, feature engineering, algorithm selection, and model training.

7.1 Handling Missing Data
7.2 Data Transformation
7.3 String Manipulation

8.2 Combining and Merging Datasets
8.3 Reshaping and Pivoting

10. Data Aggregation and Group Operations


How to Make Your Data Models Modular to Avoid highly coupled systems and unexpected production bugs [16]. 


### Import data

For ML projects, it can be confusing to determine which library to choose to read and manipulate datasets, especially image and text [1] [2].  

- Split data along delimiters (CSV)

- Extract parts from data entries (Do you only need part of a certain attribute?)

- Remove leading and trailing spaces

### Format adjustments

- Standardize types (decimal separators, date formats, or measurement units)

- Replace unrecognizable or corrupted characters

- Check for truncated entries (data entries that are cut off at a certain position)

### Correct inconsistencies

- Check for invalid values (age is 200 or negative)

- Check for extreme values in numerical data

- Check for wrong categories in categorical data (similar products should not be put into different categories)

### Handle errors in variables

- Missing Data: can happen due to forgotten to store, inappropriate data handling, inefficient data entry at the ground level, etc. 

- High Cardinality: the number of different labels in categorical data is very high which causes problems to model to learn.

- Outliers: the extreme cases that may be due to error, but not in every case.


-----


## Data Cleaning

Data cleaning refers to identifying and correcting errors in the dataset that may negatively impact a predictive model.

- Identify Columns That Contain a Single Value
- Delete Columns That Contain a Single Value
- Consider Columns That Have Very Few Values
- Remove Columns That Have A Low Variance
- Identify Rows that Contain Duplicate Data
- Delete Rows that Contain Duplicate Data

Data cleaning also includes the following [7]:

- Delete Unnecessary Columns
- Check data types
- Handle missing values
- Handle duplicate values
- Handle outliers
- Handle categorical data
- Encoding class labels
- Parsing dates
- Character encodings
- Inconsistent data entry
- Scaling and normalization


The article [9] provides some Tips and Tricks to Deal with a Messy Date String Column in Pandas Dataframe. 

### Delete Unnecessary Columns

There can be columns in the dataset that we do not need in our data analysis, so we can remove them using the `drop()` method with the specified column name [17]. 

```py
  df.drop('last_name', axis = 1, inplace = True)
```

We set the axis to 1 to specify that we want to delete a column and the inplace argument is set to True so that we modify the existing DataFrame to avoid creating a new DataFrame without the removed column.

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

Sometimes, data types might not be correct. For example, a date column might be interpreted as strings [17]. 

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

### Handle Missing Values

There are various ways of dealing with missing values and it is likely that we will need to determine which method is right for a task at hand on a case-by-case basis [10] [14]. 

The removal of samples or dropping of feature columns may not feasible because we might lose too much valuable data. 

We can use interpolation techniques to estimate the missing values from the other training samples in the dataset.

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


## Encoding Categorical Features

Most machine learning algorithms and deep learning neural networks require that input and output variables are numbers [11] which means that categorical data must be encoded to numbers before we can use it to fit and evaluate a model.

For categorical data, we need to distinguish between _nominal_ and _ordinal_ features. 

Ordinal features can be understood as categorical values that can be sorted or ordered. For example, T-shirt size would be an ordinal feature because we can define an order XL > L > M. 

Nominal features do not imply any order. Thus, T-shirt color is a nominal feature since it typically does not make sense to say that red is larger than blue.

There are many ways to encode categorical variables:

  1. Integer (Ordinal) Encoding: each unique label/category is mapped to an integer.
  
  2. One Hot Encoding: each label is mapped to a binary vector.
  
  3. Dummy Variable Encoding
  
  4. Learned Embedding: a distributed representation of the categories is learned.

### Integer (Ordinal) Encoding

To make sure that the ML algorithm interprets the ordinal features correctly, we need to convert the categorical string values into integers. 

Unfortunately, there is no convenient function that can automatically derive the correct order of the labels of our size feature. 

Thus, we have to define the mapping manually.

```py
    size_mapping = { 'XL': 3, 'L': 2, 'M': 1}
    df['size'] = df['size'].map(size_mapping)
```

### Encoding Class Labels

Many machine learning libraries require that class labels are encoded as integer values. 

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

### One-Hot Encoding of Nominal Features

A one-hot encoding is a type of encoding in which an element of a finite set is represented by the index in that set where only one element has its index set to “1” and all other elements are assigned indices within the range [0, n-1]. 

In contrast to binary encoding schemes where each bit can represent 2 values (0 and 1), one-hot encoding assigns a unique value to each possible value.

In the previous section, we used a simple dictionary-mapping approach to convert the ordinal size feature into integers. 

Since scikit-learn's estimators treat class labels without any order, we can use the convenient `LabelEncoder` class to encode the string labels into integers.

```py
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
```

After executing the code above, the first column of the NumPy array X now holds the new color values which are encoded as follows: blue = 0, green = 1, red = 2 n

However, we will make one of the most common mistakes in dealing with categorical data. Although the color values are not ordered, a ML algorithm will now assume that green is larger than blue, and red is larger than green. Thus, the results would not be optimal.

A common workaround is to use a technique called _one-hot encoding_ to create a new dummy feature for each unique value in the nominal feature column. 

Here, we would convert the color feature into three new features: blue, green, and red. 

Binary values can then be used to indicate the particular color of a sample; for example, a blue sample can be encodedas blue=1, green=0, red=0. 

We can use the `OneHotEncoder` that is implemented in the scikit-learn.preprocessing module. 

An even more convenient way to create those dummy features via one-hot encoding is to use the get_dummies method implemented in pandas. Applied on a DataFrame, the get_dummies method will only convert string columns and leave all other columns unchanged:

```py
    pd.get_dummies(df[['price', 'color', 'size']])
```

### Dummy Variable Encoding

Most machine learning algorithms cannot directly handle categorical features that are _text values_.

Therefore, we need to create dummy variables for our categorical features which is called _one-hot encoding_.

The one-hot encoding creates one binary variable for each category which includes redundancy. 

In contrast, a dummy variable encoding represents N categories with N-1 binary variables.

```py
    pd.get_dummies(df, columns=['Color'], prefix=['Color'])
```

In addition to being slightly less redundant, a dummy variable representation is required for some models such as linear regression model (and other regression models that have a bias term) since a one hot encoding will cause the matrix of input data to become singular which means it cannot be inverted, so the linear regression coefficients cannot be calculated using linear algebra. Therefore, a dummy variable encoding must be used.

However, we rarely encounter this problem in practice when evaluating machine learning algorithms other than linear regression.


### One-Hot Encoding Example

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

## Parsing dates

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


## Inconsistent Data Entry

TODO: This will most likely vary 


## Imbalanced Datasets

Imbalanced data occurs when there is an uneven distribution of classes or labels [10].

Models trained with imbalanced data usually have high precision and recall scores for the majority class, whereas these scores will likely drop significantly for the minority class.

In a credit card detection task, the number of non-fraudulent transactions will likely be much greater than the number of fraudulent credit card transactions.


Need to upsample but categories with only 1 entry when oversampled will give a 100% accuracy and artificially inflate the total accuracy/precision.

- We can use `UpSample` in Keras/PyTorch and `pd.resample()` in Pandas


----------



## Train-Test Split

Also see [Train-Test Split](./ml/train_teat_split.md)

A key step in ML is the choice of model.  

> Split first, normalize later.

A train-test split conists of the following:

1. Split the dataset into training, validation and test set

2. We normalize the training set only (fit_transform). 

3. We normalize the validation and test sets using the normalization factors from train set (transform).


> Instead of discarding the allocated test data after model training and evaluation, it is a good idea to retrain a classifier on the entire dataset for optimal performance.


----------


## Data Pipelines

[Data Pipelines](./pipelines.md)

There are multiple stages to running machine learning algorithms since it involves a sequence of tasks including pre-processing, feature extraction, model fitting, performance, and validation.



## Bootstrapping

The goal of bootstrap is to create an estimate (sample mean x̄) for a population parameter (population mean θ) based on multiple data samples obtained from the original sample.

Bootstrapping is done by repeatedly sampling (with replacement) the sample dataset to create many simulated samples. 

Each simulated bootstrap sample is used to calculate an estimate of the parameter and the estimates are then combined to form a sampling distribution.

The bootstrap sampling distribution then allows us to draw statistical inferences such as estimating the standard error of the parameter.


----------



## Code Examples and References

### Data Preparation

[Data Science Primer](https://elitedatascience.com/primer)

[Data Preparation for Machine Learning (Python)](https://machinelearningmastery.com/start-here/#dataprep)

[Tour of Data Preparation Techniques for Machine Learning](https://machinelearningmastery.com/data-preparation-techniques-for-machine-learning/)

[How to Perform Data Cleaning for Machine Learning with Python?](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)


[Preprocessing of the data using Pandas and SciKit](https://mclguide.readthedocs.io/en/latest/sklearn/preprocessing.html)

[Missing Values Be Gone](https://towardsdatascience.com/missing-values-be-gone-a135c31f87c1?source=rss----7f60cf5620c9---4&gi=d11a8ff041dd)

[ML Guide Quick Reference](https://mclguide.readthedocs.io/en/latest/sklearn/guide.html)

[The Lazy Data Scientist’s Guide to AI/ML Troubleshooting](https://medium.com/@ODSC/the-lazy-data-scientists-guide-to-ai-ml-troubleshooting-abaf20479317?source=linkShare-d5796c2c39d5-1638394993&_branch_referrer=H4sIAAAAAAAAA8soKSkottLXz8nMy9bLTU3JLM3VS87P1Xcxy8xID4gMc8lJAgCSs4wwIwAAAA%3D%3D&_branch_match_id=994707642716437243)


[How to Select a Data Splitting Method](https://towardsdatascience.com/how-to-select-a-data-splitting-method-4cf6bc6991da)


### Categorical Data

[Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)

[3 Ways to Encode Categorical Variables for Deep Learning](https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/)


[4 Categorical Encoding Concepts to Know for Data Scientists](https://towardsdatascience.com/4-categorical-encoding-concepts-to-know-for-data-scientists-e144851c6383)

[Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)

[Stop One-Hot Encoding Your Categorical Variables](https://towardsdatascience.com/stop-one-hot-encoding-your-categorical-variables-bbb0fba89809)


### Scaling

[How to Selectively Scale Numerical Input Variables for Machine Learning](https://machinelearningmastery.com/selectively-scale-numerical-input-variables-for-machine-learning/)

[How to use Data Scaling Improve Deep Learning Model Stability and Performance](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)

[How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)

[The Mystery of Feature Scaling is Finally Solved](https://towardsdatascience.com/the-mystery-of-feature-scaling-is-finally-solved-29a7bb58efc2?source=rss----7f60cf5620c9---4)


### Normalization

[How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)

[How to Use Power Transforms for Machine Learning](https://machinelearningmastery.com/power-transforms-with-scikit-learn/)


### Train-Test Split

[Training-validation-test split and cross-validation done right](https://machinelearningmastery.com/training-validation-test-split-and-cross-validation-done-right/)

[A Gentle Introduction to k-fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)

[How to Configure k-Fold Cross-Validation](https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/)


## Data Sampling

[Data Sampling Methods in Python](https://towardsdatascience.com/data-sampling-methods-in-python-a4400628ea1b)

[Common Data Problems (and Solutions)](https://www.kdnuggets.com/2022/02/common-data-problems-solutions.html)



## References

W. McKinney, Python for Data Analysis, 2nd ed., Oreilly, ISBN: 978-1-491-95766-0, 2018.

[1] [Read datasets with URL](https://towardsdatascience.com/dont-download-read-datasets-with-url-in-python-8245a5eaa919)

[2] [13 ways to access data in Python](https://towardsdatascience.com/13-ways-to-access-data-in-python-bac5683e0063)


[3] [INFOGRAPHIC: Data prep and Labeling](https://www.cognilytica.com/2019/04/19/infographic-data-prep-and-labeling/)

[4] [Kaggle Data Cleaning Challenge: Missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)


[6] [A Better Way for Data Preprocessing: Pandas Pipe](https://towardsdatascience.com/a-better-way-for-data-preprocessing-pandas-pipe-a08336a012bc)

[7] [Introduction to Scikit-learn’s Pipelines](https://towardsdatascience.com/introduction-to-scikit-learns-pipelines-565cc549754a)


[8] [Refactoring for Scalable Python Code With Pandas](https://betterprogramming.pub/refactoring-for-scalable-python-code-with-pandas-727d15f14852)

[9] [Clean a Messy Date Column with Mixed Formats in Pandas](https://towardsdatascience.com/clean-a-messy-date-column-with-mixed-formats-in-pandas-1a88808edbf7)


[10] [Major Problems of Machine Learning Datasets: Part 1](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-1-5d5a06221c90)

[11] [Major Problems of Machine Learning Datasets: Part 2](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-2-ba82e551fee2)

[12] [Major Problems of Machine Learning Datasets: Part 3](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-3-eae18ab40eda)

[13] [ELT vs ETL: Why not both?](https://medium.com/geekculture/elt-vs-etl-why-not-both-d0c4a0d30fc0)


[14] [How to Detect Missing Values and Dealing with Them: Explained](https://medium.com/geekculture/ow-to-detect-missing-values-and-dealing-with-them-explained-13232230cb64)

[15] [Deduplicate and clean up millions of location records](https://towardsdatascience.com/deduplicate-and-clean-up-millions-of-location-records-abcffb308ebf)

[16] [How to Make Your Data Models Modular](https://towardsdatascience.com/how-to-make-your-data-models-modular-71b21cdf5208)

[17] [Mastering the Art of Data Cleaning in Python](https://www.kdnuggets.com/mastering-the-art-of-data-cleaning-in-python)


[Build an Anomaly Detection Pipeline with Isolation Forest and Kedro](https://towardsdatascience.com/build-an-anomaly-detection-pipeline-with-isolation-forest-and-kedro-db5f4437bfab)

[6 Tips for Dealing With Null Values](https://towardsdatascience.com/6-tips-for-dealing-with-null-values-e16d1d1a1b33)

[Customizing Sklearn Pipelines: TransformerMixin](https://towardsdatascience.com/customizing-sklearn-pipelines-transformermixin-a54341d8d624?source=rss----7f60cf5620c9---4)


