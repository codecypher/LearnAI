# Exploratory Data Analysis (EDA)

## Summary Statistics

It is important to know how to extract information from descriptive statistics.

```py
    # get the data info
    df.info()

    # summary stats
    df.describe()

    # inspect the data
    df.head()

    df.value_counts()

    # Get value counts for our target variable
    df['income'].value_counts()

    # include string and categorical features
    df.describe(include=['int', 'float', 'object', 'category'])
```

## Python Graphing Libraries

Here are some useful Python graphing libraries [17]:

Altair: Declarative Visualization Made Simple
 
Altair is a declarative statistical visualization library focusing on simplicity and expressiveness that minimizes boilerplate code and emphasizes interactive charts.

DuckDB: High-Performance SQL OLAP
 
DuckDB is an in-process SQL OLAP database optimized for analytical workload which allows seamless integration with Python tools like Pandas and Jupyter.

FlashText: Efficient Text Search and Replacement

FlashText is a lightweight library for keyword extraction and replacement, outperforming regex in speed and simplicity for many use cases.

Missingno: Visualizing Missing Data

Missingno provides quick and intuitive visualizations for missing data, helping identify patterns and correlations.

NetworkX: Analyzing Graph Data
 
NetworkX is a versatile library for analyzing and visualizing graph structures from social networks to transportation systems.

Ydata Profiling: Automated Data Insights
 
Ydata Profiling automates dataset exploration by generating detailed HTML reports that highlight distributions, correlations, and data quality.


## Exploratory Data Analysis

Exploratory Data Analysis (EDA) is one of the first steps of the data science process which involves learning as much as possible about the data without spending too much time.

The basic steps are the following (see [Data Preparation](./data_prep.md)):

1. Basic Exploration
2. Check Data Types
3. Handle Missing Values
4. Handle Duplicate Values
5. Handle Outliers
6. Visualize the Data

We can get an instinctive as well as a high-level practical understanding of the data including a general idea of the structure of the data set, some cleaning ideas, the target variable and possible modeling techniques.

We can display the summary statistics and create histograms for the numeric variables of the dataset.

```py
    # check the shape of the dataframe
    df.shape

    # Print a concise summary of a DataFrame.
    # print information about a DataFrame including the index dtype
    # and columns, non-null values, and memory usage.
    df.info()

    # summary statistics fir numeric data
    df.describe()


    # get categorical data
    cat_data = df.select_dtypes(include=['object'])

    # show counts values of each categorical variable
    for colname in cat_data.columns:
        print (colname)
        print (cat_data[colname].value_counts(), '\n')


    # check data types
    df.dtypes

    df.size

    # get column list
    df.columns.tolist()


    # check the distribution of categorical columns
    df["product_group"].value_counts()

    # find percent share of each value by using the normalize parameter
    df["product_group"].value_counts(normalize=True)

    # check the average price of products for each product group
    df.groupby("product_group", as_index=False).agg(
        avg_price = ("price","mean")
    )


    # change column data type
    df[['age', 'weight']] = df[['age', 'weight']].astype(float)


    # Set new column value based on multiple criteria
    # Use bitwise operators instead of AND and OR
    df.loc[(df.AvgProduction> 1000000) & (df.Age > 5), 'Category'] = 'Priority 1'


    # Check for missing timestamps or rows
    time_range = pd.date_range(startdate , enddate, freq='1min')
    ts =   pd.DataFrame(time_range)
    ts.rename(columns = {ts.columns[0]:'timestamp'}, inplace = True)
    ## now complete a merge to join the sets together


    # Filter data based on string match
    df1 = df[df[‘Flag’].str.contains(“CHECK ME NOW”)]

    # Aggregate across columns
    df['StateAverage'] = df_mo[['school1', 'school2','school3', 'school4']].mean(axis=1)


    pd.DataFrame({"values":{col:df[col].unique() for col in df},
              'type':{col:df[col].dtype for col in df},
              'unique values':{col:len(df[col].unique()) for col in df},
              'NA values':{col:str(round(sum(df[col].isna())/len(df),2))+'%' for col in df},
              'Duplicated Values':{col:sum(df[col].duplicated()) for col in df}
    })
```

```py
    numeric_variables = list(df.select_dtypes(include=['int64', 'float64'])) #select the numeric variables

    df[numeric_variables].describe().apply(lambda x:round(x,2)).T #apply describe method

    # create the histograms
    histograms = df[numeric_variables].hist(bins =10,
        xlabelsize=10,
        ylabelsize=10,
        grid=False,
        sharey= True,
        figsize = (15,15))
```

```py
    # numerical features
    df.describe()

    # include string and categorical features
    df.describe(include=['int', 'float', 'object', 'category'])
    # unique = number of unique categories
    # top = dominant category
    # freq = count of dominant category
```

### Visualize the Data

We can use univariate plots or plots such as histograms and boxplots on single variables. Then, we can use bivariate (or multi-variate) plots across different variables.

- Univariate Plots
- Multivariate Plots
- Correlation Matrix or Heatmap
- Pair Plots

```py
    plt.figure(figsize=(7,6))
    sns.boxplot(x="y", y="duration", data=df, showfliers=True)

    sns.histplot(x='age', data=df, bins=20, kde=True)
```

```py
    # correlation matrix shows the correlation
    # between all the variables in the dataset.
    corr = df.corr()
    f, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, square=False, ax=ax, linewidth = 1)
    plt.title('Pearson Correlation of Features')
```

Pair Plots is to a correlation matrix, but gives a scatterplot for each of the X and Y pairs.

```py
    g.fig.set_size_inches(12,12)
    g=sns.pairplot(df, diag_kind = 'auto', hue="y")
```


## Guide to EDA and Data Preparation

Here is a guide to the fundamental pre-processing techniques and how to address some common problems [15]. 

```py
# Load the dataset
df = pd.read_csv("file.csv")

# Check the data type of the dataset
type(df)

# Check the shape of the dataset
df.shape

# Display column names
df.columns

# Display the first few rows of dataset
df.head()

# Display random sample of 10 rows from dataset
df.sample(10)

# Display dataset information
df.info()

# Describe non-numeric data
df.describe(include=object)

# Describing numeric data
df.describe()

# Extract numeric columns using list comprehension
selected_columns = [list(df.columns.values)[i] for i in [0, 1, 4]]

# Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Salary Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Salary'], kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.show()

# Gender Distribution (categorical)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Gender')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Compute correlation matrix
correlation_matrix = df[['Age', 'Salary', 'Exam_Score']].corr()

# Visualize the correlation matrix with heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation between Quantitative Variables')
plt.show()

# Compute contingency table
contingency_table = pd.crosstab(df['Gender'], df['Education_Level'])

# Duplicate Values
duplicates = df.duplicated()

# Display the duplicate rows
df[duplicates]

# Check for negative values in Salary
df[df['Salary'] < 0]

# Replace negative values with NaN (missing values)
df['Salary'] = df['Salary'].apply(lambda x: x if x >= 0 else None)


# Using isna() method to check for missing values in each column
missing_values = df.isna().sum()
print(missing_values)

# Calculate the percentage of missing values in each column
missing_percentage = (df.isna().mean() * 100).round(2)

# Display the percentage of missing values
print(missing_percentage)
```

- For quantitative variables, we use correlation or Pearson correlation.

- For qualitative variables, we study association.

We can use methods such as ANOVA (Analysis of Variance) To study the relationship between a quantitative and a qualitative variable. 

Association Between Qualitative Variables — Chi-Squared Test, Cramer’s V Test



## Essential Code Blocks

1. Shape (dimensions) of the DataFrame
2. Data types of the various columns
3. Display a few rows

We may observe that our dataset has a combination of categorical (object) and numeric (float and int) features.

What to look for:

- Can you understand the column names? Do they make sense? (Check the variable definitions if needed)

- Do the values in the columns make sense? Numeric features that should be categorical and vice versa.

- Are there significant missing values (NaN)?

- What types of classes do the categorical features have?

### Distribution

Distribution refers to how the values in a feature are distributed or how often they occur.

For numeric features, we see how many times groups of numbers appear in a particular column.

For categorical features, we view the classes for each column and their frequency.

We will use both graphs and actual summary statistics.

The graphs enable us to get an overall idea of the distributions while the statistics give us factual numbers.

Both graphs and statistics are recommended since they complement each other.

### Summary statistics of numerical features

```py
    print(df.describe())

    # count null values
    df.isnull().sum()

    # count of students whose physics marks are greater than 11
    df[df['Physics'] > 11]['Name'].count())

    # count students whose physics marks are greater than 10
    # and math marks are greater than 9.
    df[(df['Physics'] > 10 ) & (df['Math'] > 9)]

    # Multi-column frequency count
    count = df.groupby(col_name).count()
```

We can see for each numeric feature, the count of valueS, the mean value, std or standard deviation, minimum value, the 25th percentile, the 50th percentile or median, the 75th percentile, and the maximum value.

What to look for:

- Missing values: their count is not equal to the total number of rows of the dataset.
- Minimum or maximum values they do not make sense.
- Large range in values (min/max)

### Summary statistics of categorical features

```py
   df.describe(include=['category'])
```

### Plot of numeric features

```py
    df.hist(figsize=(14,14), xrot=45)
    plt.show()
```

What to look for:

- Possible outliers that cannot be explained or might be measurement errors.

- Numeric features that should be categorical such as Gender represented by 1 and 0.

- Boundaries that do not make sense such as percentage values > 100.

### Plot of categorical features

```py
    for column in df.select_dtypes(include='object'):
        if df[column].nunique() < 10:
            sns.countplot(y=column, data=df)
    plt.show()
```

What to look for:

- Sparse classes which have the potential to affect a model’s performance.
- Mistakes in labeling of the classes, for example 2 exact classes with minor spelling differences.


### Grouping and segmentation

Segmentation allows us to cut the data and observe the relationship between categorical and numeric features.

#### Segment the target variable by categorical features

Compare the _target_ feature (Price) between the various classes of our main categorical features (Type, Method, and Regionname) and see how the target changes with the classes.

```py
    # Plot boxplot of each categorical feature with Price.
    for column in data.select_dtypes(include=’object’):
        if data[column].nunique() < 10:
            sns.boxplot(y=column, x=’Price’, data=data)
    plt.show()
```

What to look for: Classes that most affect the target variable(s).


#### Group numeric features by each categorical feature

See how all the other numeric features (not just target feature) change with each categorical feature by summarizing the numeric features across the classes.

```py
    # For the 3 categorical features with less than 10 classes,
    # we group the data, then calculate the mean across the numeric features.
    for column in df.select_dtypes(include='object'):
        if df[column].nunique() < 10:
            display(df.groupby(column).mean())
```


### Relationships between numeric features

#### Correlation matrix for numerical features

A _correlation_ is a value between -1 and 1 that amounts to how closely values of two separate features move simultaneously.

A _positive_ correlation means that as one feature increases the other one also increases while a _negative_ correlation means one feature increases as the other decreases.

Correlations close to 0 indicate a _weak_ relationship while closer to -1 or 1 signifies a _strong_ relationship.

```py
    corrs = df.corr()
    print(corrs)
```

This might not mean much now, so we can plot a **heatmap** to visualize the correlations.

#### Heatmap of the correlations

```py
    # Plot the grid as a rectangular color-coded matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corrs, cmap='RdBu_r', annot=True)
    plt.show()
```

What to look for:

- Strongly correlated features: either dark red (positive) or dark blue(negative).
- Target variable: If it has strong positive or negative relationship with other features.


----------



## How to Identify Outliers

Many machine learning algorithms are sensitive to the range and distribution of the input data [7].

Outliers in input data can affect the training process of ML algorithms abd result in less accurate models.


Keep in mind that some suspicious points are often wrongly reduced to outliers [8].

In a regression analysis, these terms are related besides they have different meanings:

- **Influential point:** a subset of the data having an influence on the estimated model. Removing it appears to modify significantly the results of the estimated coefficient.

- **Outlier:** an observation that falls outside the general pattern of the rest of the data. In a regression analysis, these points remain out of the confidence bands due to high residual.

- **High Leverage:** an observation that has the potential to influence predominantly the predicted response. It can be assessed through ∂y_hat/∂y: how much the predicted response differs if we shift slightly the observed response. Generally, they are extreme x-values.


### Extreme Value Analysis

Start with a simple extreme value analysis:

- Focus on univariate methods

- Visualize the data using scatterplots, histograms, and box and whisker plots to identify extreme values

- Assume a normal distribution (Gaussian) and look for values more than 2 or 3 standard deviations from the mean or 1.5 times from the first or third quartile

- Filter out outliers from training set and evaluate model performance

### Proximity Methods

Next, consider trying proximity-based methods:

- Use clustering methods to identify clusters in the data (such as k-Means and DBSCAN)

- Identify and mark the cluster centroids

- Identify data instances that are a fixed distance or percentage distance from the cluster centroids

- Filter out outliers from training set and evaluate model performance

### Projection Methods

Finally, projection methods can be used to  identify outliers:

- Use projection methods to summarize your data to two dimensions (such as PCA, MDS, and t-SNE)

- Visualize the mapping and identify outliers

- Use proximity measures from projected values or vectors to identify outliers

- Filter out outliers from training set and evaluate model performance

### Methods Robust to Outliers

An alternative approach is to try models that are robust to outliers.

There are robust forms of regression that minimize the median least square errors rather than mean called robust regression but they are more computationally intensive.

There are also models such as decision trees that are robust to outliers.

Spot check some methods that are robust to outliers to see if there is a significant improvement in model performance metrics.


## Plotly Express

The `plotly.express` module contains functions that can create entire figures at once.

Plotly Express is a built-in part of the `plotly` library and is the recommended starting point for creating most common figures.

We can use `plotly.express` scatter or line charts to quickly run a linear regression between 1 variable and 1 target [16].

### Time series data without Trendline

Time series data typically comes with a timestamp column (eg Datetime) and the value you want to plot (listed below as ‘y’) [16].

```py
    # Line plot
    px.line(df, x='Datetime', y='y')

    # Scatter plot
    px.scatter(df, x='Datetime', y='y')
```

### Correlated variables with Trendline

Using the trendline keyword in `px.scatter` draws a linear regression trendline with your x and y variables using OLS (Ordinary Least Squares) regression algorithm which is basically just a standard linear regression [16].

Here is an example using the Wine Quality dataset from the UCI Machine Learning Repository (CC by 4.0 license) where we have plotted 2 features against each other to see the relationship between them [16].

The OLS trendline provides you with the y=mx+b linear equation as well as the R².

```py
    # Scatter plot with trendline
    px.scatter(df, x='fixed_acidity',y='density',trendline='ols')
```

### Statistical Distribution

#### Mean

With the mean value, you are trying to get a sense of what an average data point looks like.

#### Standard Deviation

Standard deviation is a measure of variation/dispersion of data points with respect to the mean.

Smaller STD indicates that the data are mostly centered around the mean whereas a higher STD value indicates the data points are rather dispersed.

#### Median (50%)

The 50th percentile (the 50% column) is also known as the median. Like mean, it’s another measure of central tendency.

Median is a preferred metric rather than mean if there are outliers or high variability in the data.

If the difference between mean and median is _small_, you can infer that the data is symmetrically distributed.

If the median is higher than the mean, data is likely _left-skewed_ in distribution.

#### Min and Max

Min and max values represent the lower and upper limit of a variable in the dataset.


### Anomalies

You can get a sense of outliers, anomalies, and other points of interest in the dataset using descriptive statistics.

#### Outliers

A large difference between the 75th percentile and the maximum value indicates the presence of potential outliers.

Likewise, a large difference between the minimum value and the 25th percentile indicates the presence of potential outliers.

To confirm outliers you can create a boxplot for visual inspection:

```py
    sns.boxplot(y=df['total_bill']);
```

#### Red flags

Sometimes descriptive statistics can raise red flags.

Places with unexpected minimum values (0 or negative) or absolutely unacceptible maximum values (such as someone’s age 120 years!).

These are obvious indications that there are issues in the data and need further investigation.




## References

[1]: [Reading and interpreting summary statistics](https://towardsdatascience.com/reading-and-interpreting-summary-statistics-df34f4e69ba6)

[2]: [Python Cheat Sheet for Data Science](https://chipnetics.com/tutorials/python-cheat-sheet-for-data-science/)

[3]: [Data Cleaning for Visualization versus Machine Learning](https://medium.com/geekculture/data-cleaning-for-visualization-versus-machine-learning-e66126ebb65c)


[4]: [11 Essential Code Blocks for EDA Regression Task](https://towardsdatascience.com/11-simple-code-blocks-for-complete-exploratory-data-analysis-eda-67c2817f56cd)

[5]: [6 Pandas Functions for a Quick Exploratory Data Analysis](https://sonery.medium.com/6-pandas-functions-for-a-quick-exploratory-data-analysis-ff9ece0867d7)

[6]: [My Top 10 Pandas Functions for Preparing Data](https://betterprogramming.pub/my-top-10-pandas-functions-for-preparing-data-3ec7a1451a84)

[7]: [How to Identify Outliers in your Data](https://machinelearningmastery.com/how-to-identify-outliers-in-your-data/)

[8]: [Do Not Dismiss Unusual Data Points as Outliers](https://towardsdatascience.com/do-not-dismiss-unusual-data-points-as-outliers-5132380f2e67)

[9]: [A recipe to empirically answer any question quickly](https://towardsdatascience.com/a-recipe-to-empirically-answer-any-question-quickly-22e48c867dd5)

[10]: [Major Problems of Machine Learning Datasets: Part 2](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-2-ba82e551fee2)

[11]: [Explore and Validate Datasets with TensorFlow Extended](https://pub.towardsai.net/explore-and-validate-datasets-with-tensorflow-extended-fc52cc5e582)

[12]: [Advanced EDA Made Simple Using Pandas Profiling](https://pub.towardsai.net/advanced-eda-made-simple-using-pandas-profiling-35f83027061a)


[13]: [Detecting Outliers Using Python](https://towardsdatascience.com/detecting-outliers-using-python-66b25fc66e67)

[14]: [Detecting Outliers with Simple and Advanced Techniques](https://towardsdatascience.com/detecting-outliers-with-simple-and-advanced-techniques-cb3b2db60d03)

[15]: [Practical Guide to Data Analysis and Preprocessing](https://towardsdatascience.com/practical-guide-to-data-analysis-and-preprocessing-080815548173)

[16]: [5 Python One-Liners to Kick Off Your Data Exploration](https://towardsdatascience.com/5-python-one-liners-to-kick-off-your-data-exploration-d6221f94291e)

[17]: [10 Little-Known Python Libraries That Will Make You Feel Like a Data Wizard](https://www.kdnuggets.com/10-little-known-python-libraries-data-wizard)


[Data Analytics: The Four Approaches to Analyzing Data and How To Use Them Effectively](https://www.kdnuggets.com/2023/04/data-analytics-four-approaches-analyzing-data-effectively.html)

[How to build a Machine Learning (ML) Predictive System](https://towardsdatascience.com/machine-learning-ml-based-predictive-system-to-predict-the-satisfaction-level-of-airlines-f0780dbdbc87?source=rss----7f60cf5620c9---4)

[Huber and Ridge Regressions in Python: Dealing with Outliers](https://towardsdatascience.com/huber-and-ridge-regressions-in-python-dealing-with-outliers-dc4fc0ac32e4?gi=c292a23ceab7)

[Use k-medians clustering method if you have many outliers](https://towardsdatascience.com/use-this-clustering-method-if-you-have-many-outliers-5c99b4cd380d?gi=5c97a562ff04)

[Sensitivity Analysis of Dataset Size vs Model Performance](https://machinelearningmastery.com/sensitivity-analysis-of-dataset-size-vs-model-performance/)

[Analysing Fairness in Machine Learning (with Python)](https://towardsdatascience.com/analysing-fairness-in-machine-learning-with-python-96a9ab0d0705)
