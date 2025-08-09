# Pandas

## Basics

Here are some tips on using Pandas [18] to [21].

```py
  # Load a dataset (CSV file)
  titanic_df = pd.read_csv('titanic.csv')
  df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
  

  # Export the DataFrame to a CSV file
  df.to_csv('output_dataset.csv', index=False)

  # Display basic information
  df.info()

  # Display the first few rows
  titanic_df.head()

  # Generate descriptive statistics
  df.describe()
  titanic_df.describe(include = 'all')
```

`describe()` provides an overview of key statistics such as mean, standard deviation, and quartiles for numerical columns.

Adding “include = all” shows the summary for qualitative (string/object variables).

```py
  # find missing values
  titanic_df.isnull().sum()

  # Fill missing values with a specific value
  titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())

  # Filter data based on a condition
  titanic_df.loc[titanic_df['Age'] > 30]

  # Sort data by a specific column
  titanic_df_sorted =   titanic_df.sort_values(by='Fare')
```

```py
  # Convert a column to datetime format
  df['Date'] = pd.to_datetime(df['Date'])

  # Extract month from the 'Date' column
  pdf['Month'] = df['Date'].dt.month

  # Remove duplicate rows based on selected columns
  df_no_duplicates = titanic_df.drop_duplicates(subset=['PassengerId'])

  # Rename columns for clarity
  titanic_df.rename(columns={'SibSp': 'sibbling_spouse'}, inplace=True)

  # Convert the 'Price' column to numeric
  df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

  # Apply a custom function to a column
  df['Discounted_Price'] = df['Price'].apply(lambda x: x * 0.9)

  # Convert a column to categorical type
  df['Category'] = df['Category'].astype('category')
```

```py
  # Group data by a categorical variable and calculate the mean
  titanic_df.groupby('Sex')['Survived'].mean()

  # Create new column based on existing columns
  titanic_df['total_relative'] = titanic_df['SibSp'] + titanic_df['Parch']

  # Merge two DataFrames based on a common column
  merged_df = pd.merge(df1, df2, on='ID')

  # Pivot the data to reshape it
  titanic_df_pivot = titanic_df.pivot_table(index='Survived', columns='Sex', values='Age', aggfunc='mean')
```


### Automatically Convert to Best Data Types

When we load data as Pandas dataframe, Pandas automatically assigns a datatype to the variables/columns in the dataframe, typically the datatypes would be int, float and object datatypes. With the recent Pandas 1.0.0, we can make Pandas infer the best datatypes for the variables in a dataframe.

We will use the Pandas `convert_dtypes()` function and convert the to best data types automatically. Another big advantage of using convert_dtypes() is that it supports Pandas new type for missing values `pd.NA`.

```py
    import pandas as pd

    # check version
    print(pd.__version__)

    data_url = "https://raw.githubusercontent.com/cmdlinetips/data/master/gapminder-FiveYearData.csv"
    df = pd.read_csv(data_url)

    print(df.info())


    print(df.dtypes)

    df = df.convert_dtypes()
    print(df.dtypes)
```

By default, `convert_dtypes` will attempt to convert a Series (or each Series in a DataFrame) to dtypes that support `pd.NA`.

By using the options convert_string, convert_integer, and convert_boolean, it is possible to turn off individual conversions to StringDtype, the integer extension types, or BooleanDtype, respectively.


### Dates

[11 Essential Tricks To Demystify Dates in Pandas](https://towardsdatascience.com/11-essential-tricks-to-demystify-dates-in-pandas-8644ec591cf1)

[Dealing With Dates in Pandas](https://towardsdatascience.com/dealing-with-dates-in-pandas-6-common-operations-you-should-know-1ea6057c6f4f)


### Iteration

[How To Loop Through Pandas Rows](https://cmdlinetips.com/2018/12/how-to-loop-through-pandas-rows-or-how-to-iterate-over-pandas-rows/amp/)


### String

[String Operations on Pandas DataFrame](https://blog.devgenius.io/string-operations-on-pandas-dataframe-88af220439d1)


### Indexes

[How To Convert a Column to Row Name/Index in Pandas](https://cmdlinetips.com/2018/09/how-to-convert-a-column-to-row-name-index-in-pandas/amp/)

[8 Quick Tips on Manipulating Index with Pandas](https://towardsdatascience.com/8-quick-tips-on-manipulating-index-with-pandas-c10ef9d1b44f)


### Functions

[apply() vs map() vs applymap() in Pandas](https://towardsdatascience.com/apply-vs-map-vs-applymap-pandas-529acdf6d744)

[How to Combine Data in Pandas](https://towardsdatascience.com/how-to-combine-data-in-pandas-5-functions-you-should-know-651ac71a94d6)


### Aggregate

[6 Lesser-Known Pandas Aggregate Functions](https://towardsdatascience.com/6-lesser-known-pandas-aggregate-functions-c9831b366f21)

[Pandas Groupby and Sum](https://cmdlinetips.com/2020/07/pandas-groupby-and-sum/amp/)


### Pivot

[5 Minute Guide to Pandas Pivot Tables](https://towardsdatascience.com/5-minute-guide-to-pandas-pivot-tables-df2d02786886)



----------



## How to Speedup Pandas

### 1. Use itertuples() instead of iterrows()

Data manipulation often requires iterating over DataFrame rows.

```py
    answers_list : list[int] = []
    for index, row in df.iterrows():
        answers_list.append(simple_sum(row["col1"], row["col2"]))
```

However, `iterrows()` is often the go-to option for such use cases. However, it is very slow and can be easily swapped by `itertuples()`.

```py
    answers_list : list[int] = []
    for row in df.itertuples():
        answers_list.append(simple_sum(row.col1, row.col2))
```

### 2. Appending new rows efficiently:

Consider the same task as above i.e. adding two columns of a dataframe. Only this time, the goal is to create a new dataframe containing original data along with a column of summed values.

An inefficient way of accomplishing the task would involve initializing a new dataframe and appending new rows to it from a loop. For a dataframe with 100k rows, it takes roughly 56s to complete the task.

Appending rows to a dataframe in a loop is an extremely bad idea.

A better alternative would be to accumulate the results in a `list` and create a new dataframe from the list.

```py
    new_list = []

    for row in df.itertuples():

        new_row = {"col1": [row.col1], "col2": [row.col2], "col3": [simple_sum(row.col1, row.col2)]}
        new_list.append(new_row)

    new_df = pd.DataFrame(new_list)
```

### 3. apply() is just a glorified for loop

A more traditional way of applying a function to dataframe rows involves using the `apply()` method which uses a loop with an added overhead.

This can often be avoided by leveraging vectorized operations.

```py
    def conditional_multiplication(a: int) -> int:
        """
        Multiply by 2 if input is > 1000 else multiply by 3
        """
        if a >= 1000:
            return a * 2
        else:
            return a * 3

    # A suboptimal solution
    df["conditional_mul_result"] = df["col2"].apply(conditional_multiplication)
```


```py
    # Here we leverage NumPy vector operations
    df["conditional_mul_result_optimized"] = np.where(df["col2"] >= 1000, df["col2"]* 2, df["col2"] * 3)
```



## Performance Tips

### Using at in place of loc

Using `loc` and `iloc` inside loops is not optimal. Instead, we should use `at` and `iat` which are much faster than `loc` and `iloc` [14].

The `at` and `iat` methods are used to access a scalar which is a single element in the DataFrame.

```py
    import time

    start = time.time()

    # Iterating through DataFrame
    for index, row in df.iterrows():
        df.at[index,'c'] = row.a + row.b

    end = time.time()
    print(end - start)

    ### Time taken: 40 seconds
```

The `loc` and `iloc` functions are used to access multiple elements (series/dataframe) at the same time, potentially to perform vectorized operations.

```py
    import time

    start = time.time()

    # Iterating through the DataFrame df
    for index, row in df.iterrows():
            df.loc[index,'c'] = row.a + row.b

    end = time.time()
    print(end - start)

    ### Time taken: 2414 seconds
```

**NOTE:** If we try to access a series using at and iat, it will throw an error.

```py
    df.at[2,'a']
    ### Output: 22

    df.iat[2,0]
    ### Output: 22

    ## This will generate an error since we are trying to access multiple rows
    df.at[:3,'a']
```

```py
    df.loc[:3,'a']
    ##0    26
    ##1    10
    ##2    22
    ##3    22

    df.loc[:3,0]
    ##0    26
    ##1    10
    ##2    22
    ##3    22
```


Here are some of the best practices for some most common data manipulation operations in Pandas [17].

### Indexing efficiently

For choosing a row or multiple rows, `iloc` is faster [17].

```py
tps = pd.read_csv("data/train.csv")

tps.shape
# (957919, 120)

# Choose rows
tps.iloc[range(10000)]
```

In contrast, `loc` is best for choosing columns with their labels:

```py
tps.loc[:, ["f1", "f2", "f3"]]
```

For sampling columns or rows, the built-in `sample` function is the fastest.

```py
# Sampling rows
tps.sample(7, axis=0)

# Sampling 5 columns and 7 rows
tps.sample(5, axis=1).sample(7, axis=0)
```


### Replacing values efficiently

Many programmers use `loc` or `iloc` to replace specific values in a DataFrame [17].

```py
adult_income = pd.read_csv("data/adult.csv")

# Replacing the “?” values in ‘workclass’
# column with missing values.
adult_income.loc[adult_income["workclass"] == "?", "workclass"] = np.nan
```

This method seems the fastest because we specify the exact location of the value to be replaced instead of letting Pandas search it. However, this method is clumsy and not as fast as `replace`:

```py
# Replacing “?” with np.nan using the replace
adult_income.replace(to_replace="?", value=np.nan, inplace=True)
```

- replace is more flexible since many operations would take multiple calls with index-based replacement.

- replace allows using lists or dictionaries to change multiple values simultaneously:

```py
# Replace multiple items with different values simultaneously.
adult_income.replace(["Male", "Female"], ["M", "F"], inplace=True)
```

When replacing a list of values with another, they will have a one-to-one, index-to-index mapping.
We can be more explicit with dictionaries.

```py
# Using dictionary mapping to replace values.
adult_income.replace({"United States": "USA", "US": "USA"}, inplace=True)
```

It is possible to go even more granular with nested dictionaries:

```py
# replace values only in education and income columns
adult_income.replace(
    {
        "education": {"HS-grad": "High school", "Some-college": "College"},
        "income": {"<=50K": 0, ">50K": 1},
    },
    inplace=True,
    )
```

There are other benefits of replace, including regex-based replacement.

### Iterating efficiently

The rule for applying operations on entire columns or data frames is to **never use loops|l** [17].

The trick is to start thinking about arrays as vectors and the whole data frame as a matrix.

We know in Linear Algebra that mathematical operations work on the entire vector. This idea of operating on mathematical vectors is implemented in Pandas and NumPy as _vectorization_.

We will choose ~1M row dataset of the old Kaggle TPS September competition:

```py
import datatable as dt

tps = dt.fread("data/train.csv").to_pandas()
tps.shape
# (957919, 120)
```

We can perform some mathematical operations on a few columns with Pandas `apply` function which is perhaps the fastest built-in iterator of Pandas.

```py
%time tps['f1000'] = tps.apply(lambda row: crazy_function(
                                row['f1'], row['f56'], row['f44']
                              ), axis=1)
```

But it is much faster if we pass columns as vectors rather than scalars.

```py
%time tps['f1001'] = crazy_function(tps['f1'], tps['f56'], tps['f44'])
```

We cabn add `.values` to get the underlying NumPy ndarray of the Pandas Series sincr NumPy arrays are usually faster.

### More Examples

There are a few more ways to improve performance of Pandas when we have very large datasets (1M rows or more).

We can increase the dataset size ten times to give ourselves a challenge:

```py
massive_df = pd.concat([tps.drop(["f1000", "f1001"], axis=1)] * 10)

massive_df.shape
# (9579190, 120)

memory_usage = massive_df.memory_usage(deep=True)
memory_usage_in_mbs = np.sum(memory_usage / 1024 ** 2)

memory_usage_in_mbs
# 8742.604093551636
```

Here we use the `crazy_function` starting with NumPy vectorization as a baseline:

```py
%%time

massive_df["f1001"] = crazy_function(
    massive_df["f1"].values, massive_df["f56"].values, massive_df["f44"].values
)

# Wall time: 324 ms
```

We can improve the runtime even more using Numba.

We can use the `eval` function of Pandas - pd.eval and df.eval.

```py
%%time

massive_df.eval("f1001 = (f1 ** 3 + f56 ** 2 + f44 * 10) ** 0.5", inplace=True)

# Wall time: 651 ms
```



----------



## Best Format to Save Pandas Data

The article [2] provides a small comparison of various ways to serialize a pandas data frame to the persistent storage.

When the number of observations in your dataset is high, the process of saving and loading data back into the memory becomes slower, and now each kernel’s restart steals your time and forces you to wait until the data reloads. Thus, the CSV files or any other plain-text formats lose their attractiveness.

THe article goes through several methods to save pandas.DataFrame onto disk to see which one is better in terms of I/O speed, consumed memory, and disk space.

### Formats to Compare

We consider the following formats to store our data:

- Plain-text CSV — a good old friend of a data scientist
- Pickle — a Python’s way to serialize things
- MessagePack — it’s like JSON but fast and small
- HDF5 —a file format designed to store and organize large amounts of data
- Feather — a fast, lightweight, and easy-to-use binary file format for storing data frames
- Parquet — an Apache Hadoop’s columnar storage format

All of these formats are widely used and (except for MessagePack) often encountered in practice.

### Chosen Metrics

In order to determine the best buffer format to store the data between notebook sessions, the following metrics were chosen for comparison:

- size_mb: the size of the file (in Mb) with the serialized data frame
- save_time: an amount of time required to save a data frame onto a disk
- load_time: an amount of time needed to load the previously dumped data frame into memory
- save_ram_delta_mb: the maximal memory consumption growth during a data frame saving process
- load_ram_delta_mb: the maximal memory consumption growth during a data frame loading process

Note that the last two metrics become important when we use efficiently compressed binary data formats such as Parquet which could help to estimate the amount of RAM required to load the serialized data, in addition to the data size itself.

It seems that feather format is an ideal candidate to store the data between Jupyter sessions. It shows high I/O speed, does not take too much memory on the disk, and does not need any unpacking when loaded back into RAM.

This comparison does not imply that you should use this format in every possible case. For example, the feather format is not expected to be used as a long-term file storage. Also, it does not take into account all possible situations when other formats could perform better.



## Better Data Formats

The CSV file format has been commonly used as data storage in many Python projects because of its simplicity. However, it is large and slow!

By its nature as a text file, CSV takes larger disk space and longer loading time => Lower Performance. In this article, two better-curated data formats (Parquet and Feather) have been proved to outperform CSV in every way [Reading time, Writing time, Disk storage] as shown in the

Figure: Overview Performance of CSV | Parquet | Feather

Figure: Storage Comparison — CSV | Parquet | Feather

### Setup

```py
    import numpy as np
    import pandas as pd

    import feather
    import pickle
    import pyarrow as pa
    import pyarrow.orc as orc
    from fastavro import writer, reader, parse_schema

    np.random.seed = 42
    df_size = 10_000_000

    df = pd.DataFrame({
        'a': np.random.rand(df_size),
        'b': np.random.rand(df_size),
        'c': np.random.rand(df_size),
        'd': np.random.rand(df_size),
        'e': np.random.rand(df_size)
    })
    df.head()
```

### Parquet vs Feather

- Feather appears to have a slightly better reading/writing performance when using Google Colab.
- Feather has better performance with Solid State Drive (SSD).
- Parquet has better reading performance when read from the network.
- Parquet has better interoperability with the Hadoop system.

Use Feather if your project is mainly on Python or R (not integrated with Hadoop) and has SSD as data storage. Otherwise, use Parquet.

Using Parquet or Feather formats in Python significantly improves data writing, reading, and data storage performance.

### Parquet Format

Parquet is a column-oriented data file format that provides efficient data compression and encoding schemes with enhanced performance to handle complex data in bulk. It was first developed and used in the Apache Hadoop ecosystem. Later, it was adopted by Apache Spark and widely used by cloud vendors like Amazon, Google, and Microsoft for data warehousing.

In Python, the Pandas module has natively supported Parquet, so you can directly integrate the use of Parquet in your project, as in an example below.

```py
    # import module
    import pandas as pd

    # read parquet file as df
    df = pd.read_parquet('<filename>.parquet')

    # do something with df ...

    # write df as parquet file
    df.to_parquet(‘name.parquet’)
```

### Feather Format

Feather was developed using Apache Arrow for fast, interoperable frame storage.

Pandas now supports Feather format natively (starting with version 1.1.0).

We can read/write a Feather file format the same way as CSV/Parquet.

```py
    # import module
    import pandas as pd

    # read feather file as df
    df = pd.read_feather('<filename>.feather')

    # do something with df ...

    # write df as feather file
    df = pd.to_feather('<filename>.feather')
```


## Top Five Alternatives to CSV

The article [3] discuss five alternatives to CSV for data storage:

![Write time comparison|600xauto {Figure 2: Write time comparison in seconds.}](https://miro.medium.com/max/1400/1*y1k2bephs6fp5d7SyMj1Zw.png)


The difference in write times is very interesting. Pretty much any format will have much faster write time than CSV.

![Read time comparison|600xauto {Figure 3: Read time comparison in seconds.}](https://miro.medium.com/max/1400/1*Z0MGbtavEHrSH2SiR0my6A.png)

The read time for CSV is not bad but Apache Avro is terrible. Pickle has the fastest read time, so it looks like the most promising option when working only in Python.

![File size comparison|600xauto {Figure 4: File size comparison in MB.}](https://miro.medium.com/max/1400/1*mcbAJp_cwseCI0pxfPI2FQ.png)

Pretty much any file format has a smaller file size than CSV. The file size reduction ranges from 2.4x to 4.8x, depending on the file format.

### ORC

ORC stands for Optimized Row Columnar which is a data format optimized for reads and writes in Hive.

Since Hive is painfully slow, the developers at Hortonworks decided to develop the ORC file format to improve speed.

In Python, we can use the `read_orc()` function from Pandas to read ORC files. Unfortunately, there is no alternative function for writing ORC files, so we will have to use PyArrow.

```py
    # save pandas dataframe to ORC file
    table = pa.Table.from_pandas(df, preserve_index=False)
    orc.write_table(table, '10M.orc')

    # read ORC file to dataframe
    df = pd.read_orc('10M.orc')
```

### Avro

Avro is an open-source project that provides services of data serialization and exchange for Apache Hadoop.

Avro stores a JSON-like schema with the data, so the correct data types are known in advance which where the compression happens.

Avro has an API for every major programming language, but it does not support Pandas by default.

### Parquet

Apache Parquet is a data storage format designed for efficiency using the column storage architecture, since it allows you to skip data that isn’t relevant quickly. Therefore, both queries and aggregations are faster which results in hardware savings.

Pandas has full support for Parquet files.

```py
    # save dataframe to parquet file
    df.to_parquet('10M.parquet')

    # read [arquet file to dataframe
    df = pd.read_parquet('10M.parquet')
```

### Pickle

We can use the pickle module to serialize objects and save them to a file. We can then deserialize the serialized file to load them back when needed.

Pickle has one major advantage over other formats: we can use it to store any Python object.

One of the most widely used functionalities is saving machine learning models after the training is complete.

The biggest downside is that Pickle is Python-specific, so cross-language support is not guaranteed which could be a deal-breaker for any project requiring data communication between Python and R, for example.

```py
    # save dataframe to pickle file
    with open('10M.pkl', 'wb') as f:
        pickle.dump(df, f)

    # load pickle file
    with open('10M.pkl', 'rb') as f:
        df = pickle.load(f)
```

### Feather

Feather is a data format for storing data frames. It’s designed around a simple premise — to push data frames in and out of memory as efficiently as possible. It was initially designed for fast communication between Python and R, but you’re not limited to this use case.

We can use the feather library to work with Feather files in Python. It’s the fastest available option currently.

```py
    # save Pandas DataFrames to Feather file
    feather.write_dataframe(df, '10M.feather')

    # load Feather file
    df = feather.read_dataframe('10M.feather')
```


----------



## Effective use of Data Types

Make effective use of data types to prevent crashing of memory.

Pandas offer a vast list of API for data explorations and visualization which makes it more popular among the data scientist community.

Dask, modin, Vaex are some of the open-source packages that can scale up the performance of Pandas library and handle large-sized datasets.

When the size of the dataset is comparatively larger than memory using such libraries is preferred, but when dataset size comparatively equal or smaller to memory size, we can optimize the memory usage while reading the dataset.

Here, we discuss how to optimize memory usage while loading the dataset using `read_csv()`.

Using `df.info()` we can view the default data types and memory usage.


The default list of data types assigned by Pandas are:

|  dtype        |                 Usage                  |
| :------------ | :------------------------------------- |
| object        | Text or mixed numeric and text values  |
| int64         | Integer numbers                        |
| float64       | Floating-point numbers                 |
| bool          | True/False values                      |
| datetime64    | Date and time values                   |
| timedelta[ns] | Difference between two datetime values |
| category      | Finite list of text values             |


### Numerical Features

For all numerical values, Pandas assigns float64 data type to a feature column having at least one float value, and int64 data type to a feature column having all feature values as integers.

Here is a list of the ranges of each datatype:

|  Data Type    |                            Description                                   |
| :------------ | :----------------------------------------------------------------------- |
| bool_         | Boolean stored as a byte                                                 |
| int_          | Default integer type (same as C long)                                    |
| intc          | Same as C int (int32 or int64)                                           |
| intp          | Integer used for indexing (ssize_t - int32 or int64)                     |
| int8          | Integer (-2^7 to 2^7 - 1)                                                |
| int16         | Integer (-2^15 to 2^15 - 1)                                              |
| intn          | Integer (-2^(n-1) to 2^(n-1) - 1)                                        |
| uint8         | Unsigned integer (0 to 255)                                              |
| uint16        | Unsigned integer (0 to 2^16 - 1)                                         |
| uint32        | Unsigned integer (0 to 2^32 - 1)                                         |
| float_        | Shorthand for float64                                                    |
| float16       | Half-precision float: sign bit, 5-bit exponent, and 10-bit mantissa    |
| float32       | Single-precision float: sign bit, 8-bit exponent, and 32-bit mantissa  |
| float64       | Double-precision float: sign bit, 11-bit exponent, and 52-bit mantissa |
| complex_      | Shorthand for complex128                                                 |
| complex64     | Complex number represented by two 32-bit floats                          |
| complex128     | Complex number represented by two 64-bit floats                          |


NOTE: A value with data type as int8 takes 8x times less memory compared to int64 data type.

### DateTime

By default, datetime columns are assigned as object data type that can be downgraded to DateTime format.

### Categorical

Pandas assign non-numerical feature columns as object data types which can be downgraded to category data types.

The non-numerical feature column usually has categorical variables which are mostly repeating.

For example, the gender feature column has just 2 categories ‘Male’ and ‘Female’ that are repeating over and over again for all the instances which are re-occupying the space.

Assigning gender to category datatype is a more compact representation.

**Better performance with categoricals**

If you do a lot of analytics on a particular dataset, converting to categorical can yield substantial overall performance gains.

A categorical version of a DataFrame column will often use significantly less memory.

#### Better performance with categoricals

If you do a lot of analytics on a particular dataset, converting to categorical can yield substantial overall performance gains. A categorical version of a DataFrame column will often use significantly less memory, too.

#### Categorical Methods

Series containing categorical data have several special methods similar to the `Series.str` specialized string methods. This also provides convenient access to the categories and codes.

The special attribute cat provides access to categorical methods:

```py
    s = pd.Series(['a', 'b', 'c', 'd'] * 2)

    cat_s = s.astype('category')
    cat_s.cat.codes
    cat_s.cat.categories


    actual_categories = ['a', 'b', 'c', 'd', 'e']

    cat_s2 = cat_s.cat.set_categories(actual_categories)
    cat_s2.value_counts()
    ```

In large datasets, categoricals are often used as a convenient tool for memory savings and better performance. After you filter a large DataFrame or Series, many of the categories may not appear in the data.

```py
  cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
  cat_s3.cat.remove_unused_categories()
```

Table 12-1: Categorical methods for Series in pandas

### Creating dummy variables for modeling

When using statistics or machine learning tools, we usually transform categorical data into dummy variables callwd _one-hot encoding_ which involves creating a DataFrame with a column for each distinct category; these columns contain 1s for occurrences of a given category and 0 otherwise.

```py
    cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')

    pd.get_dummies(cat_s)
```


### Converting between String and Datetime

You can format datetime objects and pandas Timestamp objects as strings using `str` or the `strftime` method passing a format specification.

```py
    stamp = datetime(2011, 1, 3)

    str(stamp)
    stamp.strftime('%Y-%m-%d')
```

You can use many of the same format codes to convert strings to dates using `datetime.strptime`.

```py
    value = '2011-01-03'

    datetime.strptime(value, '%Y-%m-%d')

    datestrs = ['7/6/2011', '8/6/2011']

    [datetime.strptime(x, '%m/%d/%Y') for x in datestrs]
```

pandas is generally oriented toward working with arrays of dates whether used as an axis index or a column in a DataFrame.

The `to_datetime` method parses many different kinds of date representations. It also handles values that should be considered missing (None, empty string, etc.).

NaT (Not a Time) is pandas’s null value for timestamp data.


12.2 Advanced GroupBy Use

12.3 Techniques for Method Chaining

The pipe Method

You can accomplish a lot with built-in pandas functions and the approaches to method chaining with callables that we just looked at.

Sometimes you need to use your own functions or functions from third-party libraries.

```py
    a = f(df, arg1=v1)
    b = g(a, v2, arg3=v3)
    c = h(b, arg4=v4)

    result = (df.pipe(f, arg1=v1)
            .pipe(g, v2, arg3=v3)
            .pipe(h, arg4=v4))
```

The statement `f(df)` and `df.pipe(f)` are equivalent but `pipe` makes chained invocation easier.


### Differences Between astype() and to_datetime() in Pandas

When working with time-series data in Pandas, you can use either `pandas.Series.astype()` or `pandas.to_datetime()` to convert date-time strings to datetime64[ns] data type [16].

Although the method `astype()` can convert the data type of multiple columns in one go, it is always better to use the `to_datetime()` method to convert the data type of time-series data [16].

Here are some tips for using to_datetime():

When you convert date-string to the data type datetime64[ns], it is best to use the optional `format` parameter to specify the date format of the input date strings.

```py
df = pd.DataFrame({"Dates": ["2022-25-12", "2021-01-12", "2022-30-08"]})
df["NewDate_using_to_datetime()"] = pd.to_datetime(df["Dates"],
                                                   format='%Y-%d-%m')
df.info()
df.head()
```

Although the method `to_datetime()` may be slower than `.astype()`, `to_datetime()` is better for handling errors in data type conversion.

Both methods, by default, raise `TypeError` and `ParserError`.

We can avoid these errors by using the `errors` parameter available in both methods.

Ignoring the errors returns the input as it is,  so the output data type remains unchanged which means string/object.

When using `to_datetime()` we can assign `coerce` to the errors parameter which means pandas will convert all the valid date strings to datetime64[ns] format, and all the invalid date strings will be set to `NaT` which stands for Not-a-Time (similar to NaN: Not-a-Number). So now we can determine exactly where we have invalid date strings.

```py
df1["Dates-to_datetime-coerce"] = pd.to_datetime(df1["Dates"], errors='coerce')
df1.info()
df1.head()
```


### Typecasting while Reading Data

The `read_csv` function includes a type parameter which accepts user-provided data types in a key-value format that can use instead of the default ones.

The DateTime feature column can be passed to the `parse_dates` parameter.

```py
    dtype_dict = {
        'vendor_id': 'int8',
        'passenger_count': 'int8',
        'pickup_longitude': 'float16',
        'pickup_latitude': 'float16',
        'dropoff_longitude': 'float16',
        'dropoff_latitude': 'float16',
        'store-and_fwd_flag': 'category',
        'trip_duration': 'int32'
    }

    dates = ['pickup_datetime', 'dropoff_datetime']

    df = pd.read_csv("../data/train.csv",
                     dtype=dtype_dict,
                     parse_dates=dates)

    print(df.shape)
    print(df.info(verbose=False, memory_usage='deep'))
```


----------



## Tips for Large Datasets

Pandas mainly uses a single core of CPU to process instructions and does not take advantage of scaling up the computation across various cores of the CPU to speed up the workflow [7].

Thus, Pandas can cause memory issues when reading large datasets since it fails to load larger-than-memory data into RAM.

There are various other Python libraries that do not load the large data at once but interacts with system OS to map the data with Python. In addition, they utilize all the cores of the CPU to speed up the computations.

The article [8] provides some tips on working with huge datasets using pandas:

- Explicitly pass the data-types
- Select subset of columns
- Convert dataframe to parquet
- Convert to pkl

### Explicitly pass the data-types

```py
    import pandas as pd
    df = pd.read_csv(data_file,
                     n_rows = 100,
                     dtype={'col1': 'object', 'col2': 'float32',})
```

### Select subset of columns

```py
    cols_to_use = ['col1', 'col2',]
    df = pd.read_csv(data_file, usecols=cols_to_use)
```

### Convert dataframe to parquet

```py
    df.to_parquet()
    df = pd.read_parquet()
```

### Convert to pkl

```py
    df.to_pickle(‘train.pkl’)
```



## Libraries for Large Datasets

The article [9] discusses four Python libraries that can read and process large-sized datasets.

- Dask
- Modin
- Vaex
- [Gigasheet](https://www.gigasheet.com)
- Pandas with chunks

### Dask

Dask is an open-source Python library that provides multi-core and distributed parallel execution of larger-than-memory datasets

Dask provides the high-performance implementation of the function that parallelizes the implementation across all the cores of the CPU.

Dask provides API similar to Pandas and Numpy which makes it easy for developers to switch between the libraries.

```py
    import dask.dataframe as dd

    # Read the data using dask
    df_dask = dd.read_csv("DATA/text_dataset.csv")

    # Parallelize the text processing with dask
    df_dask['review'] = df_dask.review.map_partitions(preprocess_text)
```

### Modin

Modin is another Python library that speeds up Pandas notebooks, scripts, or workflows.

Modin distributes both data and computations.

Modin partitions a DataFrame along both axes so it performs on a matrix of partitions.

In contrast to Pandas, Modin utilizes all the cores available in the system, to speed up the Pandas workflow, only requiring users to change a single line of code in their notebooks.

```py
    import modin.pandas as md

    # read data using modin
    modin_df = pd.read_csv("DATA/text_dataset.csv")

    # Parallel text processing of review feature
    modin_df['review'] = modin_df.review.apply(preprocess_text)
```

### Vaex

Vaex is a Python library that uses an _expression system_ and _memory mapping_ to interact with the CPU and parallelize the computations across various cores of the CPU.

Instead of loading the entire data into memory, Vaex just memory maps the data and creates an expression system.

Vaex covers some of the API of pandas and is efficient to perform data exploration and visualization for a large dataset on a standard machine.

```py
    import vaex

    # Read the data using Vaex
    df_vaex = vaex.read_csv("DATA/text_dataset.csv")

    # Parallize the text processing
    df_vaex['review'] = df_vaex.review.apply(preprocess_text)
```


### Read using Pandas in Chunks

Pandas loads the entire dataset into RAM which may cause a memory overflow issue while reading large datasets.

Instead, we can read the large dataset in _chunks_ and perform data processing for each chunk.

The idea is to load 10k instances in each chunk (lines 11–14), perform text processing for each chunk (lines 15–16), and append the processed data to the existing CSV file (lines 18–21).

```py
    # append to existing CSV file or save to new file
    def saveDataFrame(data_temp):

        path = "DATA/text_dataset.csv"
        if os.path.isfile(path):
            with open(path, 'a') as f:
                data_temp.to_csv(f, header=False)
        else:
            data_temp.to_csv(path, index=False)

    # Define chunksize
    chunk_size = 10**3

    # Read and process the dataset in chunks
    for chunk in tqdm(pd.read_csv("DATA/text_dataset.csv", chunksize=chunk_size)):
        preprocessed_review = preprocess_text(chunk['review'].values)
         saveDataFrame(pd.DataFrame({'preprocessed_review':preprocessed_review,
               'target':chunk['target'].values
             }))
```





## References

[1]: [How to Convert to Best Data Types Automatically in Pandas?](https://cmdlinetips.com/2020/04/how-to-convert-to-best-data-types-automatically-in-pandas/amp/)

[2]: [The Best Format to Save Pandas Data](https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d)

[3]: [Stop Using CSVs for Storage - Here Are the Top 5 Alternatives](https://towardsdatascience.com/stop-using-csvs-for-storage-here-are-the-top-5-alternatives-e3a7c9018de0)

[4]: [Optimize Python Performance with Better Data Storage](https://towardsdatascience.com/optimize-python-performance-with-better-data-storage-d119b43dd25a)


[5]: [How to Speed up Pandas by 100x](https://medium.com/geekculture/simple-tricks-to-speed-up-pandas-by-100x-3b7e705783a8)

[6]: [How we optimized Python API server code 100x](https://towardsdatascience.com/how-we-optimized-python-api-server-code-100x-9da94aa883c5)


[7]: [4 Python Libraries that make it easier to Work with Large Datasets](https://towardsdatascience.com/4-python-libraries-that-ease-working-with-large-dataset-8e91632b8791)

[8]: [Pandas tips to deal with huge datasets](https://kvirajdatt.medium.com/pandas-tips-to-deal-with-huge-datasets-f6a012d4e953)

[9]: [Optimize Pandas Memory Usage for Large Datasets](https://towardsdatascience.com/optimize-pandas-memory-usage-while-reading-large-datasets-1b047c762c9b)

[10]: [Top 2 tricks for compressing and loading huge datasets](https://medium.com/the-techlife/top-2-tricks-for-compressing-and-loading-huge-datasets-91a7e394c933)


[11]: [How to Speed Up Pandas with Modin](https://towardsdatascience.com/how-to-speed-up-pandas-with-modin-84aa6a87bcdb)

[12]: [Speed Up Your Pandas Workflow with Modin](https://towardsdatascience.com/speed-up-your-pandas-workflow-with-modin-9a61acff0076)

[13]: [Never Worry About Optimization. Process GBs of Tabular Data 25x Faster With Gigasheet](https://towardsdatascience.com/never-worry-about-optimization-process-gbs-of-tabular-data-25x-faster-with-no-code-pandas-e85ede4c37d5)

[14]: [Don’t use loc/iloc with Loops In Python](https://medium.com/codex/dont-use-loc-iloc-with-loops-in-python-instead-use-this-f9243289dde7)


[15]: [How to Boost Pandas Speed And Process 10M-row Datasets in Milliseconds](https://towardsdatascience.com/how-to-boost-pandas-speed-and-process-10m-row-datasets-in-milliseconds-48d5468e269)


[16]: [3 Practical Differences Between astype() and to_datetime() in Pandas](https://towardsdatascience.com/3-practical-differences-between-astype-and-to-datetime-in-pandas-fe2c0bfc7678)


[17]: [How to Boost Pandas Speed And Process 10M-row Datasets in Milliseconds](https://pub.towardsai.net/how-to-boost-pandas-speed-and-process-10m-row-datasets-in-milliseconds-9f6b37fb407d)


[18]: [20 Pandas Codes To Elevate Your Data Analysis Skills][https://medium.com/codex/20-pandas-codes-to-elevate-your-data-analysis-skills-b62671682190]

[19]: [Practical Pandas Tricks - Part 1: Import and Create DataFrame](https://towardsdatascience.com/introduction-to-pandas-part-1-import-and-create-dataframe-e53326b6e2b1)

[20]: [4 Must-Know Parameters in Python Pandas](https://towardsdatascience.com/4-must-know-parameters-in-python-pandas-6a4e36f6ddaf)

[21]: [How To Change Column Type in Pandas DataFrames](https://towardsdatascience.com/how-to-change-column-type-in-pandas-dataframes-d2a5548888f8)
