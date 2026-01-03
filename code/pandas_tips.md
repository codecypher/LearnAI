# Pandas Tips

Here are some performance tips for using Pandas.

## Performance Tips

Here are some best practices of some of the most common data manipulation operations in Pandas [3].

### Replace loc with at

Using `loc` and `iloc` inside loops is not optimal. Instead, we should use `at` and `iat` which are much faster than `loc` and `iloc` [2].

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

Here are some of the best practices for some most common data manipulation operations in Pandas [5]].

### Indexing efficiently

For choosing a row or multiple rows, `iloc` is faster [5].

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

Many programmers use `loc` or `iloc` to replace specific values in a DataFrame [5].

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

The rule for applying operations on entire columns or data frames is to **never use loops|l** [5].

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


## How to Speedup Pandas

Here are 3 simple tricks to improve speed of Pandas operations [4].

### Use itertuples() instead of iterrows()

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

### Appending new rows efficiently:

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

### Replace apply() with vectorized operations

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

When working with time-series data in Pandas, you can use either `pandas.Series.astype()` or `pandas.to_datetime()` to convert date-time strings to datetime64[ns] data type [6].

Although the method `astype()` can convert the data type of multiple columns in one go, it is always better to use the `to_datetime()` method to convert the data type of time-series data [6].

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

## References

[1]: [How to Convert to Best Data Types Automatically in Pandas?](https://www.geeksforgeeks.org/pandas/how-to-convert-to-best-data-types-automatically-in-pandas/)

[2]: [Don’t use loc/iloc with Loops In Python](https://medium.com/codex/dont-use-loc-iloc-with-loops-in-python-instead-use-this-f9243289dde7)

[3]: [How to Boost Pandas Speed And Process 10M-row Datasets in Milliseconds](https://pub.towardsai.net/how-to-boost-pandas-speed-and-process-10m-row-datasets-in-milliseconds-9f6b37fb407d)

[4]: [How to Speed up Pandas by 100x](https://medium.com/geekculture/simple-tricks-to-speed-up-pandas-by-100x-3b7e705783a8)

[5]: [How to Boost Pandas Speed And Process 10M-row Datasets in Milliseconds](https://pub.towardsai.net/how-to-boost-pandas-speed-and-process-10m-row-datasets-in-milliseconds-9f6b37fb407d)

[6]: [3 Practical Differences Between astype() and to_datetime() in Pandas](https://towardsdatascience.com/3-practical-differences-between-astype-and-to-datetime-in-pandas-fe2c0bfc7678)


[How we optimized Python API server code 100x](https://towardsdatascience.com/how-we-optimized-python-api-server-code-100x-9da94aa883c5)


[How to Speed Up Pandas with Modin](https://towardsdatascience.com/how-to-speed-up-pandas-with-modin-84aa6a87bcdb)

[Speed Up Your Pandas Workflow with Modin](https://towardsdatascience.com/speed-up-your-pandas-workflow-with-modin-9a61acff0076)



