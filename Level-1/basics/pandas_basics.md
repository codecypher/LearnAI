# Pandas Basics

Here are some basic concepts and tips for using Pandas given in [7] to [10].

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

The `describe()` method provides an overview of key statistics such as mean, standard deviation, and quartiles for numerical columns.

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

## Convert to Best Data Types Automatically

When we load data as Pandas dataframe, Pandas automatically assigns a datatype to the variables/columns in the dataframe which usually means the datatypes would be `int`, `float` and `object` datatypes [1].

But we can make Pandas infer the best datatypes for the variables in a dataframe [1].

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

## Dates

[11 Essential Tricks To Demystify Dates in Pandas](https://towardsdatascience.com/11-essential-tricks-to-demystify-dates-in-pandas-8644ec591cf1)

[Dealing With Dates in Pandas](https://towardsdatascience.com/dealing-with-dates-in-pandas-6-common-operations-you-should-know-1ea6057c6f4f)

## Iteration

[How To Loop Through Pandas Rows](https://cmdlinetips.com/2018/12/how-to-loop-through-pandas-rows-or-how-to-iterate-over-pandas-rows/amp/)

## String

[String Operations on Pandas DataFrame](https://blog.devgenius.io/string-operations-on-pandas-dataframe-88af220439d1)

## Indexes

[How To Convert a Column to Row Name/Index in Pandas](https://cmdlinetips.com/2018/09/how-to-convert-a-column-to-row-name-index-in-pandas/amp/)

[8 Quick Tips on Manipulating Index with Pandas](https://towardsdatascience.com/8-quick-tips-on-manipulating-index-with-pandas-c10ef9d1b44f)

## Functions

[apply() vs map() vs applymap() in Pandas](https://towardsdatascience.com/apply-vs-map-vs-applymap-pandas-529acdf6d744)

[How to Combine Data in Pandas](https://towardsdatascience.com/how-to-combine-data-in-pandas-5-functions-you-should-know-651ac71a94d6)

## Aggregate

[6 Lesser-Known Pandas Aggregate Functions](https://towardsdatascience.com/6-lesser-known-pandas-aggregate-functions-c9831b366f21)

[Pandas Groupby and Sum](https://cmdlinetips.com/2020/07/pandas-groupby-and-sum/amp/)

## Pivot

[5 Minute Guide to Pandas Pivot Tables](https://towardsdatascience.com/5-minute-guide-to-pandas-pivot-tables-df2d02786886)


## References

[1]: [20 Pandas Codes To Elevate Your Data Analysis Skills][https://medium.com/codex/20-pandas-codes-to-elevate-your-data-analysis-skills-b62671682190]

[2]: [Practical Pandas Tricks - Part 1: Import and Create DataFrame](https://towardsdatascience.com/introduction-to-pandas-part-1-import-and-create-dataframe-e53326b6e2b1)

[3]: [4 Must-Know Parameters in Python Pandas](https://towardsdatascience.com/4-must-know-parameters-in-python-pandas-6a4e36f6ddaf)

[4]: [How To Change Column Type in Pandas DataFrames](https://towardsdatascience.com/how-to-change-column-type-in-pandas-dataframes-d2a5548888f8)
