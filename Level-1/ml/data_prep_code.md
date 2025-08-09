# Python Data Prep Code

Here are some code snippets which are helpful for data preparation. 

## Data Cleaning (Python)

Here are 5 functions to provide examples for creating a custom data cleaning toolset [26].

1. Remove Multiple Spaces

We can create a function  to remove excessive whitespace from text.

If we want to remove multiple spaces within a string or excessive leading or trailing spaces, this single line function will work.

We make use of regular expressions for internal spaces and `strip()` for leading/trailing whitespace.

```py
  def clean_spaces(text: str) -> str:
      """
      Remove multiple spaces from a string and trim leading/trailing spaces.

      :param text: The input string to clean
      :returns: A string with multiple spaces removed and trimmed
      """
      return re.sub(' +', ' ', str(text).strip())
```

```py
  messy_text = "This   has   too    many    spaces"
  clean_text = clean_spaces(messy_text)
  print(clean_text)
  # This has too many spaces
```

2. Standardize Date Formats

This function will standardize dates to the specified format (YYYY-MM-DD).

```py
  def standardize_date(date_string: str) -> Optional[str]:
      """
      Convert various date formats to YYYY-MM-DD.

      :param date_string: The input date string to standardize
      :returns: A standardized date string in YYYY-MM-DD format, or None if parsing fails
      """
      date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]
      for fmt in date_formats:
          try:
              return datetime.strptime(date_string, fmt).strftime("%Y-%m-%d")
          except ValueError:
              pass
      # Return None if no format matches
      return None
``

```py
  dates = ["2023-04-01", "01-04-2023", "04/01/2023", "April 1, 2023"]
  standardized_dates = [standardize_date(date) for date in dates]
  print(standardized_dates)
  # ['2023-04-01', '2023-04-01', '2023-04-01', '2023-04-01']
```

3. Handle Missing Values

To deal with missing values, we can specify the numeric data strategy to use (‘mean’, ‘median’, or ‘mode’) or categorical data strategy (‘mode’ or ‘dummy’).

```py
  def handle_missing(df: pd.DataFrame, numeric_strategy: str = 'mean', categorical_strategy: str = 'mode') -> pd.DataFrame:
      """
      Fill missing values in a DataFrame.

      :param df: The input DataFrame
      :param numeric_strategy: Strategy for handling missing numeric values ('mean', 'median', or 'mode')
      :param categorical_strategy: Strategy for handling missing categorical values ('mode' or 'dummy')
      :returns: A DataFrame with missing values filled
      """
      for column in df.columns:
          if df[column].dtype in ['int64', 'float64']:
              if numeric_strategy == 'mean':
                  df[column].fillna(df[column].mean(), inplace=True)
              elif numeric_strategy == 'median':
                  df[column].fillna(df[column].median(), inplace=True)
              elif numeric_strategy == 'mode':
                  df[column].fillna(df[column].mode()[0], inplace=True)
          else:
              if categorical_strategy == 'mode':
                  df[column].fillna(df[column].mode()[0], inplace=True)
              elif categorical_strategy == 'dummy':
                  df[column].fillna('Unknown', inplace=True)
      return df
```

```py
  df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': ['x', 'y', np.nan, 'z']})
  cleaned_df = handle_missing(df)
  print(cleaned_df)
```

```
  df[column].fillna(df[column].mode()[0], inplace=True)

            A  B
  0  1.000000  x
  1  2.000000  y
  2  2.333333  x
```

4. Remove Outliers

We can use the IQR method to remove outliers from our data.

We pass in the data and specify the columns to check for outliers and return an outlier-free dataframe.

```py
  import pandas as pd
  import numpy as np
  from typing import List

  def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
      """
      Remove outliers from specified columns using the Interquartile Range (IQR) method.

      :param df: The input DataFrame
      :param columns: List of column names to check for outliers
      :param factor: The IQR factor to use (default is 1.5)
      :returns: A DataFrame with outliers removed
      """
      mask = pd.Series(True, index=df.index)
      for col in columns:
          Q1 = df[col].quantile(0.25)
          Q3 = df[col].quantile(0.75)
          IQR = Q3 - Q1
          lower_bound = Q1 - factor * IQR
          upper_bound = Q3 + factor * IQR
          mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)

      cleaned_df = df[mask]

      return cleaned_df
```

```py
  df = pd.DataFrame({'A': [1, 2, 3, 100, 4, 5], 'B': [10, 20, 30, 40, 50, 1000]})
  print("Original DataFrame:")
  print(df)
  print("\nCleaned DataFrame:")
  cleaned_df = remove_outliers_iqr(df, ['A', 'B'])
  print(cleaned_df)
```

```
  Original DataFrame:
       A     B
  0    1    10
  1    2    20
  2    3    30
  3  100    40
  4    4    50
  5    5  1000

  Cleaned DataFrame:
     A   B
  0  1  10
  1  2  20
```

5. Normalize Text Data

We can create a function to convert all text to lowercase, strip out whitespace, and remove special characters.

```py
  def normalize_text(text: str) -> str:
      """
      Normalize text data by converting to lowercase, removing special characters, and extra spaces.

      :param text: The input text to normalize
      :returns: Normalized text
      """
      # Convert to lowercase
      text = str(text).lower()

      # Remove special characters
      text = re.sub(r'[^\w\s]', '', text)

      # Remove extra spaces
      text = re.sub(r'\s+', ' ', text).strip()

      return text
```

```py
  messy_text = "This is MESSY!!! Text   with $pecial ch@racters."
  clean_text = normalize_text(messy_text)
  print(clean_text)
  # this is messy text wit
```


## Python One-Liners for Data Cleaning

Some useful Python one-liners for common data cleaning tasks [30].

### Quick Data Quality Checks

Here are some essential one-liners to help identify common data quality issues [30]. 

```py
  df.info()

  # Check for Missing Values
  missing_values = df.isnull().sum()
  print("Missing Values:\n", missing_values)

  # Identify Incorrect Data Types
  print("Data Types:\n", df.dtypes)
```

1. Convert Dates to a Consistent Format

Here we convert ‘TransactionDate’ to a consistent datetime format. Any unconvertible values—invalid formats—are replaced with NaT (Not a Time).

```py
  df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
  print(df["TransactionDate"])
```

2. Find Outliers in Numeric Columns

Finding outliers in numeric columns is another important check but it requires some domain knowledge to identify potential outliers. Here, we filter the rows where the ‘Price’ is less than 0, flagging negative values as potential outliers.

```py
  outliers = df[df["Price"] < 0]
  print("Outliers:\n", outliers)
```

3. Check for Duplicate Records

We can check for duplicate rows based on ‘CustomerName’ and ‘Product’, ignoring unique TransactionIDs. Duplicates can indicate repeated entries.

```py
  duplicates = df.duplicated(subset=["CustomerName", "Product"], keep=False)
  print("Duplicate Records:\n", df[duplicates])
```

4. Standardize Text Data

Here we standardize CustomerName by removing extra spaces and ensuring proper capitalization ( "jane rust" → "Jane Rust").

```py
  df["CustomerName"] = df["CustomerName"].str.strip().str.title()
  print(df["CustomerName"])
```

5. Validate Data Ranges

We need to verify that numeric values lie within the expected range.

Here we check if all prices fall within a realistic range, say 0 to 5000. Rows with price values outside this range are flagged.

```py
  invalid_prices = df[~df["Price"].between(0, 5000)]
  print("Invalid Prices:\n", invalid_prices)
```

6. Count Unique Values in a Column

We can check how many times each product appears using the `value-counts()` method which is useful for finding typos or anomalies in categorical data.

```py
  unique_products = df["Product"].value_counts()
  print("Unique Products:\n", unique_products)
```

7. Check for Inconsistent Formatting Across Columns

We can check for inconsistently formatted entries in 'CustomerName' using regex flags to find names that may not match the expected title case format.

```py
  inconsistent_names = df["CustomerName"].str.contains(r"[A-Z]{2,}", na=False)
  print("Inconsistent Formatting in Names:\n", df[inconsistent_names])
```

8. Find Rows with Multiple Issues

We can find rows with more than one issue such as missing values, negative prices, or invalid dates that may need more careful review.

```py
  issues = df.isnull().sum(axis=1) + (df["Price"] < 0) + (~df["TransactionDate"].notnull())
  problematic_rows = df[issues > 1]
  print("Rows with Multiple Issues:\n", problematic_rows)
```

### Capitalize Strings

```py
  # Convert Strings to Uppercase
  strings = ["hello", "world", "python", "rocks"]
  uppercase_strings = [s.upper() for s in strings]

  df['name'] = df['name'].apply(lambda x: x.lower())
```

### String Manipulation

Here are Python one-liners that perform string manipulation.

```py
  # Find Strings Containing a Specific Substring
  fruits = ["apple", "banana", "cherry", "apricot", "blueberry"]
  filtered = [s for s in fruits if "ap" in s]

  # Reverse Strings
  to_do = ["code", "debug", "refactor"]
  reversed_strings = [task[::-1] for task in to_do]

  # Split Strings into Substrings
  strings = ["learn python", "python is fun"]
  split_strings = [s.split() for s in strings]

  # Replace Substrings in Strings
  strings = ["Learn C", "Code in C"]
  replaced_strings = [string.replace("C", "Python") for string in strings]

  # Count Occurrences of a Character
  strings = ["apple", "banana", "cherry"]
  char_counts = [s.count("a") for s in strings]

  # Join Strings
  strings = ["Python", "is", "great"]
  sentence = " ".join(strings)

  # Find the Length of Strings
  strings = ["elephant", "cat", "dinosaur", "ant"]
  lengths = [len(s) for s in strings]

  # Check if Strings are Alphanumeric
  strings = ["hello123", "world!", "python3.12", "rocks"]
  is_alphanumeric = [s.isalnum() for s in strings]

  # Add Suffixes to Strings
  files = ["main", "test", "app"]
  suffixed_files = [file + ".py" for file in files]

  # Extract the First Letter of Each String
  strings = ["banana", "cherry", "date", "blueberry"]
  first_letters = [s[0] for s in strings]

  # Sort Strings Alphabetically in Lowercase
  strings = ["Apple", "banana", "Cherry", "date"]
  sorted_strings = sorted(strings, key=lambda s: s.lower())
```

### Numerical Extraction from Text

Regular expressions (Regex) are character lists that match a pattern in text.

Regex are usually used when we want to perform specific text manipulation [31].

We can use a combination of Regex and map to extract numbers from the text.

```py
  import re

  list(map(int, re.findall(r'\d+', "Sample123Text456")))

  # [123, 456]
```

### Validate Email

Formatting inconsistencies are common with text fields.

Here we check that email addresses are valid and replacing invalid ones with a default address:

```py
  # Verifying that the email contains both an "@" and a ".";
  #assigning 'invalid@example.com' if the format is incorrect
  data = [{**d, "email": d["email"] if "@" in d["email"] and "." in d["email"] else "invalid@example.com"} for d in data]
```

### Trim Whitespace

Sometimes we need to remove unnecessary whitespaces from strings.

Here is a one-liner to trim leading and trailing spaces from the name strings:

```py
  # Trim whitespace from names for cleaner data
  strings = ["  fun ", " funky "]
  trimmed_strings = [s.strip() for s in strings]
```

### Convert Data Types

Ensuring that data types are consistent and correct across the dataset is necessary for accurate analysis.

```py
  # Converting age to an integer type, defaulting to 25 if conversion fails
  data = [{**d, "age": int(d["age"]) if isinstance(d["age"], (int, float)) else 25} for d in data]
```

### Validate Numeric Ranges

It is important to check that numeric values fall within acceptable ranges.

```py
  # Ensuring age is an integer within the range of 18 to 60; otherwise, set to 25
  data = [{**d, "age": d["age"] if isinstance(d["age"], int) and 18 <= d["age"] <= 60 else 25} for d in data]
```

### Handle Missing Values

Missing values are another common problem in most datasets.

Here we check for missing salary values and replace with a default value:

```py
  # Assign default salary of 30,000 if the salary is missing
  data = [{**d, "salary": d["salary"] if d["salary"] is not None else 30000.00} for d in data]
```

We can fill the numerical missing data with the median and the categorical missing data with the mode [31].

```py
  df.fillna({col: df[col].median() for col in df.select_dtypes(include='number').columns} |
            {col: df[col].mode()[0] for col in df.select_dtypes(include='object').columns}, inplace=True)
```

### Standardize Date Formats

It is important to have all dates and times in the same format.

Here we convert various date formats to a single default format with a placeholder for invalid entries:

```py
  from datetime import datetime

  # Attempting to convert the date to a standardized format and defaulting to '2023-01-01' if invalid
  data = [{**d, "join_date": (lambda x: (datetime.strptime(x, '%Y-%m-%d').date() if '-' in x and len(x) == 10 else datetime.strptime(x, '%d-%m-%Y').date()) if x and 'invalid-date' not in x else '2023-01-01')(d['join_date'])} for d in data]
```

It might be better to break this down into multiple steps instead.

Read "Why You Should Not Overuse List Comprehensions in Python" to learn why you should not use comprehensions at the cost of readability and maintainability.

### Remove Negative Values

Sometimes we need to check that certain numerical fields have only non-negative values such as age, salary, and more.

Here we replace any negative salary values with zero:

```py
  # Replace negative salary values with zero to ensure all values are non-negative
  data = [{**d, "salary": max(d["salary"], 0)} for d in data]
```

### Check for Duplicates

Removing duplicate records is important before we can analyze the dataset.

Here we check that only unique records remain by checking for duplicate names:

```py
  # Keeping only unique entries based on the name field

  # Use set to remove duplicates
  data = {tuple(d.items()) for d in data}

  # Convert back to list of dictionaries
  data = [dict(t) for t in data]
```

### Scale Numeric Values

Scaling numeric values can usuallt help with consistent analysis.

We can a list comprehension to scale salaries to a percentage of the maximum salary in the dataset:

```py
  # Normalizing salary values to a percentage of the maximum salary
  max_salary = max(d["salary"] for d in data)
  data = [{**d, "salary": (d["salary"] / max_salary * 100) if max_salary > 0 else 0} for d in data]
```

### Remove Highly Correlated Features

Multicollinearity occurs when our dataset contains many independent variables that are highly correlated with each other instead of with the target which negatively impacts the model performance, so we want to keep less correlated features [31].

We can combine the Pandas correlation feature with the conditional selection to quickly select the less correlated features. For example, here is how we can choose the features that have the maximum Pearson correlation with the others below 0.95.

```py
  df = df.loc[:, df.corr().abs().max() < 0.95]
```

We can try using the correlation features and the threshold to evaluate the prediction model.



## Common Data Wrangling

Here are some common task needed for data wrangling.

### Boolean Masks for Filtering

When working with the NumPy array and its derivate object, we often want to filter the data according to our requirements [31].

We can create a boolean mask to filter the data based on the boolean condition we set.

```py
  import numpy as np

  data = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
```

We can use the boolean mask to filter the data to show only even numbers.

```py
  data[(data % 2 == 0)]
```

This is also the basis of the Pandas filtering, but a Boolean mask can be more versatile since it also works with NumPy arrays.

### Conditional Column Apply

Creating a new column with multiple conditions can sometimes be complicated, and the line to perform them can be long [31].

We can use the `apply` method from Pandas to use specific conditions to create the new feature.

Here are examples of creating a new column where the values are based on the condition of the other column values.

```py
  df['new_col'] = df.apply(lambda x: x['A'] * x['B'] if x['C'] > 0 else x['A'] + x['B'], axis=1)
```

### Finding Common and Different Element

The `Set` data type is unique data that represents an unordered list of data but only with unique elements which can be used to find the common or different elements between two sets [31].

```py
  set1.intersection(set2)
  set1.difference(set2)
```

### List Count Occurrence

When working with a list or any other data with multiple values, there are times when we want to know the frequency for each value [31].

We can use the `counter` function to count values automatically.

```py
  data = [10, 10, 20, 20, 30, 35, 40, 40, 40, 50]

  
  from collections import Counter

  Counter(data)

  # Counter({10: 2, 20: 2, 30: 1, 35: 1, 40: 3, 50: 1})
```

The result is a dictionary for the count occurrence.

### Flatten Nested List

In data preparation, we often encounter nested list data that contains a list within a list [31].

We usually want to flatten the nested list for further data processing.

```py
  nested_list = [

      [1, 2, 3],

      [4, 5],

      [6, 7, 8, 9]

  ]
```

We can then flatten the list with the following code.

```py
  sum(nested_list, [])
  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Now we can use this one-dimensional list to further analyze the data in a more straightforward manner.

### List to Dictionary

Sometimes we have several lists that we want to combine into dictionary format [31].

We can convert the list we have into a dictionary using the zip function.

```py
  fruit = ['apple', 'banana', 'cherry']
  values = [100, 200, 300]
```

We can use zip and dict to combine both of the lists into one structure that can then be used for further data preprocessing.

```py
  dict(zip(fruit, values))
  # {'apple': 100, 'banana': 200, 'cherry': 300}
```

### Dictionary Merging

When we have a dictionary that contains the information we require for data preprocessing, we usually want to combine them.

```py
  fruit_mapping = {'apple': 100, 'banana': 200, 'cherry': 300}
  furniture_mapping = {'table': 100, 'chair': 200, 'sofa': 300}
```

We can combine dictionaries using the following one-liner.

```py
  {**fruit_mapping, **furniture_mapping }
```

Now both dictionaries have been merged into one dictionary which is very useful in many cases that require us to aggregate data.


## Handle Outliers

Outliers can skew the analysis of numerical columns.

We can use the 25th and 75th quartile on numerical data to get the inter-quartile range. Then we can filter out any values outside this range [34].

In statistics, outliers are usually defined as values occurring outside 1.5 times the interquartile range (IQR) from the first quartile (Q1) or third quartile (Q3).

```py
  Q1 = df['salary'].quantile(0.25)
  Q3 = df['salary'].quantile(0.75)
  IQR = Q3 - Q1
```

The above methods find the inter-quartile range on the salary column. Then, we can now filter out outliers using conditional indexing as shown before. This removes the outliers and we are left with rows with values within the acceptable range [34].

```py
  # Filter salaries within the acceptable range
  df = df[(df['salary'] >= Q1 - 1.5 * IQR) & (df['salary'] <= Q3 + 1.5 * IQR)]
```


## References


[26]: [5 DIY Python Functions for Data Cleaning](https://machinelearningmastery.com/5-diy-python-functions-for-data-cleaning/)

[30]: [10 Useful Python One-Liners for Data Cleaning](https://www.kdnuggets.com/10-useful-python-one-liners-for-data-cleaning)

[31]: [10 Python One-Liners That Will Boost Your Data Science Workflow](https://machinelearningmastery.com/10-python-one-liners-that-will-boost-your-data-science-workflow/)


[33]: [15 Useful Python One-Liners for String Manipulation](https://www.kdnuggets.com/15-useful-python-one-liners-string-manipulation)

[34]: [10 Essential Pandas Commands for Data Preprocessing](https://www.kdnuggets.com/10-essential-pandas-commands-data-preprocessing)


[36]: [How to Fully Automate Data Cleaning with Python in 5 Steps](https://www.kdnuggets.com/how-to-fully-automate-data-cleaning-with-python-in-5-steps)
