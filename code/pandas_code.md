# Pandas Code Snippets

## Code Snippets

Here are some useful Pandas code snippets for data science and analysis projects [1].

```py
    import pandas pd

    # Combine multiple excel sheets of the same file
    # into single dataframe and save as .csv
    excel_file = pd.read_excel(‘file.xlsx’, sheet_name=None)
    dataset_combined = pd.concat(excel_file.values())

    # Extract the year from a date data in pandas
    df['year'] = df['date'].dt.year
    df.head()
```

### Find boundaries for outliers

```py
    def find_boundaries(df, variable, distance=1.5):

        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

        lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
        upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

        return upper_boundary, lower_boundary
```

### Compute cardinality of datasets

```py
    data.nunique().plot.bar(figsize=(12,6))
    plt.ylabel('Number of unique categories')
    plt.xlabel('Variables')
    plt.title('Cardinality')

    ## Version with 5% threshold

    fig = label_freq.sort_values(ascending=False).plot.bar()
    fig.axhline(y=0.05, color='red')
    fig.set_ylabel('percentage of cars within each category')
    fig.set_xlabel('Variable: class')
    fig.set_title('Identifying Rare Categories')
    plt.show()
```

### Find Missing data with charts

```py
    data.isnull().mean().plot.bar(figsize=(12,6))
    plt.ylabel('Percentage of missing values')
    plt.xlabel('Variables')
    plt.title('Quantifying missing data')
```

### Add categorical/text labels

```py
    # Add column that gives a unique number to each of these labels
    df['label_num'] = df['label'].map({
        'Household' : 0,
        'Books': 1,
        'Electronics': 2,
        'Clothing & Accessories': 3
    })

    # check the results
    df.head(5)
```

### Preprocess text with Spacy

```py
    import spacy

    # load english language model and create nlp object from it
    nlp = spacy.load("en_core_web_sm")

    def preprocess(text):
        """
        utlity function for pre-processing text
        """
        # remove stop words and lemmatize the text
        doc = nlp(text)
        filtered_tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            filtered_tokens.append(token.lemma_)

        return " ".join(filtered_tokens)

        df['preprocessed_txt'] = df['Text'].apply(preprocess)
```

## Pandas one-liners

Here are some helpful Pandas one-liners [2], [3]. [4].

```py
  # n-largest values in a series
  # find the top-n paid roles
  data.nlargest(n, "Employee Salary", keep = "all")
```

```py
  # n-smallest values in a series
  data.nsmallest(n, "Employee Salary", keep = "all")
```

### Filtering

Suppose we want to update the values in a column based on whether some condition is true.

The following code works but we overwrite our original data which means we would first have to copy the original column then run this line on the new column.

```py
  df.loc[df["alcohol"] >= 10, "quality"] = 0
```

The alternative is to use the `mask` method:

```py
  df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

  # Set to NaN, all values where the age is over 30
  new_df = df.mask(df["age"] > 30)

  df["quality_updated"] = df["quality"].mask(df["alcohol"] >= 10)
```

### crosstab and pivot

`crosstab` computes a cross-tabulation of two (or more) columns/series and returns a frequency of each combination.

```py
  # compute the number of employees working from each location within every company
  pd.crosstab(data["Company Name"], data["Employee Work Location"])

  result_crosstab = pd.crosstab(data["Company Name"], data["Employee Work Location"])
  sns.heatmap(result_crosstab, annot=True)

  # compute aggregation on average salary
  result_crosstab = pd.crosstab(index = data["Company Name"],
                columns=data["Employment Status"],
                values = data["Employee Salary"],
                aggfunc=np.mean)
  sns.heatmap(result_crosstab, annot=True, fmt='g')
```

Similar to crosstabs, pivot tables in Pandas provide a way to cross-tabulate your data.

```py
  pd.pivot_table(data,
               index=["Company Name"],
               columns=["Employee Work Location"],
               aggfunc='size',
               fill_value=0)

  result_pivot = pd.pivot_table(data,
            index=["Company Name"],
            columns=["Employee Work Location"],
            aggfunc='size',
            fill_value=0)

  sns.heatmap(result_pivot, annot=True, fmt='g')
```

### Mark duplicate rows

```py
  # Marks all duplicates as True except for the first occurrence.
  new_data.duplicated(keep="first")

  # create filtered Dataframe with no duplicates
  # Marks all duplicates as True
  new_data[~new_data.duplicated(keep=False)]

  # check duplicates on a subset of columns
  new_data.duplicated(subset=["Company Name", "Employee Work Location"], keep=False)

  # Remove duplicates
  new_data.drop_duplicates(keep="first")

  # drop duplicates on a subset of columns
  new_data.drop_duplicates(subset=["Company Name", "Employee Work Location"], keep = False)
  view raw
```

### Remove columns with null values

```py
  # remove columns with any number of null values
  df.drop(df.columns[df.isnull().sum() > 0], axis=1, inplace=True)
```

The `apply` method:

```py
  # create a new column based on existing columns
  df['new_col'] = df.apply(lambda x: x['col_1'] * x['col_2'], axis=1)
```

### Aggregate operations

```py
  # group and calculate the mean of columns
  df.groupby('group_col').mean()

  # filter rows based on a specific value
  df.loc[df['col'] == 'value']

  # sort the dataframe by a specific column
  df.sort_values(by='col_name', ascending=False)
```

### Create a dictionary from a list

```py
  grades = ["A", "A", "B", "B", "A", "C", "A", "B", "C", "A"]

  pd.Series(grades).value_counts().to_dict()
```

### Create a DataFrame from a JSON file

```py
  with open("data.json") as f:
      data = json.load(f)

  df = pd.json_normalize(data, "data")
```

### Reformat using explode

Consider a case where you have a list of items that match a particular record. You need to reformat it in a way that there is a separate row for each item in that list.

```py
  df_new = df.explode(column="data").reset_index(drop=True)
```

The reset_index assigns a new integer index to the resulting DataFrame. Otherwise, the index before exploding would be preserved (i.e. all the rows with a key value of A would have an index of 0).

### Using combine_first

The `combine_first` function serves for a specific purpose but simplifies that specific task greatly.

If there is a row in column A with a missing value (i.e. NaN), we want it to be filled with the value of the same row in column B.

```py
  df["A"].combine_first(df["B"])
```

If there are 3 columns that we want to use, we can chain `combine_first` functions.

The following line of code first checks column A. If there is a missing value, it takes it from column B. If the corresponding row in column B is also NaN, then it takes the value from column C.

```py
  df["A"].combine_first(df["B"]).combine_first(df["C"])
```

## References

[1]: [8 useful python code snippets for data science and analysis projects](https://medium.com/mlearning-ai/8-useful-python-code-snippets-for-data-science-and-analysis-projects-e76b0f391fb)

[2]: [Powerful One-liners in Pandas Every Data Scientist Should Know](https://towardsdatascience.com/powerful-one-liners-in-pandas-every-data-scientist-should-know-737e721b81b6)

[3]: [10 Pandas One Liners for Data Access, Manipulation, and Management](https://www.kdnuggets.com/2023/01/pandas-one-liners-data-access-manipulation-management.html)

[4]: [4 Pandas One-Liners That Solve Particular Tasks Efficiently](https://towardsdatascience.com/4-pandas-one-liners-that-surprised-me-in-a-good-way-b67955211f81)
