# Python for Large Datasets

## Pandas for Large Datasets

Pandas mainly uses a single core of CPU to process instructions and does not take advantage of scaling up the computation across various cores of the CPU to speed up the workflow [1].

Thus, Pandas can cause memory issues when reading large datasets since it fails to load larger-than-memory data into RAM.

There are various other Python libraries that do not load the large data at once but interacts with system OS to map the data with Python. In addition, they utilize all the cores of the CPU to speed up the computations.

The article [2] provides some tips on working with huge datasets using pandas:

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

## Python Libraries for Large Datasets

The article [3] discusses four Python libraries that can read and process large-sized datasets.

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

### Read in Chunks using Pandas

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

[1]: [4 Python Libraries that make it easier to Work with Large Datasets](https://towardsdatascience.com/4-python-libraries-that-ease-working-with-large-dataset-8e91632b8791)

[2]: [Pandas tips to deal with huge datasets](https://kvirajdatt.medium.com/pandas-tips-to-deal-with-huge-datasets-f6a012d4e953)

[3]: [Optimize Pandas Memory Usage for Large Datasets](https://towardsdatascience.com/optimize-pandas-memory-usage-while-reading-large-datasets-1b047c762c9b)

[4]: [Top 2 tricks for compressing and loading huge datasets](https://medium.com/the-techlife/top-2-tricks-for-compressing-and-loading-huge-datasets-91a7e394c933)

[5]: [How to Boost Pandas Speed And Process 10M-row Datasets in Milliseconds](https://pub.towardsai.net/how-to-boost-pandas-speed-and-process-10m-row-datasets-in-milliseconds-9f6b37fb407d)

[6]: [Never Worry About Optimization. Process GBs of Tabular Data 25x Faster With Gigasheet](https://towardsdatascience.com/never-worry-about-optimization-process-gbs-of-tabular-data-25x-faster-with-no-code-pandas-e85ede4c37d5)
