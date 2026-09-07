# AI/ML Tools

Here is a list of tools that I have found to be helpful for AI engineering:

[GitHub Stars](https://github.com/codecypher?tab=stars)

[Github Student Developer Pack](https://education.github.com/pack)

[HuggingFace Spaces](https://huggingface.co/spaces/launch)


## Python Machine Learning Libraries

- numpy
- pandas

- dateutil
- tqdm
- urllib3

- pingouin
- scipy
- seaborn
- statsmodels
- sympy

### Modin

[Modin](https://github.com/modin-project/modin) is a drop-in replacement for pandas.

While pandas is single-threaded, Modin lets you speed up your workflows by scaling pandas so it uses all of your cores.

Modin works especially well on larger datasets where pandas becomes painfully slow or runs out of memory.

Using modin is as simple as replacing the pandas import:

```py
  # import pandas as pd
  import modin.pandas as pd
```

There is a sample [Notebook](../python/book_recommender_knn.ipynb) that demonstrates using modin.

Since Modin is still under development, I do experience occasional warning/error messages but everything seems to be working. However, the developers seem to be quick to answer questions and provide assistance in troubleshooting issues. Highly recommend trying it out.


## How to choose an ML framework

[Keras vs PyTorch for Deep Learning](https://towardsdatascience.com/keras-vs-pytorch-for-deep-learning-a013cb63870d)


## Machine Learning Tools

- Kedro
- Comet
- DagsHub

- OpenML

- ONNX
- openai/gym
- PyMC (Bayesian statistical modeling)
- Snap ML

### Jupyterlab

JupyterLabis the next-generation user interface for Project Jupyter offering all the familiar building blocks of the classic Jupyter Notebook (notebook, terminal, text editor, file browser, rich outputs, etc.) in a flexible and powerful user interface. JupyterLab will eventually replace the classic Jupyter Notebook.

Jupyterlab has an updated UI/UX with a tab interface for working with multiple files and notebooks.

Since Jupyter is really a web server application, it runs much better on a remote server.

I currently have Jupyterlab installed and running as a Docker container on a VM droplet which runs much better than on my local machine. The only issue is that my VM only has 4GB memory. However, I have had great success so far using Jupyterlab and Modin with notebooks that I am unable to run on my local machine with 32GB memory (out of memory issues) without any performance issues.

### PySpark

PySpark is an interface for Apache Spark in Python. It not only allows you to write Spark applications using Python APIs, but also provides the PySpark shell for interactively analyzing your data in a distributed environment. PySpark supports most of Spark’s features such as Spark SQL, DataFrame, Streaming, MLlib (Machine Learning) and Spark Core.

### Snap ML

Snap ML is a library that provides high-speed training of popular machine learning models on modern CPU/GPU computing systems. 

[Snap ML](https://www.zurich.ibm.com/snapml/)

[Snap ML is 30x Faster than Scikit-Learn](https://medium.com/@irfanalghani11/this-library-is-30-times-faster-than-scikit-learn-206d1818d76f)

[IBM Snap ML Examples](https://github.com/IBM/snapml-examples)

-----

[A Gentle Introduction to Bayesian Belief Networks](https://machinelearningmastery.com/introduction-to-bayesian-belief-networks/)

[Building DAGs with Python](https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/)

[bnlearn](https://github.com/erdogant/bnlearn)



## Graphing Libraries

- Plotly
- Seaborn
- Bokeh
- Cufflinks

- Altair
- ydata Profiling

- SciPy
- Statsmodels

Here are some useful Python graphing libraries:

### Altair: Declarative Visualization Made Simple
 
Altair is a declarative statistical visualization library focusing on simplicity and expressiveness that minimizes boilerplate code and emphasizes interactive charts.

### DuckDB: High-Performance SQL OLAP
 
DuckDB is an in-process SQL OLAP database optimized for analytical workload which allows seamless integration with Python tools like Pandas and Jupyter.

### FlashText: Efficient Text Search and Replacement

FlashText is a lightweight library for keyword extraction and replacement, outperforming regex in speed and simplicity for many use cases.

### Missingno: Visualizing Missing Data

Missingno provides quick and intuitive visualizations for missing data, helping identify patterns and correlations.

### NetworkX: Analyzing Graph Data
 
NetworkX is a versatile library for analyzing and visualizing graph structures from social networks to transportation systems.


## Data Preprocessing Tools

Here are some libraries to help with the data cleaning process [6]:

### Great Expectations

Great Expectations for Data Validation and Quality Checks

Great Expectations is a data quality framework that lets you define, document, and enforce expectations about what your data should look like.

Rather than writing one-off assert statements that fail silently in production, we can build a suite of named checks covering column types, value ranges, null rates, and referential integrity — checks that run against every batch of incoming data.

Great Expectations integrates with pandas, Spark, and SQL databases, and produces human-readable validation reports that can be shared with non-technical stakeholders. The declarative expectation model also doubles as living documentation: the spec tells anyone reading it exactly what "clean data" means for a given pipeline stage. Here's an overview of the features:

- Expectations cover column presence, type constraints, value ranges, uniqueness, regex patterns, and distributional checks.

- Validation results are rendered as browsable HTML reports with pass/fail breakdowns per expectation.

- Data Docs auto-generate data documentation from your expectation suites, keeping specs in sync with the codebase.

- Checkpoints let you run validation as a step inside Airflow, Prefect, or any orchestration pipeline.

Learning resource: "Data quality use cases | Great Expectations" covers almost all use cases you'll need.

### pyjanitor

pyjanitor for Fluent, Chainable DataFrame Cleaning

pyjanitor is a Python package built on top of pandas that adds a clean, verb-based API for common data cleaning tasks.

pyjanitor lets you chain operations (rename columns, drop nulls, encode categoricals, filter rows) all in a single readable pipeline instead of scattering mutations across multiple assignment statements.

pyjanitor extends pandas using the method-chaining pattern, so there is no new mental model to adopt. 

Learning resources: The "pyjanitor API documentation" is thorough and example-driven. "10 PyJanitor's Miscellaneous Functions for Enhancing Data Cleaning | AskPython" is a helpful resource, too.

### Joblib

Joblib is an open-source Python library that helps to save pipelines to a file that can be used later.

### ftfy

ftfy for Fixing Broken Unicode and Text Encoding Problems

ftfy, or "fixes text for you," is a small, focused library that repairs mojibake, incorrect encodings, and mangled Unicode that appears in real-world text data. If you have ever seen garbled accented characters from a CSV exported through Excel, ftfy handles it.

The library has a single purpose: take broken text and return the version that was almost certainly intended.

This focus makes it extremely useful when building pipelines that ingest user-generated content, scraped web data, or records that have passed through multiple legacy systems.

ftfy handles the following:

- Detects and corrects encoding errors caused by misidentified or double-encoded character sets.

- Handles mojibake from common sources.

- Normalizes Unicode to consistent forms, removing invisible characters and zero-width spaces that break downstream matching.

- Runs as a simple `ftfy.fix_text(s)` call with no configuration required for most use cases.

Learning resources: The "ftfy documentation" includes a clear explanation of why these encoding problems occur in the first place. The "ftfy GitHub README" shows the most common failure modes with before-and-after examples.

### fg-data-profiling

fg-data-profiling for Instant Dataset Audits

fg-data-profiling generates a comprehensive exploratory data analysis (EDA) report from any DataFrame in a single line of code [7].

fg-data-profiling detects missing values, duplicate rows, skewed distributions, high-cardinality categoricals, correlations, and outliers — the full checklist of things you would otherwise check by hand before touching the data.

The report is interactive HTML that you can share with teammates or embed in a notebook. Running it at the start of any new dataset gives you an immediate map of where the quality problems live, so cleaning effort goes to the right places instead of being discovered during model training or dashboard queries.

Here are the key features of fg-data-profiling:

- Generates a full statistical profile including distribution plots, correlation matrices, and missing-value heatmaps.

- Flags duplicate rows, constant columns, high-correlation pairs, and columns with suspicious cardinality without any configuration.

- Outputs to HTML, JSON, or notebook widgets, making reports easy to share across technical and non-technical audiences.

- ProfileReport accepts any pandas DataFrame and can compare two datasets side-by-side to detect drift between train and test splits.

Learning resource: The "ydata-profiling documentation" covers configuration, comparison reports, and integration with pandas and Spark.

### Cerberus

Cerberus for Lightweight Schema Validation on Arbitrary Data Structures

Cerberus is a schema validation library for Python dictionaries and nested data structures.

Cerberus is useful when cleaning data that arrives as JSON — such as API responses, event logs, configuration files, and document store exports — where column-level DataFrame validation does not apply but you still need to enforce types, required fields, value constraints, and custom rules.

Cerberus has no dependencies, runs anywhere, and is easy to embed in a cleaning function or ingestion pipeline. You define a schema as a plain Python dictionary, call validator.validate(document), and inspect errors per field. The error messages are structured enough to log, return from an API, or surface to whoever sent the malformed data.

Here are the key features of Cerberus:

- Schema definitions are plain Python dicts with no special syntax to learn; field names map to rule dictionaries with type, required, allowed, and regex keys.

- Coercion rules cast incoming strings to int, float, or datetime as part of validation, combining type-checking and conversion in a single pass.

- Nested document validation handles arbitrarily deep JSON structures, including lists of subdocuments.

- Custom validators are just Python functions, making domain-specific rules like valid SKUs, ISO country codes, and internal ID formats easy to add without external dependencies.

Learning resource: The "Cerberus documentation" covers the full schema rules reference with examples for every constraint type.

## Data Exploration Tools

- Orange
- DataPrep
- Bamboolib
- TensorFlow Data Validation
- Great Expectations

**NOTE:** It is best to install the Orange native executable on your local machine rather than install using anaconda and/or pip.


## Feature Engineering Tools

There are many tools that will help you in automating the entire feature engineering process and producing a large pool of features in a short period of time for both classification and regression tasks.

- Feature-engine
- Featuretools
- AutoFeat


## MLOps Tools

### Kedro

Kedro is an open-source Python framework for creating reproducible, maintainable, and modular data science code.

### Kestra

Kestra is an infinitely scalable orchestration and scheduling platform, creating, running, scheduling, and monitoring millions of complex pipelines.

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


## Deep Learning Tools

### Pretrained Models

- MXNet

- [Model Zoo](https://modelzoo.co/)
- [TensorFlow Hub](https://tfhub.dev/)
- [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official)
- [Hugging Face](https://github.com/huggingface)
- [PyTorch Hub](https://pytorch.org/hub/)
- [Papers with Code](https://paperswithcode.com/)

### Hydra

[Hydra](https://hydra.cc/docs/intro/) is an open-source Python framework that simplifies the development of research and other complex applications.

The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

The name Hydra comes from its ability to run multiple similar jobs - similar to a Hydra with multiple heads.

Hydra provides a configuration file for the entire experiment. We can have different parameters to be set. It can be very helpful when we want to share our code with someone else or run the experiments on a different machine.

Hydra provides the flexibility to set the desired configurations such as learning rate, model hidden layer sizes, epochs, data set name, etc. without exposing someone to make changes to the actual code.

### H5py

[H5py](https://docs.h5py.org/en/stable/quick.html) can be used to store all the intermediate loss values in a dictionary mapped to appropriate key which can be loaded to be reused as a python code.

### Pickle

Pickle can be used to save and load the python classes or PyTorch models for reuse. We can pickle the objects and load it in future to save the time for preprocessing.

### Pipreqs

[Pipreqs](https://pypi.org/project/pipreqs/) is useful when we want to port our code to a different machine and install all the dependencies.

Pipreqs scans all the .py files in a given directory and looks for the imports which means it should write only the libraries you actually use to `requirements.txt`.

Pipreqs helps us to create a list of python dependencies along with the versions that the current code is using and saves it in a file.

```py
  # show the libraries are used in the project
  pipreqs . --print
```

### Tqdm

When used with a loop (here we use with a loop over a torch.utils.data.DataLoader object), [Tqdm](https://tqdm.github.io/) provides a viewe of time per gradient step or epoch which can help us to set our logging frequency of different results or saving the model or get an idea to set the validation intervals.


## GenAI

- 1min.ai
- console.groq.com
- app.aimagicx.com
- you.com
- insmind.com (photos and images)


## CV Libraries

- OpenCV
- openpilot

- ageitgey/face_recognition
- qubvel/segmentation_models

## Time Series

- statsmodels
- stumpy
- AutoTS
- Darts
- TsFresh

## NLP Libraries

- NLTK
- GenSim
- Polyglot
- SpaCy
- Textblob

- Pattern
- clean-text

- Presidio
- PySBD
- SymSpell
- TextAttack


## References

[1]: https://towardsdatascience.com/all-top-python-libraries-for-data-science-explained-with-code-40f64b363663 "All Top Python Libraries for Data Science Explained"

[2]: https://towardsdatascience.com/26-github-repositories-to-inspire-your-next-data-science-project-3023c24f4c3c "26 GitHub Repositories To Inspire Your Next Data Science Project"

[3]: https://towardsdatascience.com/4-amazing-python-libraries-that-you-should-try-right-now-872df6f1c93 "4 Amazing Python Libraries That You Should Try Right Now"

[4]: https://towardsdatascience.com/tools-for-efficient-deep-learning-c9585122ded0 "Tools for Efficient Deep Learning"

[5]: https://www.kdnuggets.com/5-python-libraries-that-make-data-cleaning-more-enjoyable "5 Python Libraries That Make Data Cleaning More Enjoyable"

[6]: https://www.kdnuggets.com/5-python-libraries-that-make-data-cleaning-more-enjoyable "5 Python Libraries That Make Data Cleaning More Enjoyable"

[7]: https://github.com/Data-Centric-AI-Community/fg-data-profiling#fg-data-profiling "fg-data-profiling"

-----

[PySpark Getting Started](https://spark.apache.org/docs/latest/api/python/getting_started/index.html)

[Orange Docs](https://orangedatamining.com/docs/)

[A Great Python Library: Great Expectations](https://towardsdatascience.com/a-great-python-library-great-expectations-6ac6d6fe822e)

[The Only Web Scraping Tool you need for Data Science](https://medium.com/nerd-for-tech/the-only-web-scraping-tool-you-need-for-data-science-f388e2afa187)

[Using Joblib to speed up your Python pipelines](https://medium.com/data-science/using-joblib-to-speed-up-your-python-pipelines-dd97440c653d)

[Lightweight Pipelining In Pytho Using Joblib](https://medium.com/data-science/lightweight-pipelining-in-python-1c7a874794f4)

