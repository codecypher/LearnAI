# OpenML

## Get Started

OpenML is readily integrated with scikit-learn through the Python API.

```py
    from sklearn import ensemble
    from openml import tasks, flows, Runs

    task = tasks.get_task(3954)
    clf = ensemble.RandomForestClassifier()
    flow = flows.sklearn_to_flow(clf)
    run = runs.run_flow_on_task(task, flow)
    result = run.publish()
```

Key features:

- Query and download OpenML datasets and use them however you like.
- Build any sklearn estimator or pipeline and convert to OpenML flows.
- Run any flow on any task and save the experiment as run objects.
- Upload your runs for collaboration or publishing.
- Query, download, and reuse all shared runs.

----------



## OpenML Python: Get Started

The OpenML Python package allows to use datasets and tasks from OpenML together with scikit-learn and share the results online.

```py
    import openml
    from sklearn import impute, tree, pipeline

    # Define a scikit-learn classifier or pipeline
    clf = pipeline.Pipeline(
        steps=[
            ('imputer', impute.SimpleImputer()),
            ('estimator', tree.DecisionTreeClassifier())
        ]
    )
    # Download the OpenML task for the german credit card dataset with 10-fold cross-validation.
    task = openml.tasks.get_task(32)

    # Run the scikit-learn model on the task.
    run = openml.runs.run_model_on_task(clf, task)

    # Publish the experiment on OpenML (optional, requires an API key.
    # You can get your own API key by signing up to OpenML.org)
    run.publish()

    print(f'View the run online: {run.openml_url}')
```

You can find more examples in our **Examples Gallery**.



## OpenML Python: User Guide

This document will guide you through the most important use cases, functions and classes in the OpenML Python API. 

We use `pandas` to format and filter tables.

The OpenML Python package is a connector to OpenML that allows you to use and share datasets and tasks, run machine learning algorithms on them, and share the results online.

### Configuration

The configuration file resides in a directory `.config/openml` in the home directory of the user and is called config (More specifically, it resides in the configuration directory specified by the XDGB Base Directory Specification). 

The config file consists of `key = value` pairs that are separated by newlines. 

The following keys are defined:

- apikey
- server
- verbosity

- cachedir
- avoid_duplicate_runs
- retry_policy
- connection_n_retries


### Docker

It is also possible to try out the latest development version of `openml-python` with docker:

```bash
  docker run -it openml/openml-python
```

See the **openml-python docker documentation** for more information.


### Key concepts

OpenML contains several key concepts which it needs to make machine learning research shareable. 

A machine learning experiment consists of one or more _runs_ which describe the performance of an algorithm (called a _flow_ in OpenML) using its hyperparameter settings (called a _setup_) on a _task_. 

A _Task_ is the combination of a dataset, a split, and an evaluation metric. 

In this user guide, we will go through listing and exploring existing tasks to actually running machine learning algorithms on them. 

In a further user guide, we will examine how to search through datasets in order to curate a list of tasks.

A further explanation is given in the **OpenML user guide**.

### Working with tasks

We can think of a task as an experimentation protocol that describes how to apply a machine learning model to a dataset in a way that is comparable with the results of others. 

Tasks are containers that define which dataset to use, what kind of task we are solving (regression, classification, clustering, etc…), and which column to predict. 

Tasks describe how to split the dataset into a train and test set, whether to use several disjoint train and test splits (cross-validation), and whether this should be repeated several times. 

The task also defines a target metric for which a flow should be optimized.

Below you can find our tutorial regarding tasks and if you want to know more you can read the **OpenML guide: Tasks**. 

### Running machine learning algorithms and uploading results

In order to upload and share results of running a machine learning algorithm on a task, we need to create an `OpenMLRun`. 

A run object can be created by running a `OpenMLFlow` or a scikit-learn compatible model on a task. 

We will focus on the simpler example of running a `scikit-learn` model.

A flow is a description of something runable which does the machine learning. 

A flow contains all information to setup the necessary machine learning library and its dependencies as well as all possible parameters.

A _run_ is the outcome of running a flow on a task. 

A run contains all parameter settings for the flow, a setup string (most likely a command line call), and all predictions of that run. 

When a run is uploaded to the server, the server automatically calculates several metrics which can be used to compare the performance of different flows to each other.

So far, the OpenML Python connector works only with estimator objects following the scikit-learn estimator API which can be directly run on a task and a flow will automatically be created or downloaded from the server if it already exists.

The next tutorial covers how to train different machine learning models, how to run machine learning models on OpenML data, and how to share the results: **Flows and Runs**. 

### Datasets

OpenML provides a large collection of datasets and the benchmark “OpenML100” which consists of a curated list of datasets.

You can find the dataset that best fits your requirements by making use of the available **metadata**. 

The tutorial which follows explains how to get a list of datasets, how to filter the list to find the dataset that suits your requirements, and how to download a dataset: **Datasets**. 

OpenML is about sharing machine learning results and the datasets they were obtained on. 

Learn how to share your datasets in the following tutorial: **Dataset upload tutorial**. 



## Examples Gallery

- Introductory Examples
- In-Depth Examples
- Usage in research papers

### Introductory Examples

Introductory examples to the usage of the OpenML python connector.

### In-Depth Examples

Extended examples for the usage of the OpenML python connector.

### Usage in research papers

These examples demonstrate how OpenML-Python can be used for research purposes by re-implementing its use in recent publications.


----------


## OpenML User Guide

- Datasets
- Tasks
- Flows
- Runs

### Concepts

OpenML operates on a number of core concepts which are important to understand:

**Datasets**

Datasets simply consist of a number of rows (called instances), usually in tabular form.

Example: The iris dataset

**Tasks**

A task consists of a dataset, a machine learning task to perform such as classification or clustering, and an evaluation method. 

For supervised tasks, this also specifies the target column in the data.

Example: Classifying different iris species from other attributes and evaluate using 10-fold cross-validation.

**Flows**

A flow identifies a particular machine learning _algorithm_ from a particular library or framework such as Weka, mlr, or scikit-learn.

Example: WEKA's RandomForest

**Runs**

A run is a particular flow (algorithm) with a particular parameter setting, applied to a particular task.

Example: Classifying irises with WEKA's RandomForest

### Data¶

You can upload and download datasets through the website or API; Data hosted elsewhere can be referenced by URL.

Data consists of columns (also known as features or covariates), each of which is either numeric, nominal or a string, and has a unique name. 

A column can also contain any number of missing values.

Most datasets have a "default target attribute" which denotes the column that is usually the target, also known as dependent variable, in supervised learning tasks. 

The default target column is denoted by "(target)" in the web interface. 

Not all datasets have such a column, though, and a supervised task can pick any column as the target (as long as it is of the appropriate type).

OpenML automatically analyzes the data, checks for problems, visualizes it, and computes data characteristics or _data qualities_ (including simple ones such as number of features but also more complex statistics such as kurtosis or the AUC of a decision tree of depth 3). 

The data qualities can be useful to find and compare datasets.

Every dataset gets a dedicated page with all known information (check out zoo), including a wiki, visualizations, statistics, user discussions, and the tasks in which it is used.

**Dataset ID and versions**

A dataset can be uniquely identified by its dataset ID which you can find in the URL of the dataset page such as 62 for zoo. 

Each dataset also has a name, but several datasets can have the same name. 

When several datasets have the same name, they are called "versions" of the same dataset (although that is not necessarily true). 

The version number is assigned according to the order of upload. 

Different versions of a dataset can be accessed through the drop down menu at the top right of the dataset page.


### Tasks

Tasks describe what to do with the data. 

OpenML covers several **task types** such as classification and clustering. 

You can also create tasks online.

Tasks are little containers including the data and other information such as train/test splits, and define what needs to be returned.

Tasks are machine-readable so that machine learning environments know what to do, and you can focus on finding the best algorithm. 

You can run algorithms on your own machine(s) and upload the results. 

OpenML evaluates and organizes all solutions online.

Tasks are real-time, collaborative data mining challenges: you can study, discuss, and learn from all submissions (code has to be shared) while OpenML keeps track of who was first.

Tasks specify the dataset, the kind of machine learning task (regression), the target attribute (which column in the dataset should be predicted), the number of splits for cross-validated evaluation and the exact dataset splits, and optional evaluation metric (mean squared error). 

Given this specification, a task can be solved using any of the integrated machine learning tools such as Weka, mlr, and scikit-learn.

NOTE: You can also supply hidden test sets for the evaluation of solutions. Novel ways of ranking solutions will be added in the near future.


### Flows

Flows are algorithms, workflows, or scripts solving tasks. 

You can upload flows through the website or APIs. 

Code hosted elsewhere (such as GitHub) can be referenced by URL, though typically they are generated automatically by machine learning environments.

Flows contain all the information necessary to apply a particular workflow or algorithm to a new task. 

Usually a flow is specific to a task-type which means you cannot run a classification model on a clustering task.

Every flow gets a dedicated page with all known information (check out WEKA's RandomForest) including a wiki, hyperparameters, evaluations on all tasks, and user discussions.

NOTE: Each flow specifies requirements and dependencies, and you need to install these locally to execute a flow on a specific task. We aim to add support for VMs so that flows can be easily (re)run in any environment.


### Runs

Runs are applications of flows to a specific task. 

Runs are typically submitted automatically by machine learning environments (through the OpenML APIs) with the goal of creating a reproducible experiment (though exactly reproducing experiments across machines might not be possible because of changes in numeric libraries and operating systems).

OpenML organizes all runs online, linked to the underlying data, flows, parameter settings, people, and other details. 

OpenML also independently evaluates the results contained in the run given the provided predictions.

You can search and compare everyone's runs online, download all results into your favorite machine learning environment, and relate evaluations to known properties of the data and algorithms.

### Tags

Datasets, tasks, runs, and flows can be assigned tags via the web interface or the API. 

These tags can be used to search and annotate datasets. 

For example, the tag "OpenML100" refers to benchmark machine learning algorithms used as a benchmark suite. 

Anyone can add or remove tags on any entity.



## References

[OpenML Get Started](https://docs.openml.org)

[OpenML Python: Get Started](https://docs.openml.org/Python-start/)


