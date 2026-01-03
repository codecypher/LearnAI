# Kedro

Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code [4].

Kedro applies software engineering best-practices to machine learning code including modularity, separation of concerns and versioning [4].

## Commands

```zsh
    kedro new --telemetry=no --starter=spaceflights-pandas
    source ~/pyenv/coreml/bin/activate     # activate virtual environment
```

## Overview

In a data science project, various coding components can be thought of as a flow of data: data flow from the source, to feature engineering, to modelling, to evaluation, etc. [1].

This flow of data is made more complex with training, evaluation, and scoring pipelines where the flow for each pipeline can be potentially very different [1].

`Kedro` is a Python framework to help structure code into a modular data pipeline [1].

Kedro allows reproducible and easy (one-line commands) running of different pipelines and even ad-hoc rerunning of a small portion of a pipeline [1].

- Kedro helps to accelerate data pipelining, enhance data science prototyping, and promote pipeline reproducibility [2].

- Kedro applies software engineering concepts to developing production-ready machine learning code to reduce the time and effort needed for successful model deployment [2]

- Kedro virtually eliminates re-engineering work from low-quality code and standardization of project templates for seamless collaborations [2].

The articles in [1] and [2] discuss the components and terminologies used in Kedro including Python examples on how to setup, configure, and run Kedro pipelines.

## What is Kedro?

Kedro is an open-source Python framework for creating reproducible, maintainable and modular data science code.

Here are some of the key concepts applied within Kedro [2]:

- Reproducibility: Ability to recreate the steps of a workflow across different pipeline runs and environments accurately and consistently.

- Modularity: Breaking down large code chunks into smaller, self-contained, and understandable units that are easy to test and modify.

- Maintainability: Use of standard code templates that allow teammates to readily comprehend and maintain the setup of any project, thereby promoting a standardized approach to collaborative development

- Versioning: Precise tracking of the data, configuration, and machine learning model used in each pipeline run.

- Documentation: Clear and structured information for easy understanding
Seamless Packaging: Allowing data science projects to be documented and shipped efficiently into production (with tools like Airflow or Docker).

## Why Kedro?

The path of bringing a data science project from pilot development to production is fraught with challenges [2]:

- Code that needs to be rewritten for production environments which leads to project delays.
- Disorganized project structures that make collaboration challenging.
- Data flow that is hard to trace.
- Functions that are too leong and difficult to test or reuse.
- Relationships between functions that are hard to understand.

## Kedro Concepts

Kedro is the first open-source software tool developed by McKinsey and is recently donated to Linux Foundation. It is a Python framework for creating reproducible, maintainable, and modular codes.

Kedro combines the best practices of software engineering with the world of data science

Here are the core components of Kedro [1]:

- Node: Function wrapper which wraps input to function, the function itself, and function output together (defines what codes should run)

- Pipeline: Link nodes together which resolves dependencies and determines the execution order of functions (defines what order the codes should be run)

- DataCatalog: Wrapper for data which links the input and output names specified in node to a file path

- Runner: Object that determines how the pipeline (code) is run such as sequentially or in parallel.

## Kedro Docs

Here is an outline of some helpful topics covered in the Kedro documentation [4]:

- IDE Support: Setup Visual Studio Code
- Create: Kedro starters
- Create: Kedro Tools

- Configure: Parameters
- Configure: Credentials

### Getting Started

- Concepts
- Glossary
- Kedro architecture

### Data Catalog

- Introduction
- Kedro Data Catalog
- Data Catalog YAML examples
- Lazy loading

### Develop

- Logging
- Debugging

### Integration & Plugins

- MLflow
- DVC
- PySpark

### Introduction to the Data Catalog

- The basics of catalog.yml
- Dataset access credentials
- Dataset versioning
- Use the Data Catalog within Kedro configuration

### Data Catalog YAML examples

This page contains examples of the YAML configuration file provided in `conf/base/catalog.yml` `or conf/local/catalog.yml`.

## References

[1]: [Kedro as a Data Pipeline in 10 Minutes](https://towardsdatascience.com/kedro-as-a-data-pipeline-in-10-minutes-21c1a7c6bbb)

[2]: [Build an Anomaly Detection Pipeline with Isolation Forest and Kedro](https://towardsdatascience.com/build-an-anomaly-detection-pipeline-with-isolation-forest-and-kedro-db5f4437bfab)

[3]: [Kedro — A Python Framework for Reproducible Data Science Project](https://towardsdatascience.com/kedro-a-python-framework-for-reproducible-data-science-project-4d44977d4f04)

[4]: [Kedro concepts](https://docs.kedro.org/en/stable/getting-started/kedro_concepts/)

-----

[Level Up Your MLOps Journey with Kedro](https://towardsdatascience.com/level-up-your-mlops-journey-with-kedro-5f000e5d0aa0)

[How to perform anomaly detection with the Isolation Forest algorithm](https://towardsdatascience.com/how-to-perform-anomaly-detection-with-the-isolation-forest-algorithm-e8c8372520bc?gi=9b318130c70a)
