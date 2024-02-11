# AI Checklists

## When not to use ML

The article [1] discusses four reasons when you should not use machine learning.


### Data-related issues

In the [AI hierarchy of needs][^ai_hierarchy], it is important that you have a robust process for collecting, storing, moving, and transforming data. Otherwise, GIGO. 

Not only do you need your data to be **reliable** but you need **enough** data to leverage the power of machine learning. 


### Interpretability

There are two main categories of ML models: 

- Predictive models focus on the model’s ability to produce accurate predictions.

- Explanatory models focus on understanding the relationships between the variables in the data.

ML models (especially ensemble models and neural networks) are predictive models that are much better at predictions than traditional models such as linear/logistic regression.

However, when it comes to understanding the relationships between the predictive variables and the target variable, these models are a _black box_. 

You may understand the underlying mechanics behind these models, but it is still not clear how they get to their final results.

In general, ML and deep learning models are great for prediction but lack explainability.


### Technical Debt

Maintaining ML models over time can be challenging and expensive. 

There are several types of debt to consider when maintaining ML models:

- **Dependency debt:** The cost of maintaining multiple versions of the same model, legacy features, and underutilized packages.

- **Analysis debt:** This refers to the idea that ML systems often end up influencing their own behavior if they update over time, resulting in direct and hidden feedback loops.

- **Configuration debt:** The configuration of ML systems incur a debt similar to any software system.


### Better Alternatives

ML should not be used when simpler alternatives exist that are equally as effective. 

You should start with the simplest solution that you can implement and iteratively determine if the marginal benefits from the next best alternative outweighs the marginal costs.

> Simpler = Better (Occam's Razor)



## Mistakes to Avoid in AI

Here are eight mistakes to avoid when using machine learning [12]:

1. Not understanding the user

What does your user or business really want, you must understand from the beginning

2. Not performing failure analysis

If you do not perform a failure analysis (an analysis of the frequency of different categories of failure of your system) you may be expending a lot of effort for little result.

3. Not looking at the model

Clearly look for the weights and splits which may end up causing you to choose the wrong model

4. Not using existing solutions

Explore the existing solutions from the major technology companies. It is not always a good idea to create unique solutions. 

5. Not comparing to a simple baseline model

It is natural to want to start with a complex model. But sometimes a single neutron(logistic regression) performs as well as a deep neural network with six hidden layers

6. Not looking for data leakage

In case of data leakage, the proper information or clues wont be available at the time of prediction, as a result wrong solution would come

7. Not looking at the data

When you do not look at the data carefully, you can miss useful insights which will lead to a data error and missing data

8. Not qualifying the use case

Before starting a machine learning project, it is important to determine whether the project is worth doing and to consider its ramifications. 



----------


## Feature Engineering

Feature engineering is the process of transforming data to extract valuable information.

If done correctly, feature engineering can play even a bigger role in model performance than hyperparameter tuning.

A checklist for transforming features for better model performance is given in [5]. 

The article [6] explains and implements PCA in Python. 



----------



## AI Checklist

A checklist for transforming features for better model performance.

The software development lifecycle (SDLC) of an AI project can be divided into six stages [2]:

1. **Problem definition:** The formative stage of defining the scope, value definition, timelines, governance, resources associated with the deliverable.

2. **Dataset Selection:** This stage can take a few hours or a few months depending on the overall data platform maturity and hygiene. Data is the lifeblood of ML, so getting the right and reliable datasets is crucial.

3. **Data Preparation:** Real-world data is messy. Understanding data properties and preparing properly can save endless hours down the line in debugging.

4. **Design:** This phase involves feature selection, reasoning algorithms, decomposing the problem, and formulating the right model algorithms.

5. **Training:** Building the model, evaluating with the hold-out examples, and online experimentation. 

6. **Deployment:** Once the model is trained and tested to verify that it met the business requirements for model accuracy and other performance metrics, the model is ready for deployment. There are two common approaches to deployment of ML models to production: embed models into a web server or offload the model to an external service. Both ML model serving approaches have pros and cons.

7. **Monitoring:** This is the post-deployment phase involving observability of the model and ML pipelines, refresh of the model with new data, and tracking success metrics in the context of the original problem.


The two most common architectures for ML model serving are:

1. **Precomputed Model Prediction:** This is one of the earliest used and simplest architecture for serving machine learning models. It is an indirect method for serving the model where we precompute predictions for all possible combinations of input variables and store them in a database. This architecture is generally used in recommendation systems — recommendations are precomputed and stored and shown to the user at login.

2. **Microservice Based Model Serving:** The model is served independently of the application, and predictions are provided in real-time as per request. This type of architecture provides flexibility in terms of model training and deployment.


### When to retrain the model

The performance of an ML model degrades over time in production, so it is best to evaluate retraining requirements before model serving. Based on the use case, model monitoring, and evaluation, one can decide when to retrain the model again. One good way to decide on retraining time is to use out-of-time analysis on different time windows.

### How to retrain the model

Retraining is essential and it helps to keep the model up to date. There are broadly two ways to retrain machine learning models — online & offline training.

Online Training: The model is re-trained while in production. True labels are circulated back to the model at a certain interval to update/ retrain the model. This requires a separate architecture and is generally hard to implement.

For example, when we predict ad-click probability we can get feedback (clicked or not clicked) which can be used to update the model online.

Offline Training: The model is re-trained from scratch, so we have full control over the new model and data to train. The new model is pushed in production using A/B testing or shadow testing.


----------



## Problem Definition Checklist

1. Verify there is quantifiable business value in solving the problem. 

2. Verify that simpler alternatives (such as hand-crafted heuristics) are not sufficient to address the problem.

3. Ensure that the problem has been decomposed into the smallest possible units.

4. Clear understanding of how the AI output will be applied to accomplish the desired business outcome.

5. Clear measurable metric(s) to measure the success of the solution.

6. Clear understanding of precision versus recall tradeoff of the problem.

7. Verify impact when the logistic classification prediction is incorrect.

8. Ensure project costs include the cost of managing the corresponding data pipelines.



## Dataset Selection Checklist

1. Verify the meaning of the dataset attributes.

2. Verify the derived metrics used in the project are standardized.

3. Verify data from warehouse is not stale due to data pipeline errors.

4. Verify schema compliance of the dataset.

5. Verify datasets comply with data rights regulations (such as GDPR, CCPA, etc.).

6. Ensure there is a clear change management process for dataset schema changes.

7. Verify dataset is not biased.

8. Verify the datasets being used are not orphaned (without data stewards).



## Data Preparation Checklist

1. Verify data is IID (Independent and Identically Distributed).

2. Verify expired data is not used -- historic data values that may not be relevant.

3. Verify there are no systematic errors in data collection.

4. Verify dataset is monitored for sudden distribution changes.

5. Verify seasonality in data (if applicable) is correctly taken into account.

6. Verify data is randomized (if applicable) before splitting into training and test data.

7. Verify there are no duplicates between test and training examples.

8. Verify sampled data is statistically representative of the dataset as a whole.

9. Verify the correct use of normalization and standardization for scaling feature values.

10. Verify outliers have been properly handled.

11. Verify proper sampling of selected samples from a large dataset.



## Design Checklist

1. Ensure feature crossing is experimented before jumping to non-linear models (if applicable).

2. Verify there is no feature leakage.

3. Verify new features are added to the model with justification documented on how they increase the model quality.

4. Verify features are correctly scaled.

5. Verify simpler ML models are tried before using deep learning.

6. Ensure hashing is applied for sparse features (if applicable).

7. Verify model dimensionality reduction has been explored.

8. Verify classification threshold tuning (in logistic regression) takes into account business impact.

9. Verify regularization or early stopping in logistic regression is applied (if applicable).

10. Apply embeddings to translate large sparse vectors into a lower-dimensional space (while preserving semantic relationships).

11. Verify model freshness requirements based on the problem requirements.

12. Verify the impact of features that were discarded because they only apply to a small fraction of data.

13. Check if feature count is proportional to the amount of data available for model training.



## Training Checklist

1. Ensure interpretability is not compromised prematurely for performance during early stages of model development.

2. Verify model tuning is following a scientific approach (rather than ad-hoc).

3. Verify the learning rate is not too high.

4. Verify root causes are analyzed and documented if the loss-epoch graph is not converging.

5. Analyze specificity versus sparsity trade-off on model accuracy.

6. Verify that reducing loss value improves recall/precision.

7. Define clear criteria for starting online experimentation (canary deployment).

8. Verify per-class accuracy in multi-class classification.

9. Verify infrastructure capacity or cloud budget allocated for training.

10. Ensure model permutations are verified using the same datasets (for an apples-to-apples comparison).

11. Verify model accuracy for individual segments/cohorts, not just the overall dataset.

12. Verify the training results are reproducible -- snapshots of code (algo), data, config, and parameter values.

13. Verify there are no inconsistencies in training-serving skew for features.

14. Verify feedback loops in model prediction have been analyzed.

15. Verify there is a backup plan if the online experiment does not go as expected.

16. Verify the model has been calibrated.

17. Leverage automated hyperparameter tuning (if applicable).

18. Verify prediction bias has been analyzed.

19. Verify dataset has been analyzed for class imbalance.

20. Verify model experimented with regularization lambda to balance simplicity and training data fit.

21. Verify the same test samples are not being used over and over for test and validation.

22. Verify batch size hyperparameter is not too small.

23. Verify initial values in neural networks.

24. Verify the details of failed experiments are captured.

25. Verify the impact of wrong labels before investing in fixing them.

26. Verify a consistent set of metrics are used to analyze the results of online experiments.

27. Verify multiple hyperparameters are not tuned at the same time.



## Pre-deployment Checklist

1. How is the model going to be deployed (microservice, package, stand-alone app)?

2. Will the model run in real-time or in batch mode?

3. What computational resources are needed (GPUs, memory, etc.)?

4. How to handle model versioning and data versioning?

5. What tests will you run for the code? How often?

6. How to deploy a new version of the model (manual inspection, canary deployment, A/B testing)?

7. How often do we need to re-train the model? What is the upper bound (“at least”) and lower bound (“not sooner than”)?

8. How to unroll the deployment if something goes wrong?



## Deployment Checklist

1. Verify CI is in place
 
2. Verify tests for the full training pipeline

3. Check validation tests
 
4. Verfify functionality tests

5. Verify unit tests

6. Verify CD is in place

7. Verify CT is in place
 
8. Verify blue/green deployment is in place

9. Verify deployment of a model in shadow mode

10. Verify monitoring of memory/CPU consumption, latency, downtime, requests per second, and any other server resource metrics.

11. Verify monitoring of prediction confidence over time

12. Verify detection of failed model on a given datapoint and a corresponding fallback

13. Verify reasonable “ML Test Score” (Table 3-4) [10]

14. How does the model interact with other services or parts of the software? What could go wrong?



## Monitoring Checklist

1. How to gather the “hard” metrics (runtime, memory consumption, compute, disk space)?

2. What data quality and model metrics should be monitored in the production?

3. What KPI’s need to be monitored?

4. When deploying a new model, what metrics would be needed to decide between switching between the models?

5. How to monitor input drift and model degradation?

6. Are there any potential feedback loops that need to be addressed?

7. How to access the metrics (MLflow, comet.ml, Neptune)?

8. Verify data pipelines used to generate time-dependent features are performant for low latency.

9. Verify validation tests exist for data pipelines.

10. Verify model performance for the individual data slices.

11. Avoid using two different programming languages between training and deployment.

12. Ensure appropriate model scaling so that inference threshold is within the threshold.

13. Verify data quality inconsistencies are checked at source, ingestion into the lake, and ETL processing.

14. Verify cloud spend associated with the AI product is within budget.

15. Ensure optimization phase to balance quality with model depth and width.

16. Verify monitoring for data and concept drift.

17. Verify unnecessary calibration layers have been removed.

18. Verify monitoring to detect slow poisoning of the model due to intermittent errors.


### What is a Key Performance Indicator (KPI)?

[Key Performance Indicators (KPIs)][^kpi] are the critical (key) indicators of progress toward an intended result. KPIs provides a focus for strategic and operational improvement, create an analytical basis for decision making and help focus attention on what matters most. As Peter Drucker famously said, “What gets measured gets done.”

Managing with the use of KPIs includes setting targets (the desired level of performance) and tracking progress against that target. Managing with KPIs often means working to improve leading indicators that will later drive lagging benefits. Leading indicators are precursors of future success; lagging indicators show how successful the organization was at achieving results in the past. 


----------



## 5 Steps to follow for Successful AI Project

Using machine learning to help your business achieve edge on competition requires a plan and roadmap [13]. 

You cannot simply hire a group of data scientists and hope that they will be able to produce results for the business.

1. Focus on the Business Problem

Identify Business Problem

Where are the hidden data resources that you can take advantage of?

2. The Machine Learning Cycle

3. Pilot Project

Step 1: Define an opportunity for growth
Step 2: Conduct pilot project with your concrete idea from Step 1
Step 3: Evaluation
Step 4: Next actions

4. Determining the Best Learning Model

5. Tools to determine algorithm selection



## Common Machine Learning Techniques for AI Development

The article [14] discusses ten ML techniques for AI development:

Supervised ML

1. Regression
2. Classification
3. Transfer Learning
4. Ensemble Methods

Unsupervised ML

5. Clustering
6. Neural Networks and Deep Learning
7. Dimensionality Reduction (Supervised and Unsupervised)
8. Word Embeddings (dimensionality reduction)
9. Natural Language Processing 
10. Reinforcement Learning 



----------



## NLP Checklist

There is an NLP checklist given in [7]. 

### NLP Python Libraries

- spacy
- NLTK
- genism
- lexnlp
- Holmes
- Pytorch-Transformers

### Text Preprocessing

Text Preprocessing is the data cleaning process for an NLP application. 

When we are dealing with text removing null values and imputing them with mean and median isn’t enough.

- Removing punctuations like . , ! $( ) * % @
- Removing URLs
- Conver to Lowercase
- Converting numbers into words / removing numbers

Some code examples using NLTK are given for the following:

- Removing Stop words
- Tokenization
- Stemming
- Lemmatization
- Part-of-Speech Tagging

### Word Embeddings

Word Embeddings are used to convert the text into a vector of numerical values. The dimensions differ for each model. 

Word Embeddings help in faster calculations and reduce storage space. 

There are two types of word embeddings: Context-less and Context-driven.

**Context-Less Word Embeddings**

These models are used to convert each word to vector without taking into what situation the text was written. 

They focus more on the statistical part of the sentence structure rather than the context. 

TF-IDF uses how many times (frequency) a word is used in a sentence and gives priority accordingly.

- Term Frequency-Inverse Document Frequency (TF-IDF)
- Word2Vec
- GloVe

**Context-Driven Word Embeddings**

Context-Driven Word Embeddings holds the context of the sentence together. 

These models will be able to differentiate between the nuances of how a single word can be used in different contexts. 

This drastically increases the model’s understanding of the data compared to Context-less Word Embeddings.

- Elmo
- OpenAI GPT
- BERT
- RoBERTa
- ALBERT
- ELECTRA
- Distil BERT
- XLNet

Pre-trained word embeddings:

- Word2Vec (Google, 2013), uses Skip Gram and CBOW
- Vectors trained on Google News (1.5GB)
- Stanford Named Entity Recognizer (NER)
- LexPredict: pre-trained word embedding models for legal or regulatory text

### Text Similarity

Text Similarity means to find out how similar sentences or words are (King and Male), (Queen, Female, Mother. These groups are similar to each other this can be found out using text similarity.

Note: The words have to be converted to vectors using any of the models above to find the similarity. 

1. Euclidean distance
2. Cosine Similarity

### Named Entity Recognition

NER is the process to find the important labels that are present in the text.


### NLP Deep Learning Models

- Seq2Seq Models
- RNN
- N-gram Language Models
- LSTM



## References

[1] [4 Reasons Why You Shouldn’t Use Machine Learning](https://towardsdatascience.com/4-reasons-why-you-shouldnt-use-machine-learning-639d1d99fe11?source=rss----7f60cf5620c9---4&gi=204e8d695029)

[2] [AI Checklist](https://towardsdatascience.com/the-ai-checklist-fe2d76907673)


[3] [Machine Learning Performance Improvement Cheat Sheet](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)

[4] [Deploy Your Predictive Model To Production](https://machinelearningmastery.com/deploy-machine-learning-model-to-production/)

[5] [Feature engineering A-Z](https://towardsdatascience.com/feature-engineering-a-z-aa8ce9639632)

[6] [Dimensionality Reduction Explained](https://towardsdatascience.com/dimensionality-reduction-explained-5ae45ae3058e)


[7] [NLP Cheatsheet](https://medium.com/javarevisited/nlp-cheatsheet-2b19ebcc5d2e)


[8] [ML Checklist — Best Practices for a Successful Model Deployment](https://medium.com/analytics-vidhya/ml-checklist-best-practices-for-a-successful-model-deployment-2cff5495efed)

[9] [Machine Learning Model Deployment — A Simple Checklist](https://towardsdatascience.com/machine-learning-model-deployment-a-simplistic-checklist-dc5558a88d1b)

[10] [Machine Learning Project Checklist](https://github.com/sjosund/ml-project-checklist)

[11] [Deploying Machine Learning Models: A Checklist](https://twolodzko.github.io/ml-checklist)


[12] [8 Mistakes to avoid while using Machine Learning](https://medium.com/@monodeepets77/8-mistakes-to-avoid-while-using-machine-learning-d61af954b9c9)

[13] [5 Steps to follow for Successful Machine Learning Project](https://addiai.com/successful-machine-learning-project/)

[14] [10 Machine Learning Techniques for AI Development](https://daffodilsw.medium.com/10-machine-learning-techniques-for-ai-development-15df0953f05f)


[15] [The Data Quality Hierarchy of Needs](https://www.kdnuggets.com/2022/08/data-quality-hierarchy-needs.html)

[16] [Major Problems of Machine Learning Datasets: Part 3](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-3-eae18ab40eda)


[17] [7 Tips for Choosing the Right Machine Learning Infrastructure](https://www.aiiottalk.com/right-machine-learning-infrastructure/)


[^ai_hierarchy]: https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007

[^kpi]: https://kpi.org/KPI-Basics
