# AI Checklists

The first step is to understand how AI projects are different from traditional software projects [8] and [9].

## Getting Started

Here is a checklist for transforming features for better model performance.

The software development lifecycle (SDLC) of an AI project can be divided into six stages [3]:

1. **Problem definition:** The formative stage of defining the scope, value definition, timelines, governance, resources associated with the deliverable.

2. **Dataset Selection:** This stage can take a few hours or a few months depending on the overall data platform maturity and hygiene. Data is the lifeblood of ML, so getting the right and reliable datasets is crucial.

3. **Data Preparation:** Real-world data is messy. Understanding data properties and preparing properly can save endless hours down the line in debugging.

4. **Design:** This phase involves feature selection, reasoning algorithms, decomposing the problem, and formulating the right model algorithms.

5. **Training:** Building the model, evaluating with the hold-out examples, and online experimentation.

6. **Deployment:** Once the model is trained and tested to verify that it met the business requirements for model accuracy and other performance metrics, the model is ready for deployment. There are two common approaches to deployment of ML models to production: embed models into a web server or offload the model to an external service. Both ML model serving approaches have pros and cons.

7. **Monitoring:** This is the post-deployment phase involving observability of the model and ML pipelines, refresh of the model with new data, and tracking success metrics in the context of the original problem.

## AI/ML Checklists

Here are some articles with tips on AI/ML:

[AI Checklist](https://towardsdatascience.com/the-ai-checklist-fe2d76907673)

[ML Checklist — Best Practices for a Successful Model Deployment](https://medium.com/analytics-vidhya/ml-checklist-best-practices-for-a-successful-model-deployment-2cff5495efed)

[Machine Learning Model Deployment — A Simple Checklist](https://towardsdatascience.com/machine-learning-model-deployment-a-simplistic-checklist-dc5558a88d1b)

[The Machine Learning Engineer’s Checklist: Best Practices for Reliable Models](https://machinelearningmastery.com/the-machine-learning-engineers-checklist-best-practices-for-reliable-models/)

[Machine Learning Performance Improvement Cheat Sheet](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)

[Deploy Your Predictive Model To Production](https://machinelearningmastery.com/deploy-machine-learning-model-to-production/)

## Feature Engineering

Feature engineering is the process of transforming data to extract valuable information.

If done correctly, feature engineering can play even a bigger role in model performance than hyperparameter tuning.

A checklist for transforming features for better model performance is given in [3].

The article [4] explains and implements PCA in Python.

## Common Architectures

The two most common architectures for ML models are:

1. **Precomputed Model Prediction:** This is one of the earliest used and simplest architecture for serving machine learning models. It is an indirect method for serving the model where we precompute predictions for all possible combinations of input variables and store them in a database. This architecture is generally used in recommendation systems — recommendations are precomputed and stored and shown to the user at login.

2. **Microservice Based Model Serving:** The model is served independently of the application, and predictions are provided in real-time as per request. This type of architecture provides flexibility in terms of model training and deployment.

### When to retrain the model

The performance of an ML model degrades over time in production, so it is best to evaluate retraining requirements before model serving. Based on the use case, model monitoring, and evaluation, one can decide when to retrain the model again. One good way to decide on retraining time is to use out-of-time analysis on different time windows.

### How to retrain the model

Retraining is essential and it helps to keep the model up to date. There are broadly two ways to retrain machine learning models — online & offline training.

Online Training: The model is re-trained while in production. True labels are circulated back to the model at a certain interval to update/ retrain the model. This requires a separate architecture and is generally hard to implement.

For example, when we predict ad-click probability we can get feedback (clicked or not clicked) which can be used to update the model online.

Offline Training: The model is re-trained from scratch, so we have full control over the new model and data to train. The new model is pushed in production using A/B testing or shadow testing.

## AI Process Checklists

### Problem Definition Checklist

1. Verify there is quantifiable business value in solving the problem.

2. Verify that simpler alternatives (such as hand-crafted heuristics) are not sufficient to address the problem.

3. Ensure that the problem has been decomposed into the smallest possible units.

4. Clear understanding of how the AI output will be applied to accomplish the desired business outcome.

5. Clear measurable metric(s) to measure the success of the solution.

6. Clear understanding of precision versus recall tradeoff of the problem.

7. Verify impact when the logistic classification prediction is incorrect.

8. Ensure project costs include the cost of managing the corresponding data pipelines.

### Dataset Selection Checklist

1. Verify the meaning of the dataset attributes.

2. Verify the derived metrics used in the project are standardized.

3. Verify data from warehouse is not stale due to data pipeline errors.

4. Verify schema compliance of the dataset.

5. Verify datasets comply with data rights regulations (such as GDPR, CCPA, etc.).

6. Ensure there is a clear change management process for dataset schema changes.

7. Verify dataset is not biased.

8. Verify the datasets being used are not orphaned (without data stewards).

### Data Preparation Checklist

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

### Design Checklist

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

### Training Checklist

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

### Pre-deployment Checklist

1. How is the model going to be deployed (microservice, package, stand-alone app)?

2. Will the model run in real-time or in batch mode?

3. What computational resources are needed (GPUs, memory, etc.)?

4. How to handle model versioning and data versioning?

5. What tests will you run for the code? How often?

6. How to deploy a new version of the model (manual inspection, canary deployment, A/B testing)?

7. How often do we need to re-train the model? What is the upper bound (“at least”) and lower bound (“not sooner than”)?

8. How to unroll the deployment if something goes wrong?

### Deployment Checklist

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

13. Verify reasonable “ML Test Score” (Table 3-4) in [5]

14. How does the model interact with other services or parts of the software? What could go wrong?

### Monitoring Checklist

1. How to gather the “hard” metrics (runtime, memory consumption, compute, disk space)?

2. What data quality and model metrics should be monitored in the production?

3. What KPI’s need to be monitored?

4. When deploying a new model, what metrics would be needed to decide between switching between the models?

5. How to monitor input drift and model drift / degradation?

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

### Key Performance Indicator (KPI)

[Key Performance Indicators (KPIs)][^kpi] are the essential indicators of progress toward an intended result.
KPIs provides a focus for strategic and operational improvement, create an analytical basis for decision making and help focus attention on what matters most. As Peter Drucker famously said, “What gets measured gets done.”

Managing with the use of KPIs includes setting targets (the desired level of performance) and tracking progress against that target. Managing with KPIs often means working to improve leading indicators that will later drive lagging benefits. Leading indicators are precursors of future success; lagging indicators show how successful the organization was at achieving results in the past.

## Common Machine Learning Techniques

The article [7] discusses ten ML techniques for AI development:

Supervised ML

1. Regression
2. Classification
3. Transfer Learning
4. Ensemble Methods

Unsupervised ML

1. Clustering
2. Neural Networks and Deep Learning
3. Dimensionality Reduction (Supervised and Unsupervised)
4. Word Embeddings (dimensionality reduction)
5. Natural Language Processing
6. Reinforcement Learning

## NLP Checklist

There is an NLP checklist given in [1] and project guide in [2].

## References

[1]: [NLP Cheatsheet](https://medium.com/javarevisited/nlp-cheatsheet-2b19ebcc5d2e)

[2]: [LLMs Project Guide: Key Considerations](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/getting-started/llmops-checklist)

[3]: [Feature engineering A-Z](https://medium.com/data-science/feature-engineering-a-z-aa8ce9639632)

[4]: [Dimensionality Reduction Explained](https://towardsdatascience.com/dimensionality-reduction-explained-5ae45ae3058e)

[5]: [Machine Learning Project Checklist](https://github.com/sjosund/ml-project-checklist)

[6]: [Deploying Machine Learning Models: A Checklist](https://twolodzko.github.io/ml-checklist)

[7]: [10 Machine Learning Techniques for AI Development](https://daffodilsw.medium.com/10-machine-learning-techniques-for-ai-development-15df0953f05f)

[8]: [How are AI Projects Different](https://pub.towardsai.net/how-are-ai-projects-different-ccfdedb7ff99)

[9]: [Elements of AI Project Management](https://pub.towardsai.net/elements-of-ai-project-management-6cac1826bdbb)

[10]: [7 Tips for Choosing the Right Machine Learning Infrastructure](https://www.aiiottalk.com/right-machine-learning-infrastructure/)

----------

[The Data Quality Hierarchy of Needs](https://www.kdnuggets.com/2022/08/data-quality-hierarchy-needs.html)

[Major Problems of Machine Learning Datasets: Part 3](https://heartbeat.comet.ml/major-problems-of-machine-learning-datasets-part-3-eae18ab40eda)

[^kpi]: <https://kpi.org/KPI-Basics>
