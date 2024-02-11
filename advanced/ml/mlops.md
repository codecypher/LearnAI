# MLOps

## AI SDLC

In general, the ML model process involves eight stages which may also include data collection and/or data labeling [1]:

1. Data preparation
2. Feature engineering
3. Model design
4. Model training and optimization
5. Model evaluation
6. Model deployment
7. Model serving
8. Model monitoring


The software development lifecycle (SDLC) of an AI project can be divided into six stages [2]:

1. **Problem definition:** The formative stage of defining the scope, value definition, timelines, governance, resources associated with the deliverable.

2. **Dataset Selection:** This stage can take a few hours or a few months depending on the overall data platform maturity and hygiene. Data is the lifeblood of ML, so getting the right and reliable datasets is crucial.

3. **Data Preparation:** Real-world data is messy. Understanding data properties and preparing properly can save endless hours down the line in debugging.

4. **Design:** This phase involves feature selection, reasoning algorithms, decomposing the problem, and formulating the right model algorithms.

5. **Training:** Building the model, evaluating with the hold-out examples, and online experimentation. 

6. **Deployment:** Once the model is trained and tested to verify that it met the business requirements for model accuracy and other performance metrics, the model is ready for deployment. There are two common approaches to deployment of ML models to production: embed models into a web server or offload the model to an external service. Both ML model serving approaches have pros and cons.

7. **Monitoring:** This is the post-deployment phase involving observability of the model and ML pipelines, refresh of the model with new data, and tracking success metrics in the context of the original problem.


**Serving:** Model serving refers to the use of a platform to deploy ML models at massive scale. Examples: Seldon, KFServing, and Ray Serve.

**Monitoring:** This is the post-deployment phase involving observability of the model and ML pipelines, refresh of the model with new data, and tracking success metrics in the context of the original problem. Key items to monitor are:  model drift, data drift, model failure, and system performance. Examples: Evidently.ai, Arize.ai, Arthur.ai, Fiddler.ai, Valohai.com, or whylabs.ai.


The two most common architectures for ML model serving are:

1. **Precomputed Model Prediction:** This is one of the earliest used and simplest architecture for serving machine learning models. It is an indirect method for serving the model, where we precompute predictions for all possible combinations of input variables and store them in a database. This architecture is generally used in recommendation systems — recommendations are precomputed and stored and shown to the user at login.

2. **Microservice Based Model Serving:** The model is served independently of the application, and predictions are provided in real-time as per request. This type of architecture provides flexibility in terms of model training and deployment.




## MLOps

Like DevOps, MLOps manages automated deployment, configuration, monitoring, resource management and testing and debugging.

Unlike DevOps, MLOps also might need to consider data verification, model analysis and re-verification, metadata management, feature engineering and the ML code itself.

But at the end of the day, the goal of MLOps is very similar. The goal of MLOps is to create a continuous development pipelines for machine learning models.

A pipeline that quickly allows data scientists and machine learning engineers to deploy, test and monitor their models to ensure that their models continue to act in the way they are expected to.

**Key ideas:** 

- Need to version code, data, and models
- Continuous delivery and continuous training

### Tracking Model Experiments

Unlike the traditional software development cycle, the model development cycle paradigm is different. 

A number of factors influence an ML model’s success in production. 

1. The outcome of a model is measured by its metrics such as an acceptable accuracy.

2. Many models and many ML libraries while tracking each experiment runs: metrics, parameters, artifacts, etc.

3. Data preparation (feature extractions, feature selection, standardized or normalized features, data imputations and encoding) are all important steps before the cleansed data lands into a feature store, accessible to your model training and testing phase or inference in deployment.

4. The choice of an ML framework for taming compute-intensive ML workloads: deep learning, distributed training, hyperparameter optimization (HPO), and inference.

5. The ability to easily deploy models in diverse environments at scale: part of web applications, inside mobile devices, as a web service in the cloud, etc.

### Managing Machine Learning Features

Feature stores address operational challenges. They provide a consistent set of data between training and inference. They avoid any data skew or inadvertent data leakage. They offer both customized capability of writing feature transformations, both on batch and streaming data, during the feature extraction process while training. And they allow request augmentation with historical data at inference which is common in large fraud and anomaly detection deployed models or recommendation systems.

### Observing and Monitoring Model in Production

**Data drift:** As we mentioned above, our quality and accuracy of the model depends on the quality of the data. Data is complex and never static, meaning what the original model was trained with the extracted features may not be as important over time. Some new features may emerge that need to be taken into account. 

**Model concept drift:** Many practitioners refer to this as model decay or model staleness. When the patterns of trained models no longer hold with the drifting data, the model is no longer valid because the relationships of its input features may not necessarily produce the model’s expected prediction. Thus, its accuracy degrades.

**Models fail over time:** Models fail for inexplicable reasons: a system failure or bad network connection; an overloaded system; a bad input or corrupted request. Detecting these failures’ root causes early or its frequency mitigates user bad experience or deters mistrust in the service if the user receives wrong or bogus outcomes.

**Systems degrade over load:** Constantly being vigilant of the health of your dedicated model servers or services deployed is just as important as monitoring the health of your data pipelines that transform data or your entire data infrastructure’s key components: data stores, web servers, routers, cluster nodes’ system health, etc.

### What could go wrong after deployment

What kind of problems Machine Learning applications might encounter over time.

- Changes in data distribution (data drifts): sudden changes in the features values.

- Model/concept drifts: how, why and when the performance of your model dropped.

- System performance: training pipelines failing, or taking long to run; very high latency...

- Outliers: the need to track the results and performances of a model in case of outliers or unplanned situations.

- Data quality: ensuring the data received in production is processed in the same way as the training data.



----------



## Modeling Pipeline

[Modeling Pipeline](./pipelines.md)

A _pipeline_ is a linear sequence of data preparation options, modeling operations, and prediction transform operations.

A pipeline allows the sequence of steps to be specified, evaluated, and used as an atomic unit.

**Pipeline:** A linear sequence of data preparation and modeling steps that can be treated as an atomic unit.



## Monitoring the model

Once a model has been deployed its behavior must be monitored. 

The predictive performance of a model is expected to degrade over time as the environment changes callef concept drift which occurs when the distributions of the input features or output target shift away from the distribution upon which the model was originally trained.

When concept drift has been detected, we need to retrain the ML model but detecting drift can difficult.

One strategy for monitoring is to use a metric from a deployed model that can be measured over time such as measuring the output distribution. The observed distribution can be compared to the training output distribution, and alerts can notify data scientists when the two quantities diverge.

Popular AI/ML deployment tools: TensorFlow Serving, MLflow, Kubeflow, Cortex, Seldon.io, BentoML, AWS SageMaker, Torchserve, Google AI.



## Monitoring Model in Production

Model monitoring is critical to model viability in the post deployment production stage which is often overlooked [6]. 

**Data drift over time:** The quality and accuracy of the model depends on the quality of the data which is complex and never static. The original model was trained with the extracted features may not be as important over time. Some new features may emerge that need to be taken into account. Such features drifts in data require retraining and redeploying the model because the distribution of the variables is no longer relevant.

**Model concept changes over time:** Many practitioners refer to this as model decay or model staleness. When the patterns of trained models no longer hold with the drifting data, the model is no longer valid because the relationships of its input features may not necessarily produce the expected prediction. This, model accuracy degrades.

**Models fail over time:** Models fail for inexplicable reasons: a system failure or bad network connection; an overloaded system; a bad input or corrupted request. Detecting these failures root causes early or its frequency mitigates bad user experience and deters mistrust in the service if the user receives wrong or bogus outcomes.

**Systems degrade over load:** Constantly being vigilant of the health of dedicated model servers or services deployed is also important: data stores, web servers, routers, cluster nodes’ system health, etc.

Collectively, these monitoring model concepts are called _model observability_ which is important in MLOps best practices. Monitoring the health of data and models should be part of the model development cycle.

NOTE: For model observability look to Evidently.ai, Arize.ai, Arthur.ai, Fiddler.ai, Valohai.com, or whylabs.ai.


Monitoring:

- performance (prediction vs actual, metrics, thresholds)
- model drift vs data drift
- model stability and population shift (PSI and CSI metrics)


### What could go wrong after deployment

Here are some of the problems Machine Learning applications can encounter over time [9]:

- Changes in data distribution (data drifts): sudden changes in the features values.

- Model/concept drifts: how, why and when the performance of your model dropped.

- System performance: training pipelines failing, or taking long to run; very high latency…

- Outliers: the need to track the results and performances of a model in case of outliers or unplanned situations.

- Data quality: ensuring the data received in production is processed in the same way as the training data.



### Why monitor ML models in Production

Here are some of the challenges a model can encounter in production [12]:

1. Data distribution changes: Why are there sudden changes in the values of my features?

2. Model Ownership in Production: Who owns the model in production? The DevOps team? Engineers? Data Scientists?

3. Training-Serving Skew: Why is the model giving poor results in production despite our rigorous testing and validation attempts during development?

4. Model/Concept drift: Why was the model performing well in production and suddenly the performance dipped over time?

5. Black box models: How to interpret and explain my model’s predictions in line with the business objective and to relevant stakeholders?

6. Concerted adversaries: How can I ensure the security of my model? Is my model being attacked?

7. Model Readiness: How to compare results from a newer version(s) of my model against the in-production version(s)?

8. Pipeline health issues: Why does the training pipeline fail when executed? Why does a retraining job take so long to run?

9. Underperforming system: Why is the latency of the predictive service very high? Why am I getting vastly varying latencies for my different models?

10. Cases of extreme events (Outliers): How to track the effect and performance of my model in extreme and unplanned situations?

11. Data Quality Issues: How to ensure the production data is being processed in the same way as the training data was?



## Testing online inference models

ML system testing is more complex of a challenge than testing manually coded systems because ML system behavior depends strongly on data and models that cannot be specified a priori [14].

Figure: The Machine Learning Test Pyramid.

ML requires more testing than traditional software engineering [14].

### A/B Test

To measure the impact of a new model, the evaluation step needs to include statistical A/B tests where the users are split into two distinct non-overlapping cohorts [16]. 

The population of users must be split into statistically identical populations that each experience a different algorithm [15].



## Verification and Validation (V&V)

To earn trust, any engineered system must go through a verification and validation (V&V) process (Russell and Norvig, 27.3.4):

- Verification means that the product satisfies the specifications. 

- Validation means ensuring that the specifications actually meet the needs of the user and other affected parties. 

- We need to verify the data that these systems learn from. 

- We need to verify the accuracy and fairness of the results, even in the face of uncertainty that makes an exact result unknowable. 

- We need to verify that adversaries cannot unduly influence the model, nor steal information by querying the resulting model.



## Model Validation Tools

Whether we are developing a simple model or a complex model, model validation is essential to measure the quality of the work [11].

We need to measure every moving part to ensure the model is adequate from validating the data, the methodology, and the machine learning model metrics [11]. 

There are many techniques to do machine learning validation but the article [11] describes three python packages for validating machine learning models: Evidently, Deepchecks, and TFDV. 

### Evidently

**Evidently** is an open-source python package to analyze and monitor machine learning models. The package is explicitly developed to establish an easy-to-monitor machine learning dashboard and detect drift in the data. It's specifically designed with production in mind, so it's better used when a data pipeline is there. However, you could still use it even in the development phase.

### Deepchecks

**Deepchecks** is a python package to validate  machine learning models with a few lines. 

Many APIs are available for detecting data drift, label drift, train-test comparison, evaluating models, and many more. 

Deepchecks is perfect to use in the research phase and before your model goes into [production](https://docs.deepchecks.com/en/stable/user-guide/when_should_you_use.html). 

### TensorFlow Data Validation

**TensorFlow Data Validation (TFDV)** is a python package designed to manage data quality issues. 

TFDV is used to automatically describe the data statistic, infer the data schema, and detect any anomalies in the incoming data.



## MLOps Tools

### MLflow

MLflow is an open source platform for managing the end-to-end machine learning lifecycle [17]. 

MLflow tackles four primary functions [17]:

- Tracking experiments to record and compare parameters and results (MLflow Tracking).

- Packaging ML code in a reusable, reproducible form in order to share with other data scientists or transfer to production (MLflow Projects).

- Managing and deploying models from a variety of ML libraries to a variety of model serving and inference platforms (MLflow Models).

- Providing a central model store to collaboratively manage the full lifecycle of an MLflow Model, including model versioning, stage transitions, and annotations (MLflow Model Registry).

MLflow is library-agnostic which means we can use it with any machine learning library, and in any programming language since all functions are accessible through a REST API and CLI. 

For convenience, the project includes APIs for Python, R, and Java.


### TensorFlow Extended

**[TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx)** is an end-to-end platform for deploying production ML pipelines
When you are ready to move your models from research to production, use TFX to create and manage a production pipeline.

A TFX pipeline is a sequence of components that implement an ML pipeline which is specifically designed for scalable, high-performance machine learning tasks. Components are built using TFX libraries which can also be used individually.

- ML Metadata
- TensorFlow Data Validation
- TensorFlow Transform
- Designing TensorFlow Modeling Code For TFX
- TensorFlow Model Analysis
- TensorFlow Serving and TensorFlow Lite


### TFX Tutorials

[The TFX User Guide](https://github.com/tensorflow/tfx/blob/master/docs/guide/index.md)

[TensorFlow in Production tutorials](https://www.tensorflow.org/tfx/tutorials)

[Train and serve a TensorFlow model with TensorFlow Serving](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)

[Deploying models to production with TensorFlow model server](https://towardsdatascience.com/deploying-models-to-production-with-tensorflow-model-server-225a81859031)


### TensorFlow Serving

**TensorFlow Serving** is a flexible, high-performance serving system for machine learning models that is designed for production environments. 

TensorFlow Serving makes it easy to deploy new algorithms and experiments while keeping the same server architecture and APIs. 

TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data.


### Kubeflow

Kubeflow is the machine learning toolkit for Kubernetes.

Kubeflow is an open-source Kubernetes-based machine learning stack used to simplify the process of deploying, scaling, and managing machine learning systems. 

Kubeflow has several different components that address various steps in building an ML system. 

Kubeflow also allows you to reuse elements of a pipeline for later projects.

You can adapt the configuration to choose the platforms and services that you want to use for each stage of the ML workflow:

1. data preparation
2. model training,
3. prediction serving
4. service management

Some of the available tools in Kubeflow:

- Data exploration
- Building and training models
- Hyper-parameter tuning (Katib)
- Model versioning
- Model deployment
- Model production


----------



## MLflow Examples

[Managing Machine Learning Lifecycles with MLflow](https://kedion.medium.com/managing-machine-learning-lifecycles-with-mlflow-f230a03c4803)

[MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)



## MLOps Examples

[Build an Anomaly Detection Pipeline with Isolation Forest and Kedro](https://towardsdatascience.com/build-an-anomaly-detection-pipeline-with-isolation-forest-and-kedro-db5f4437bfab)

[Building Scalable Edge AI Deployments with FleetTrackr](https://medium.com/@Smartcow_ai/building-scalable-edge-ai-deployments-with-fleettrackr-40f9a9ab5d65)



## References

[1] [What Is MLOps And Why Your Team Should Implement It](https://medium.com/smb-lite/what-is-mlops-and-why-your-team-should-implement-it-b05b741cdf94)

[2] [AI Checklist](https://towardsdatascience.com/the-ai-checklist-fe2d76907673)


[6] [Considerations for Deploying Machine Learning Models in Production](https://towardsdatascience.com/considerations-for-deploying-machine-learning-models-in-production-89d38d96cc23?source=rss----7f60cf5620c9---4)

[7] [Model Drift in Machine Learning](https://towardsdatascience.com/model-drift-in-machine-learning-8023e3d08217)


[8] [Model monitoring metrics](https://medium.com/prosus-ai-tech-blog/model-monitoring-1849fb3afc1e)

[9] [A Comprehensive Guide on How to Monitor Models in Production](https://medium.com/artificialis/a-comprehensive-guide-on-how-to-monitor-your-models-in-production-c069a8431723)

[10] [A Beginner’s Guide to End to End Machine Learning](https://towardsdatascience.com/a-beginners-guide-to-end-to-end-machine-learning-a42949e15a47)


[11] [Top 3 Python Packages for Machine Learning Validation](https://towardsdatascience.com/top-3-python-packages-for-machine-learning-validation-2df17ee2e13d)


[12] [A Comprehensive Guide on How to Monitor Your Models in Production](https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide)


[13] [5 Types of ML Accelerators](https://pub.towardsai.net/5-types-of-ml-accelerators-767d26a643de)


[14] [The Challenges of Online Inference (Deployment Series: Guide 04)](https://mlinproduction.com/the-challenges-of-online-inference-deployment-series-04/)

[15] Test-Driven Machine Learning Development (Deployment Series: Guide 07)

[16] [A/B Testing Machine Learning Models (Deployment Series: Guide 08)](https://mlinproduction.com/ab-test-ml-models-deployment-series-08/)


[17] [Mlflow documentation](https://mlflow.org/docs/latest/index.html)



[Design Patterns in Machine Learning for MLOps](https://towardsdatascience.com/design-patterns-in-machine-learning-for-mlops-a3f63f745ce4)


[Machine Learning Operations](https://ml-ops.org)

[The Secret of Delivering Machine Learning to Production](https://towardsdatascience.com/the-secret-of-delivering-machine-learning-to-production-1f6681f5e30c)

[How to Evaluate Different Machine Learning Deployment Solutions](https://wallarooai.medium.com/how-to-evaluate-different-machine-learning-deployment-solutions-adf51fe76a4b)


[Essential guide to Machine Learning Model Monitoring in Production](https://towardsdatascience.com/essential-guide-to-machine-learning-model-monitoring-in-production-2fbb36985108?gi=d5a42b3b9e9)

[Monitoring Machine Learning Models in Production: Why and How?](https://pub.towardsai.net/monitoring-machine-learning-models-in-production-why-and-how-b2556af448aa)

[Best Practices For Monitoring Machine Learning Models In Production](https://medium.com/artificialis/best-practices-for-monitoring-machine-learning-models-in-production-b8996f2a85b3)

[The Ultimate Guide to Deploying Machine Learning Models](https://mlinproduction.com/deploying-machine-learning-models/)


[Automating Data Drift Thresholding in Machine Learning Systems]https://towardsdatascience.com/automating-data-drift-thresholding-in-machine-learning-systems-524e6259f59)

[Monitoring unstructured data for LLM and NLP](https://towardsdatascience.com/monitoring-unstructured-data-for-llm-and-nlp-efff42704e5b?source=rss----7f60cf5620c9---4)

[SHAP for Drift Detection: Effective Data Shift Monitoring](https://towardsdatascience.com/shap-for-drift-detection-effective-data-shift-monitoring-c7fb9590adb0)




