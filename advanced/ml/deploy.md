# Deployment

## Overview

Consider using model life cycle development and management platforms such as MLflow, DVC, Weights & Biases, or SageMaker Studio. And Ray, Ray Tune, Ray Train (formerly Ray SGD), PyTorch and TensorFlow for distributed, compute-intensive, and deep learning ML workloads [1].

NOTE: Consider feature stores as part of your model development process. Look to Feast, Tecton, SageMaker, and Databricks for feature stores. 


## Deploying, Serving, and Inferencing Models at Scale

Once the model is trained and tested and meets the business requirements for model accuracy, there are seven crucial requirements for scalable model serving frameworks to consider are [1]:

**Framework agnostic:** A model serving framework should be ML framework agnostic, so it can deploy any common model built with common ML frameworks such as PyTorch, TensorFlow, XGBoost, or Scikit-learn, each with its own algorithms and model architectures.

**Business Logic:** Model prediction often requires preprocessing, post processing or ability to augment request data by connecting to a feature store or any other data store for validation. Model serving should allow this as part of its inference.

**Model Replication:** Some models are compute-intensive or network-bound. Therefor, the framework should be able to fan out requests to model replicas, load balancing among replicas to support parallel request handling during peak traffic.

**Request Batching:** Not all models in production are employed for real-time serving. Often, models are scored in large batches of requests. For deep learning models, parallelizing image requests to multiple cores and taking advantage of hardware accelerators to expedite batch scoring and utilize hardware resources.

**High Concurrency and Low Latency:** Models in production require real-time inference with low latency while handling bursts of heavy traffic of requests which is crucial for best user experience to receive millisecond responses on prediction requests.

**Model Deployment CLI and APIs:** A ML engineer responsible for deploying a model should be able to easily use model server’s deployment APIs or command line interfaces (CLI) to deploy model artifacts into production which allows model deployment from within an existing CI/CD pipeline or workflow.

**Patterns of Models in Production:** As ML applications are increasingly becoming pervasive in all sectors of industry, models trained for these ML applications are complex and composite. 

Thus, models do not exist in isolation and they do not predict results singularly. They operate jointly and often in four model patterns: pipeline, ensemble, business logic, and online learning. Each pattern has its purpose and merits.

Machine Learning engineers adopt two common approaches to deploy these patterns of models in production: embed models into a web server and offload to an external service. Each approach has its own pros and cons.

NOTE: Look to Seldon, KFServing, or Ray Serve for all these seven requirements.



## Deployment Examples

[Serving ML Models with gRPC](https://towardsdatascience.com/serving-ml-models-with-grpc-2116cf8374dd)

[The Nice Way To Deploy An ML Model Using Docker](https://towardsdatascience.com/the-nice-way-to-deploy-an-ml-model-using-docker-91995f072fe8)

[Deploying Your First Machine Learning API using FastAPI and Deta](https://www.kdnuggets.com/2021/10/deploying-first-machine-learning-api.html)

[Deploy MNIST Trained Model as a Web Service using Flask](https://towardsdatascience.com/deploy-mnist-trained-model-as-a-web-service-ba333d233a5d)

[Docker Networking](https://medium.com/codex/docker-networking-d3d259c91821)



## Deployment Frameworks

### BentoML

[BentoML](https://www.bentoml.com) is an open MLOps platform that simplifies ML model deployment and enables you to serve your models at production scale in minutes. 

BentoML claims to be the easiest way to turn your ML models into production-ready API endpoints.

- High performance model serving, all in Python.

- Standardize model packaging and ML service definition to streamline deployment.

- Support all major machine-learning training frameworks.

- Deploy and operate ML serving workload at scale on Kubernetes via Yatai.

### Fastapi

Similar to the flask, [fastapi](https://fastapi.tiangolo.com) is also a popular framework in Python for web backend development. 

Fastapi focuses on using the least code to write regular Web APIs which is good if the backend is not too complex. 

### gRPC

The trend now is toward gRPC for microservices since it is more secure, faster, and more robust (especially with IoT).

[gRPC with REST and Open APIs](https://grpc.io/blog/coreos/)

- gRPC was recommended for developing microservices by my professor in Distributed Computing course.

- gRPC uses HTTP/2 which enables applications to present both a HTTP 1.1 REST/JSON API and an efficient gRPC interface on a single TCP port (available for Go). 

- gRPC provides developers with compatibility with the REST web ecosystem while advancing a new, high-efficiency RPC protocol. 

- With the recent release of Go 1.6, Go ships with a stable net/http2 package by default.



## Cloud Services

### Deta

[Deta](https://www.deta.sh) is a free, friendly cloud platform. 

**Deta Micros** is a service to deploy Python and Node.js apps/APIs on the internet in seconds. 

**Deta Base** is a super easy to use production-grade NoSQL database that comes with unlimited storage.

**Deta Drive** is an easy to use cloud storage solution by Deta (get 10GB for free)

### DagsHub

[DagsHub](https://dagshub.com) is the GitHub for ML and data projects.

### Digital Ocean

[Digital Ocean](https://www.digitalocean.com/) is one of the best cloud VM droplet hosting services at an affordable price. 

### Hetzner

[Hetzner](https://www.hetzner.com/cloud) is located in Germany and currently one of the cheapest cloud hosting services. 

### Streamlit

[Streamlit](https://streamlit.io/) is a Python package that makes it easy to create dashboards and data applications without the need for any front-end programming expertise

All in Python. All for free.

### Web3.Storage

[Web3.Storage](https://web3.storage)

[Store Unlimited Data for Free on IPFS and Retrieve it with Python](https://towardsdatascience.com/store-unlimited-data-for-free-on-ipfs-and-retrieve-it-with-python-8db7297b493d)


## PaaS

Heroku has traditionally been a great option for platform-as-a-service solution (PaaS). However, Heroku has been criticized a lot lately for their response to a serious security incident in which [their communication][^heroku_incident] about the whole event was abysmal.

Here are [few alternatives][^three_paas_alternatives] for PaaS:

[DigitalOcean](https://www.digitalocean.com/)

[Fly](https://fly.io)

[Render](https://render.com)

[porter](https://www.getporter.dev)



-------



## References

[1] [Considerations for Deploying Machine Learning Models in Production](https://towardsdatascience.com/considerations-for-deploying-machine-learning-models-in-production-89d38d96cc23?source=rss----7f60cf5620c9---4)

[2] [Machine Learning in Production: Why Is It So Hard and So Many Fail?](https://towardsdatascience.com/machine-learning-in-production-why-is-it-so-difficult-28ce74bfc732)

[3] [ML Troubleshooting Is Too Hard Today (But It Doesn’t Have to Be That Way)](https://towardsdatascience.com/ml-troubleshooting-is-too-hard-today-but-it-doesnt-have-to-be-that-way-fbf9a10cac05)

[4] [How to Deploy ML Models to Production](https://medium.com/mlearning-ai/how-to-deploy-ml-models-to-production-ab7ce1c353de)

[5] [Best practices in the deployment of AI models](https://nagahemachandchinta.medium.com/best-practices-in-the-deployment-of-ai-models-c929c3146416)


[The Easiest Way to Deploy Your ML/DL Models: Streamlit + BentoML + DagsHub](https://towardsdatascience.com/the-easiest-way-to-deploy-your-ml-dl-models-in-2022-streamlit-bentoml-dagshub-ccf29c901dac)

[AWS vs. Digital Ocean vs. Hetzner Cloud](https://betterprogramming.pub/aws-vs-digital-ocean-vs-hetzner-cloud-which-has-the-best-value-for-money-bd9cb3c06dee)

[Serving Python Machine Learning Models With Ease using MLServer](https://pub.towardsai.net/serving-python-machine-learning-models-with-ease-29e1ba9e2155)



[^three_paas_alternatives]: https://medium.com/codex/3-paas-alternatives-to-heroku-db8fc750cc6f?source=rss----29038077e4c6---4 "3 PaaS Alternatives to Heroku"

[^heroku_incident]: https://www.theregister.com/2022/05/04/heroku_security_communication_dubbed_complete/ "Communication around Heroku security incident dubbed 'train wreck'"

