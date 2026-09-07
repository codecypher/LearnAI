# Web Development

Here are some notes on web development.


## Python Web Development

- comet.ml
- DagsHub and dvc
- FastAPI

- Deta
- Streamlit
- Mogenius

- Beautiful Soup
- Django
- Selenium


### Flask vs FastAPI

Python has many web frameworks, the most popular being Django and Flask.

FastAPI is a web framework for Python and in many ways resembles the simplicity of Flask. What makes FastAPI different is that it runs on ASGI web servers (such as uvicorn) while Flask only runs on WSGI web servers. This difference can result in a huge performance gap.

In short: with asynchronous code, threads can do more work in the same amount of time which results in more work done per unit time that results in a performance boost.

A comparison of two different RestAPI frameworks [1]:

**Flask** was released in 2010, a micro web framework written in python to support the deployment of web applications with a minimal amount of code.

Flask is designed to be an easy setup that is flexible and fast to deploy as a microservice.

Flask is built on **WSGI (Python Web Server Gateway Interface)** which means the server will tie up a worker for each request.

**FastAPI** was released in 2018. FastAPI works similarly to Flask which supports the deployment of web applications with a minimal amount of code.

FastAPI is faster compare to Flask since it is built on **ASGI (Asynchronous Server Gateway Interface)** which supports concurrency / asynchronous code by declaring the endpoints with `async def` syntax.

FastAPI also supports Swagger documentation. Upon deploying with FastAPI Framework, FastAPI will generate documentation (/docs) and creates an interactive GUI (Swagger UI) which allows developers to test the API endpoints more conveniently.

### Comparison of Flask and FastAPI

Flask does not provide validation on the data format which means the user can pass any type of data such as string or integer etc. (Alternatively, a validation script on the input data receive can be built into the script, but this will require additional effort).

FastAPI allows developers to declare additional criteria and validation on the parameter received.


The error messages display in Flask are HTML pages by default. In FastAPI the error messages displayed are in JSON format.

Flask does not support asynchronous tasks while FastAPI supports asynchronous tasks.

Thus, it is probably best to adopt FastAPI in the future since it has asynchronous functions and automated generated documents which is very detailed and complete. In addition, the effort required to deploy using FastAPI is the same as Flask.


### FastApi

[How to Deploy a Secure API with FastAPI, Docker, and Traefik])https://towardsdatascience.com/how-to-deploy-a-secure-api-with-fastapi-docker-and-traefik-b1ca065b100f)


[An introduction to asyncio in python](https://medium.com/geekculture/an-introduction-to-asyncio-in-python-5d3e19a02263)

[Pydantic or dataclasses? Why not both? Convert Between Them](https://towardsdatascience.com/pydantic-or-dataclasses-why-not-both-convert-between-them-ba382f0f9a9c)


[Creating Secure API with EasyAuth and FastAPI](https://itnext.io/creating-secure-apis-with-easyauth-fastapi-6996a5e42d07)

[The Nice Way To Deploy An ML Model using Docker and VSCode](https://towardsdatascience.com/the-nice-way-to-deploy-an-ml-model-using-docker-91995f072fe8)


### Pydantic

All the data validation is performed under the hood by Pydantic, so you get all the benefits from it. And you know you are in good hands.

You can use the same type declarations with str, float, bool and many other complex data types.

Pydantic provides data validation and settings management using Python type annotations.

pydantic enforces type hints at runtime and provides user friendly errors when data is invalid.

Define how data should be in pure with canonical Python; validate it with pydantic.



## Streamlit vs Deta

Basically, Streamlit Cloud is tied closely to your GitHub repo (Pros and Cons) which means you will need admin access to be able to setup the repo with Streamlit.

Deta does not require GitHub account or repo access which is perhaps more flexible for personal use.

[Streamlit Docs](https://docs.streamlit.io)

[Deta Docs](https://docs.deta.sh/docs/home)

[Getting Started](https://docs.deta.sh/docs/micros/getting_started/)



## State Preservation

State preservation is a mechanism to store data [3].

It comes in three variations:

- Cookies: they allow you to store small bits of information on the client, and it's sent to the server during an HTTP request.

- Session variables: a unique identifier is used to associate the information stored on the server with a particular client.

- Passing data at each request-response cycle: this variation allows you to store data on the web page.



## Authentication vs Authorization

Authentication is the process which ensures that someone trying to access a system is who they say they are.

A clear example of authentication is when you access a portal and you must enter some piece of information to prove your identity.

Authorization is the process of granting someone access to a resource such as a file or web page.


## References

[1]: [Understanding Flask vs FastAPI Web Framework](https://towardsdatascience.com/understanding-flask-vs-fastapi-web-framework-fe12bb58ee75)

[2]: [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)


[6 Concepts Every Backend Engineer Should Know](https://techwithmaddy.com/concepts-every-backend-engineer-should-know#heading-6-state-preservation)

[Speed Up Your Python Code With 100% Thread Utilization using FastAPI](https://betterprogramming.pub/speed-up-your-python-code-with-100-thread-utilization-using-this-library-31378a45f0ec)


[Deploying Your First Machine Learning API](https://www.kdnuggets.com/2021/10/deploying-first-machine-learning-api.html)

[How to Dockerize Machine Learning Applications Built with H2O, MLflow, FastAPI, and Streamlit](https://towardsdatascience.com/how-to-dockerize-machine-learning-applications-built-with-h2o-mlflow-fastapi-and-streamlit-a56221035eb5)


[10 Google Fonts Every Web Designer Needs To Know](https://uxplanet.org/10-google-fonts-every-web-designer-needs-to-know-de7dc3352d2c)

[3 Best Websites to find Pro and Free Templates for your Portfolio](https://medium.com/geekculture/3-best-websites-to-find-pro-free-templates-for-your-portfolio-c7745792e60)
