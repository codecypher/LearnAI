# Web Development

Here are some notes on web development.

## CSS Tips

Here are a few handy CSS tips and features [1].

### Centered

Position an element at the center of the screen.

```css
  .centered {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, 50%);
  }

  // position in the center of another element
  .block {
      display: grid;
      place-items: center;
  }
```

### Border vs Outline

The border is inside the element — if we increase the size of the border, then we increase the size of the element.

The outline is outside the element — if we increase the size of the outline, the element will keep its size and the ribbon around it will grow.

### Auto-numbering sections

We can create a CSS counter and use it in a tag type content, so we can auto-increment a variable and prefix some elements with it.

This is done using the counter-increment and content properties:


## Design Responsive Website

Here are some tips for designing responsive websites [2];

1. em and rem units instead of px

Always try to use em, percentage, rem units instead of px for sizing so that the size of text, images etc adjust according to the device

2. Proper use of Margin and Padding

We usually use a lot of padding and margin when we make websites for desktops , to make them more attractive. While making it responsive for mobiles, tablets try decreasing the existing padding and margin

3. Use Box-sizing property

It resolves a lot of problems caused by padding. Using box sizing on HTML elements with a percentage width will take padding into account rather than having to adjust the width because of padding

4. Use flex-box property to align content
Use flexbox to align your HTML elements, such as <div>, <img> etc.It forces elements that can wrap onto multiple lines according to their width

5. Use grid property to design layouts

Use grid property in CSS sheet to create layout of website . Rather than creating extra HTML elements to contain your grid, coloumns and rows, your grid tracks are created within your style sheet

6. Use media query for different screen sizes

Media query should be used to set width and height according to the breakpoints. Breakpoints refer to the width at which the websites look distorted on a particular size of device

7. Use CSS frameworks for Responsive websites

CSS frameworks are great way to build fast and responsive websites.A framework has ready to use code snippets for different purposes. They are very easy to use and embed in your website


----------



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

A comparison of two different RestAPI frameworks [4]:

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

State preservation is a mechanism to store data [8].

It comes in three variations:

- Cookies: they allow you to store small bits of information on the client, and it's sent to the server during an HTTP request.

- Session variables: a unique identifier is used to associate the information stored on the server with a particular client.

- Passing data at each request-response cycle: this variation allows you to store data on the web page.



## Authentication vs Authorization

Authentication is the process which ensures that someone trying to access a system is who they say they are.

A clear example of authentication is when you access a portal and you must enter some piece of information to prove your identity.

Authorization is the process of granting someone access to a resource such as a file or web page.




## References

[1]: [Some handy CSS tricks](https://medium.com/codex/some-handy-css-tricks-8e5a0d3ac25c)

[2]: [7 Tips to Design Responsive Website](https://medium.com/@monocosmo77/7-tips-to-design-responsive-website-6adf4f38a487)

[3]: [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

[4]: [Understanding Flask vs FastAPI Web Framework](https://towardsdatascience.com/understanding-flask-vs-fastapi-web-framework-fe12bb58ee75)

[5]: [Speed Up Your Python Code With 100% Thread Utilization using FastAPI](https://betterprogramming.pub/speed-up-your-python-code-with-100-thread-utilization-using-this-library-31378a45f0ec)


[6]: [Deploying Your First Machine Learning API](https://www.kdnuggets.com/2021/10/deploying-first-machine-learning-api.html)

[7]: [How to Dockerize Machine Learning Applications Built with H2O, MLflow, FastAPI, and Streamlit](https://towardsdatascience.com/how-to-dockerize-machine-learning-applications-built-with-h2o-mlflow-fastapi-and-streamlit-a56221035eb5)

[8]: [6 Concepts Every Backend Engineer Should Know](https://techwithmaddy.com/concepts-every-backend-engineer-should-know#heading-6-state-preservation)


[10 Google Fonts Every Web Designer Needs To Know](https://uxplanet.org/10-google-fonts-every-web-designer-needs-to-know-de7dc3352d2c)

[3 Best Websites to find Pro and Free Templates for your Portfolio](https://medium.com/geekculture/3-best-websites-to-find-pro-free-templates-for-your-portfolio-c7745792e60)
