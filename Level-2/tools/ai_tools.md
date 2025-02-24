# AI/ML Tools

Here is a list of tools that I have found to be helpful for AI engineering.

For items without links, see **Github Stars**. 

[GitHub Stars](https://github.com/codecypher?tab=stars)

[Github Student Developer Pack](https://education.github.com/pack)

[HuggingFace Spaces](https://huggingface.co/spaces/launch)


## How to choose an ML framework

[Keras vs PyTorch for Deep Learning](https://towardsdatascience.com/keras-vs-pytorch-for-deep-learning-a013cb63870d)


## Data Exploration Tools

- Orange
- DataPrep
- Bamboolib
- TensorFlow Data Validation
- Great Expectations

**NOTE:** It is best to install the Orange native executable on your local machine rather than install using anaconda and/or pip.

### Tutorials

[Orange Docs](https://orangedatamining.com/docs/)

[A Great Python Library: Great Expectations](https://towardsdatascience.com/a-great-python-library-great-expectations-6ac6d6fe822e)


## Feature Engineering Tools

There are many tools that will help you in automating the entire feature engineering process and producing a large pool of features in a short period of time for both classification and regression tasks.

- Feature-engine
- Featuretools
- AutoFeat

### Tutorials

[The Only Web Scraping Tool you need for Data Science](https://medium.com/nerd-for-tech/the-only-web-scraping-tool-you-need-for-data-science-f388e2afa187)


## Machine Learning Tools

- Kedro
- Comet
- DagsHub

- OpenML

- ONNX
- openai/gym
- PyMC (Bayesian statistical modeling)
- Snap ML


### Poetry

[Poetry Docs](https://python-poetry.org/docs/)

Poetry: Dependency Management for Python
 
Poetry helps you declare, manage, and install dependencies of Python projects, ensuring you have the right stack everywhere.

poetry is a tool to handle dependency installation as well as building and packaging of Python packages. It only needs one file to do all of that: the new, standardized pyproject.toml.

This means poetry uses `pyproject.toml` to replace setup.py, requirements.txt, setup.cfg, MANIFEST.in and Pipfile.

### PySpark

[Getting Started](https://spark.apache.org/docs/latest/api/python/getting_started/index.html)

PySpark is an interface for Apache Spark in Python. It not only allows you to write Spark applications using Python APIs, but also provides the PySpark shell for interactively analyzing your data in a distributed environment. PySpark supports most of Spark’s features such as Spark SQL, DataFrame, Streaming, MLlib (Machine Learning) and Spark Core.

### Snap ML

[Snap ML](https://www.zurich.ibm.com/snapml/)

Snap ML is a library that provides high-speed training of popular machine learning models on modern CPU/GPU computing systems

[This Library is 30 Times Faster Than Scikit-Learn](https://medium.com/@irfanalghani11/this-library-is-30-times-faster-than-scikit-learn-206d1818d76f)

[IBM Snap ML Examples](https://github.com/IBM/snapml-examples)

### Tutorials

[A Gentle Introduction to Bayesian Belief Networks](https://machinelearningmastery.com/introduction-to-bayesian-belief-networks/)

[Building DAGs with Python](https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/)

[bnlearn](https://github.com/erdogant/bnlearn)


## Deep Learning Tools

- MXNet

### Hydra

[Hydra](https://hydra.cc/docs/intro/) is an open-source Python framework that simplifies the development of research and other complex applications. 

The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. 

The name Hydra comes from its ability to run multiple similar jobs - similar to a Hydra with multiple heads.

Hydra provides a configuration file for the entire experiment. We can have different parameters to be set. It can be very helpful when we want to share our code with someone else or run the experiments on a different machine. 

Hydra provides the flexibility to set the desired configurations such as learning rate, model hidden layer sizes, epochs, data set name, etc. without exposing someone to make changes to the actual code.

### H5py

[H5py](https://docs.h5py.org/en/stable/quick.html) can be used to store all the intermediate loss values in a dictionary mapped to appropriate key which can be loaded to be reused as a python code.

### Loguru

[Loguru](https://loguru.readthedocs.io/en/stable/api/logger.html) provides the functionality of a logger to log configuration, experiment name, and other training-related data which is helpful when we do multiple experiments and want to distinguish the results of different experiments. Thus, if we log the actual configuration as well as the results, then it is easier to map the appropriate setting to the outputs.

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


### Tutorials

[Are You Still Using Virtualenv for Managing Dependencies in Python Projects?](https://towardsdatascience.com/poetry-to-complement-virtualenv-44088cc78fd1)

[3 Tools to Track and Visualize the Execution of Your Python Code](https://www.kdnuggets.com/2021/12/3-tools-track-visualize-execution-python-code.html)


## Pretrained Models

- [Model Zoo](https://modelzoo.co/)
- [TensorFlow Hub](https://tfhub.dev/)
- [TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official)
- [Hugging Face](https://github.com/huggingface)
- [PyTorch Hub](https://pytorch.org/hub/)
- [Papers with Code](https://paperswithcode.com/)


## Python Tools

- dateutil
- tqdm
- urllib3

- pipreqs
- pip-review
- Poetry
- pyenv


- The Algorithms - Python
- vinta/awesome-python
- josephmisiti/awesome-machine-learning


### Jupyterlab

[JupyterLab](https://github.com/jupyterlab/jupyterlab) is the next-generation user interface for Project Jupyter offering all the familiar building blocks of the classic Jupyter Notebook (notebook, terminal, text editor, file browser, rich outputs, etc.) in a flexible and powerful user interface. JupyterLab will eventually replace the classic Jupyter Notebook.

Jupyterlab has an updated UI/UX with a tab interface for working with multiple files and notebooks.

Since Jupyter is really a web server application, it runs much better on a remote server. 

I currently have Jupyterlab installed and running as a Docker container on a VM droplet which runs much better than on my local machine. The only issue is that my VM only has 4GB memory. However, I have had great success so far using Jupyterlab and Modin with notebooks that I am unable to run on my local machine with 32GB memory (out of memory issues) without any performance issues.

If you do not have cloud server of your own, a nice alternative is [Deepnote](https://deepnote.com). The free tier does not offer GPU access but it does offer a shared VM with 24GB of memory running a custom version of Jupyterlab which I have found more useful than Google Colab Pro. It is definitely worth a try. 

### Modin

[Modin](https://github.com/modin-project/modin) is a drop-in replacement for pandas. 

While pandas is single-threaded, Modin lets you speed up your workflows by scaling pandas so it uses all of your cores. 

Modin works especially well on larger datasets where pandas becomes painfully slow or runs out of memory.

Using modin is as simple as replacing the pandas import:

```py
  # import pandas as pd
  import modin.pandas as pd
```

I have a sample [Notebook](../python/book_recommender_knn.ipynb) that demonstrates using modin. 

Since Modin is still under development, I do experience occasional warning/error messages but everything seems to be working. However, the developers seem to be quick to answer questions and provide assistance in troubleshooting issues. Highly recommend trying it out. 


### Pickle

Pickle can be used to save and load the python classes or PyTorch models for reuse.

### Debugging Tools

- heartrate
- Loguru
- snoop


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


## GenAI

- 1min.ai
- console.groq.com
- app.aimagicx.com
- you.com
- insmind.com (photos and images)



## References

[1]: [All Top Python Libraries for Data Science Explained](https://towardsdatascience.com/all-top-python-libraries-for-data-science-explained-with-code-40f64b363663)

[2]: [26 GitHub Repositories To Inspire Your Next Data Science Project](https://towardsdatascience.com/26-github-repositories-to-inspire-your-next-data-science-project-3023c24f4c3c)

[3]: [4 Amazing Python Libraries That You Should Try Right Now](https://towardsdatascience.com/4-amazing-python-libraries-that-you-should-try-right-now-872df6f1c93)

[4]: [Tools for Efficient Deep Learning](https://towardsdatascience.com/tools-for-efficient-deep-learning-c9585122ded0)

