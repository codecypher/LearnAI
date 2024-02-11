# Frequently Asked Questions

## What is an AI Engineer

[Artificial Intelligence Engineering](https://www.sei.cmu.edu/our-work/artificial-intelligence-engineering/)

In simplest terms, an AI Engineer is a type of software engineer specializing in development of AI/ML applications. Thus, he/she needs to have a thorough understanding of the core software engineering concepts (SWEBOK) as well as the full software development life cycle for AI/ML applications which has some  differences.

> In a nutshell, when you create a program to solve an AI problem you are performing AI engineering. 



## What is the difference between AI and ML

In AI, we define things in terms of _agents_ (agent programs) and the _standard model_ (rational agents). Thus, we are performing _machine learning_ when the agent is a _computer_ versus Robot, AV, or UAV. 

Answer: When the agent is a computer. 

If you did not know the answer, you need to reevaluate your approach to learning AI. You need to concentrate on learning the key concepts, theory, and algorithms but avoid study of SOTA algorithms since the algorithms are constantly changing. 


## Do I need a Master’s Degree

If you are going to spend the time to study AI/ML then you might as well invest in an online degree which will greatly increase your career opportunities (and a requirement for most all AI/ML engineer positions).

The best approach would be to find several job postings that look interesting to you and see what skills and tools they require.

[How to Learn Machine Learning](https://hashnode.codecypher.ai/how-to-learn-machine-learning-4ba736338a56)




## Recommended Tutorials and Books

[AI Learning Resources](https://hashnode.codecypher.ai/ai-learning-resources-b49da21fd3b8)

Open source projects can be a good resume builder such as PyCaret, scikit-learn, etc. GitHub repos usually have a “good first issue” or "help wanted" tags on the Issues tab. It can be a bit of work but u should be able to find some open-source project that u can contribute to.

It is also good to do some small projects yourself to create a portfolio that you can share on your own GitHub repos, even if it is just to fork and add some features/enhancements yourself to some small ML repo that you find interesting.


## How to access Safari Online

If you have an .edu email account you can get free access to [Safari Online][^safari_online] which has some good books for beginners as well as advanced books on various AI/ML topics.

[Creating an Account](https://ecpi.libguides.com/SafariOReilly)

Some good books are “Artificial Intelligence with Python”, “Artificial Intelligence by Example”, and “Programming Machine Learning”.


## What software do you recommend

It is best to use standalone .py files rather than notebooks. 

Some tools that I use regularly are mambaforge, Kedro, mlflow, git, dvc, scimitar-learn, tensorflow, NLTK, OpenCV, 

Some cloud tools that I use are comet.ml, DagsHub, and Deta. 

There are also some AutoML tools that I use a lot such as AutoGluon, DataPrep, and Orange. You can take a look at my Github Stars lists and LearnAI repo. 

https://github.com/codecypher



## Can I learn AI from research papers

Research papers are not a good resource for learning a topic. The reader is assumed to already know the core theory and concepts covered in textbooks. Thus, the best approach to learning AI is a good textbook. 

AI is considered a graduate level topic in computer science, so there a lot of Math and CS prerequisites that are needed first to properly learn the core theory and concepts. Otherwise, it will be problematic at some point when you try to actually use AI to solve a real-world problem. 

In general, the results of AI research articles are irreproducible. In fact, there is a major problem with integrity in AI research right now. The further away you get from reputable publications such as IEEE, the worse it gets.  I have seen numerous articles that I have no idea how they were ever published (major mistakes and errors). When learning AI, you need a healthy dose of skepticism in what you read, especially research articles. This is all discussed in the Russell and Norivg and other graduate textbooks. 

[Is My Model Really Better?](https://towardsdatascience.com/is-my-model-really-better-560e729f81d2)


## How to ask an AI/ML question

Briefly describe the following (1-2 sentences per item):

1. Give some description of your background and experience.

2. Describe the problem.

3. Describe the dataset in detail and be willing to share your dataset.

4. Describe any data preparation and feature engineering steps that you have done.

5. Describe the models that you have tried. 

6. Favor text and tables over plots and graphs.

7. Avoid asking users to help debug your code. 

[How to ask an AI/ML question](https://hashnode.codecypher.ai/how-to-ask-an-ai-ml-question-6cfddaa75bc9)

In AI and CS, you should always be able to describe in a few sentences what you are doing and why you are doing it. It is the first step in defining an AI problem. Also, there is a category and/or terminology for everything we do in CS. It always applies whether you are doing research or working on a simple problem. If you have not taken the time to think about the problem and put it into writing then u really don’t know what you are doing, do you?



## How to choose a performance metric

[Machine Learning Performance Metrics](./ml/performance_metrics.md) 

In general, all the data preparation, memory optimization, and hypertuning methods and techniques that you can possibly apply to the model and dataset will result in only a small performance improvement, say 3-5% improvement in accuracy. Therefore, it is always best to obtain a baseline for many models using the dataset with AutoML tools then choose the top performers for further study.  



## How to Choose an ML Algorithm

First, remember to take a data-centric approach, so avoid asking “what models should I use”. Thus, the first step in ML process would be to perform EDA to understand the properties of your model such as balanced (classification) or Gaussian (regression).

Concentrate on learning the key concepts such as data preparation, feature engineering, model selection, sklearn and tflow and pipelines, tflow Dataset class, etc. It also would be good to work through a few end-to-end classification/regression examples to get an idea for some of the steps involved.


There are some applied AI/ML processes and techniques given in Chapter 19 of the following textbooks that include a procedure for model selection:

S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Upper Saddle River, NJ: Prentice Hall, ISBN: 978-0-13-604259-4, 2021.

E. Alpaydin, Introduction to Machine Learning, 3rd ed., MIT Press, ISBN: 978-0262028189, 2014.

J Brownlee also describes an “Applied Machine Learning Process” that I have found most helpful in practice. 

I have some notes and checklists that I have created concerning the applied AI process in general. The process is  similar to the approach of SWE best practices given in SWEBOK. 


[Getting Started with AI](https://hashnode.codecypher.ai/getting-started-with-ai-13eafc77ac8e)

[LearnML](https://github.com/codecypher/LearnML)


[An End-to-End Machine Learning Project — Heart Failure Prediction](https://towardsdatascience.com/an-end-to-end-machine-learning-project-heart-failure-prediction-part-1-ccad0b3b468a?gi=498f31004bdf)

[A Beginner’s Guide to End to End Machine Learning](https://towardsdatascience.com/a-beginners-guide-to-end-to-end-machine-learning-a42949e15a47?gi=1736097101b9)

[End-to-End Machine Learning Workflow](https://medium.com/mlearning-ai/end-to-end-machine-learning-workflow-part-1-b5aa2e3d30e2)



## How to choose classification model

What do you do when you run a neural network that's barely optimizing (10-20%) over the baseline?

For more context, I am training a four layer neural network whose inputs are 2 x size 100 embeddings and the output is a boolean. The classes in each of my dataset are equally distributed (50% true, 50% false). As things stand, my model can detect falses with 75% accuracy and trues with just over 50%.

-----

Obviously, the model is not working since 50% accuracy is the lowest possible accuracy which is the same as random guessing (coin toss). 

First, plot the train/val loss per epoch of any models being trained to see what is happening (over/under-fitting) but I always dismiss any models with that low of a baseline accuracy (too much work). 

Next, try to find a few well-performing models with good baseline results for further study. Pretty sure NN will not be the best model (usually XGBoost and Random Forest).

For classification, I always evaluate the following models:

- Logistic Regression
- Naive Bayes
- AdaBoost
- kNN
- Random Forest
- Gradient Boosting
- SGD
- SVM
- Tree

I would start by using AutoML tools (Orange, AutoGluon, etc.) or write a test harness to get a baseline on many simpler models.  AutoML tools can perform much of the feature engineering that is needed for the different models (some models require different data prep than others) which is helpful. The tools can also quickly perform PCA and other techniques to help with feature engineering and dimensionality reduction. 

I would first get a baseline on 10-20 simpler models first before trying NN. Even then, I would use tools such as SpaCy to evaluate pretrained NN models. 

Only if all those models failed miserably would I then try to roll my own NN model. In general, you want to find a well-performing model that requires the least effort (hypertuning) - Occam's Razor. Most any model can be forced to fit a dataset using brute-force which is not the correct approach for AI.  

I have lots of notes on text classification/sentiment analysis and NLP data preparation in the "nlp” folder of my repo in nlp.md and nlp_dataprep.md. 

There are a lot of steps to text preprocessing in which mistakes can be made and it is often trial-and-error, so I prefer using AutoML to obtain baselines first before spending a lot of effort in coding. 

https://github.com/codecypher/LearnAI



## Should I start learning ML by coding an algorithm from scratch

[How to Learn Machine Learning](https://hashnode.codecypher.ai/how-to-learn-machine-learning-4ba736338a56)

I would recommend using Orange, AutoGluon, PyCaret and similar tools to evaluate many models (10-20) on your dataset. The tools will also help detect any issues with the data as well as perform the most common transforms needed for the various algorithms. Then, select the top 3 for further study. 

AutoML tools are the future of AI, so now would be a good time to see how they work rather than spend a lot of time coding the wrong algorithm from scratch which is a common beginner mistake. In short, you need to learn a data-centric approach.

The rule of thumb is that a deep learning model should be your last choice (Occam's Razor). 


## The obsession to understand how algorithms work

There seems to be an “obsession” with newcomers to understand how AI algorithms work. However, this is not the recommended approach to solving software engineering problems. 

Here are a few points to keep in mind:

1. There are just too many algorithms for any single person to understand and code from scratch. 

2. In real-world applications, there are usually thousands or millions of model parameters involved. 

3. The focus should be on how to use them (black-box) rather than how things work (white-box)  which is a basic axiom of OOA&D discussed in GoF Design Patterns. In a nutshell, black-box is the proper object-oriented approach to modeling complex systems.

4. Occam’s Razor (simpler is better than SOTA)

5. No Free Lunch Theorem (no such thing as best). 

6. With effort, almost any algorithm can be made to fit any dataset (brute-force) which is not the proper approach to AI. 

6. Any competent software engineer can implement any algorithm.




A common misconception when people begin learning ML is to implement algorithms from scratch. 

Yes that is common misconception when people begin learning ML. I would suggest you do some research into ML teaching methods.

It takes a lot of work to create a robust, reusable framework such as scikit-learn which should be your goal, not a simple novice implementation that is easy to understand. 

Most employers are going to want to see u contribute to open source projects such as scikit-learn rather than roll-your-own for exactly those reasons.

If u really insist on the code from scratch approach, there are lots of reputable sites such as machinelearningmastery.com with tutorials and sample code as well as the Keras Documentation which is pretty good for simple implementations such as you are doing. However, u will not have  accomplished much if u don’t have something to compare to once u r done.

The focus should be on _how to use_ robust ML frameworks to solve real-world problems.

> You do not have to _start_ by implementing machine learning algorithms. You will build your confidence and skill in machine learning a lot faster by learning how to use machine learning algorithms before implementing them.

In a nutshell, the scikit-learn source code is not easy to understand because of the complexity required to create a robust ML framework. The underlying algorithms are actually quite simple.

This is all discussed in the GoF Design Patterns book as well as modern AI and ML textbooks which I am confident u are not using as learning resources.

Note that J. Brownlee has a PhD in CS. Thus, this is not just my personal opinion but how learning ML is done. 

The problem with users posting this kind of material is that they inadvertently drag other newcomers down the same futile rabbit hole of code from scratch. 

[How to Learn AI](https://hashnode.codecypher.ai/how-to-learn-ai-7bb743f0bbdf)

[Stop Coding Machine Learning Algorithms From Scratch](https://machinelearningmastery.com/dont-implement-machine-learning-algorithms/)



## Is image channels first or last

A huge gotcha with both PyTorch and Keras. Actually, you need to sometimes need to to watch out when running code between OS (NHWC vs NCHW). I spent a long time tracking down an obscure error message between Linux and macOS that turned out to be the memory format.

[PyTorch Channels Last Memory Format Performance Optimization on CPU Path](https://gist.github.com/mingfeima/595f63e5dd2ac6f87fdb47df4ffe4772)

[Change Image format from NHWC to NCHW for Pytorch](https://stackoverflow.com/questions/51881481/change-image-format-from-nhwc-to-nchw-for-pytorch)

[A Gentle Introduction to Channels-First and Channels-Last Image Formats](https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/)


[Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer/)


## How to share your work

Your are always welcome to share your work in the following Discord AI forum channels of the SHARE KNOWLEDGE Section: #share-your-projects, #share-datasets, and #our-accomplishments. 

A key ML skill needed is deployment. There are several options available but the simplest method to try first would be streamlit or deta cloud services.



## How to choose a Cloud Platform

[Comparison of Basic Deep Learning Cloud Platforms](https://hashnode.codecypher.ai/comparison-of-basic-deep-learning-cloud-platforms-9a4b69f44a46)

[Colaboratory FAQ](https://research.google.com/colaboratory/faq.html#resource-limits)



## How to Choose a Dataset

When learning ML, it is best to look for the following properies in a dataset:

1. Categorical, Integer, or Real data types. 

2. Evenly distributed which means the same number of occurrences for each possible value of the class or target feature.

3. A large dataset, say thousands of samples for each possible value of the target feature.

4. Multivariate data types but not time series.

5. Try to keep the maximum number of features to 10 or so.

6. Try to avoid datasets for anomaly or fraud detection since they are almost always imbalanced and difficult to solve.



## Common Questions on Encoding

This section lists some common questions and answers when encoding categorical data.

### What if I have hundreds of categories

What if I concatenate many one-hot encoded vectors to create a many-thousand-element input vector?

You can use a one-hot encoding up to thousands and tens of thousands of categories. Having large vectors as input sounds intimidating but the models can usually handle it.

### What encoding technique is the best

This is impossible to answer. The best approach would be to test each technique on your dataset with your chosen model and discover what works best.

### What if I have a mixture of numeric and categorical data

What if I have a mixture of categorical and ordinal data?

You will need to prepare or encode each variable (column) in your dataset separately then concatenate all of the prepared variables back together into a single array for fitting or evaluating the model.

Alternately, you can use the `ColumnTransformer` to conditionally apply different data transforms to different input variables.


## Common Questions on Normalization

This section lists some common questions and answers when scaling numerical data.

### Which Scaling Technique is Best

This is impossible to answer. The best approach would be to evaluate models on data prepared with each transform and use the transform or combination of transforms that result in the best performance for your data set and model.

### Should I Normalize or Standardize

Whether input variables require scaling depends on the specifics of your problem and of each variable.

If the distribution of the values is normal, it should be standardized. Otherwise, the data should be normalized.

The data should be normalized whether the range of quantity values is large (10s, 100s, ...) or small (0.01, 0.0001, ...).

If the values are small (near 0-1) and the distribution is limited (standard deviation near 1) you might be able to get away with no scaling of the data.

Predictive modeling problems can be complex and it may not be clear how to best scale input data.

If in doubt, normalize the input sequence. If you have the resources, explore modeling with the raw data, standardized data, and normalized data and see if there is a difference in the performance of the resulting model.

### Should I Standardize then Normalize

Standardization can give values that are both positive and negative centered around zero.

It may be desirable to normalize data after it has been standardized.

Standardize then Normalize may be a good approach if you have a mixture of standardized and normalized variables and would like all input variables to have the same minimum and maximum values as input for a given algorithm such as an algorithm that calculates distance measures.

### How Do I Handle Out-of-Bounds Values

You may normalize your data by calculating the minimum and maximum on the training data.

Later, you may have new data with values smaller or larger than the minimum or maximum respectively.

One simple approach to handling this may be to check for out-of-bound values and change their values to the known minimum or maximum prior to scaling. Alternately, you can estimate the minimum and maximum values used in the normalization manually based on domain knowledge.



## Optimize Performance

Performance is always an issue with ML models upon deployment. 

Rather than trying to find the perfect tool, I would 1) analyze your system architecture for possible bottlenecks and 2) profile memory usage of both the ML model and the deployed system and 3) review common performance optimization techniques for ML models such as optimizing Python code, tensorflow code, mixed precision, other data formats such as parquet, concurrency and parallelism tools such as modin. Sorry, there really is no silver bullet. 

I have some notes on memory optimization and performance in the tips folder under LearnAI. The first step would be to identify the bottlenecks or performance issues. 

The simplest things to try would be modin and parquet which can greatly improve performance for common ML use cases. The other approaches or even trying another tool will require a lot more time and effort. However, 

https://publish.obsidian.md/serve?url=notes.cogentcoder.com/
https://medium.com/geekculture/what-is-real-time-streaming-data-processing-b726b1b271d1

https://towardsdatascience.com/machine-learning-streaming-with-kafka-debezium-and-bentoml-c5f3996afe8f

https://publish.obsidian.md/serve?url=notes.cogentcoder.com



## How does NLP work

The most common NLP problems are usually addressed by the NLP preprocessing steps: Tokenization, Stemming, and Lemmatization, plus Keyword extraction, and NER. 

In a nutshell, a well-performing NLP model will simply learn good sentences/phrases an ignores the bad. However, remember that no AI model will achieve 100% accuracy. 

In general, there are few common issues with NLP: Lexical Ambiguity, Syntax Level Ambiguity, and Referential Ambiguity. However, these problems are usually addressed by 1) training with more and larger corpora and 2) training embeddings on your own dataset. There are also other techniques as well.

The most common approach to NLP right now is to start with a pretrained model and then retrain/extend to your own custom dataset.



## TextRank

Can someone explain to me the damping factor `d` in the TextRank formula? `S = d*S.M + (1 - d)` where `M` is the adjacency matrix of nodes.

It js explained in Mihalcea & Tarau (2004) as "the probability of jumping from a given vertex to another random vertex in the graph" but it does noy quite sink it for me.


TextRank is a classic technique for keyword extraction and text summarization that is based on PageSearch except for text rather than pages. Recall that d is the damping constant and it comes from the PageRank algorithm. Basically, d is the probability (at any step) that an imaginary web surfer will continue clicking on page links. 

In general, you should be cautious about giving into the temptation to try to understand the internal details of ML algorithms. 1) there are simply too many algorithms for any person to fully understand and 2) any competent software engineer can implement any algorithm, so u really haven’t accomplished very much 3) it is paramount to learn how to _use_ the algorithms to solve common textbook problems and eventually be able to solve real-world problems, in that order. Exploring the implementation details should come much later, if at all. Anyway, that’s my expert advice. 


[PageRank](https://en.wikipedia.org/wiki/PageRank?wprov=sfti1)



## How to Develop a Chatbot

Chatbots are better to use pretrained model and software. You can take a look at Moodle and Rasa which are popular. There is also an example using NLTK that claims to be somewhat accurate. 

[How to Create a Moodle Chatbot Without Any Coding?](https://chatbotsjournal.com/how-to-create-a-moodle-chatbot-without-any-coding-3d08f95d94df)

[Building a Chatbot with Rasa](https://towardsdatascience.com/building-a-chatbot-with-rasa-3f03ecc5b324)

[Python Chatbot Project – Learn to build your first chatbot using NLTK and Keras](https://data-flair.training/blogs/python-chatbot-project/)



## Why are Robots not more common

Robot soccer is one type of classic robotics toy problem, often involving multiagent reinforcement learning (MARL). 

In robotics, there are a lot of technical issues (mainly safety related) involved besides the biomechanics of walking. 

There has been limited success for certain application areas such as the robot dog and of course robotic kits for arduino and raspberry pi (for hobbyists) but more practical applications still seem to be more elusive. 

In general, it costs a lot of money for R&D in robotics and it takes a long time for ROI. For example, openai recently disbanded its robotics division dues to lack of data and capital. Perhaps more interesting is the lack of a proper dataset which is needed for real-world robotics applications in the wild.


[OpenAI disbands its robotics research team](https://venturebeat.com/2021/07/16/openai-disbands-its-robotics-research-team/)
j
[Boston Dynamics now sells a robot dog to the public, starting at $74,500](https://arstechnica.com/gadgets/2020/06/boston-dynamics-robot-dog-can-be-yours-for-the-low-low-price-of-74500/)



[^safari_online]: https://www.oreilly.com/member/login/?next=%2Fapi%2Fv1%2Fauth%2Fopenid%2Fauthorize%2F%3Fclient_id%3D235442%26redirect_uri%3Dhttps%3A%2F%2Flearning.oreilly.com%2Fcomplete%2Funified%2F%26state%3DC3l2tLIMbQr0lpQKLDHucVJomOkg52rX%26response_type%3Dcode%26scope%3Dopenid%2Bprofile%2Bemail&locale=en "Safari Online"
