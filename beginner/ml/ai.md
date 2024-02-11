# AI Notes

## Terminology

> Occam’s Razor: The simplest model that fits the data is usually the best. 

- The simplest model that correctly fits the data is also the most consistent.

- Prefer the simplest hypothesis consistent with the data.


The term **bootstrap** means to randomly draw (with replacement) rows from the training dataset.


> Torture the data, and it will confess to anything - Ronald Coase


### No Free Lunch Theorem

**No Free Lunch Theorem:** Any two algorithms are equivalent when their performance is averaged across all possible problems.

There is no such thing as the _best_ learning algorithm. For any learning algorithm, there is a dataset where it is very accurate and another dataset where it is very poor. 

When we say that a learning algorithm is _good_, we only quantify how well its inductive bias matches the properties of the data.

Thus, a general-purpose universal optimization strategy is theoretically impossible and the only way one strategy can outperform another is if it is specialized to the specific problem under consideration. 


### The Curse of Dimensionality

The _curse of dimensionality_ refers to the problems that arise when working with data in the higher dimensions which does not exist in lower dimensions.

- As the number of features increase, the number of samples also increases proportionally. Thus, the more features we have, the more number of samples we will need to have all combinations of feature values well represented in our sample.

- As the number of features increase, the model becomes more complex and the greater the chances of overfitting. 

- A machine learning model that is trained on a large number of features becomes increasingly dependent on the data it was trained on and in turn overfitted which results in poor performance on new unseen data.

Avoiding overfitting is a major motivation for performing dimensionality reduction: the fewer features our training data has, the fewer assumptions our model makes and the simpler it will be. 



## When You Should not use ML

Always start with a feasibility study and know when not to use AI [7]. 

Technically, AI/ML is a graduate level topic which has several undergraduate prerequisites in math and CS which is probably why 80% or more of AI projects fail [4][5]. 


### Systematic Generalization

Current Machine Learning has issues with reliability due to the poor performance of OOD (Out-Of-Distribution) sample representation [12]. 

We are used to relying on the IID (Independent & Identically Distributed) hypothesis that the test distribution and the training distribution are the same. 

Without this assumption, we need some alternative hypothesis to perform the generalization.




## Disinformation in AI Research

Disinformation can also be found in scientific research publications, not just social media and online. Occurrences in research are more apparent when research or study data are irreproducible. In fact, a Harvard researcher resigned after a fraud discovery occurred.

> One study found that 33.7% of scientists surveyed admitted to questionable research practices at least once in their career [6]. 

Source data validation is necessary for research — especially funded research. The cost of source data validation is estimated to be between 20% and 30% of an overall clinical trial budget. 

What stops someone from simulating, tampering, or falsifying raw data to deliver a desired result to support a “desired” study hypothesis (this actually occured on a NIH research project that I was working on)? If data can be easily fabricated and falsified, is source data validation worth the costs? In addition, falsification may not be the only problem here; withholding data is also a problem. 

It is important to keep in mind that data _reproducibility_ can be a challenge because of improper research techniques such as when researchers look for data correlations until they find a bizarre outlier and then claim its statistical significance. Here, they could employ improper statistical techniques or change variables/combine data sets, invalidating the research/ study data and its results.  

If researchers were to maintain and use a data repository, it might create a learning community. In a learning community, outsiders not associated with the creation of the original data could request access to datasets to  test research outcomes and offer peer-reviewed improvements in a data owner’s experimental techniques. This could also discourage data tampering and falsification.

IEEE has attempted to help by providing a utility for researchers called [IEEE Dataport](https://ieee-dataport .org/) which offers researchers free data uploads and access of up to 2 TB. 

The IEEE Dataport is not only beneficial by having research data stored at a trusted organization but datasets may also be connected to IEEE journal and magazine articles which increases data and research visibility.

IEEE Dataport currently has almost 700,000 users and over 1,500 data sets.

Most importantly, this offering should support reproducible research, a topic that Computer will discuss further in future issues. 



----------



## Choose the right algorithm

How to choose the right machine learning algorithm [8]:

1. Categorize the problem

Categorize by input: 

- If it is a labeled data, it id a supervised learning problem. 

- If it is unlabeled data with the purpose of finding structure, it is an unsupervised learning problem. 

- If the solution implies to optimize an objective function by interacting with an environment, it is a reinforcement learning problem.

Categorize by output: 

- If the output of the model is a number, it is a regression problem. 

- If the output of the model is a class, it is a classification problem. 

- If the output of the model is a set of input groups, it is a clustering problem.


2. Understand the Data

The process of understanding the data plays a key role in the process of choosing the right algorithm for the right problem. 

Some algorithms can work with smaller sample sets while others require tons and tons of samples. 

Certain algorithms work with categorical data while others like to work with numerical input.

### Analyze the Data

There are two important tasks: 

- descriptive statistics
- visualization and plots

### Process the data

The components of data processing include pre-processing, profiling, and cleansing which often involve pulling together data from multiple sources.

### Transform the data

The idea of transforming data from a raw state to a state suitable for modeling is where feature engineering comes in. 

Feature engineering is the process of transforming raw data into features that better represent the underlying problem to improve model accuracy on unseen data.

3. Find the available algorithms

The next milestone is to identify the algorithms that are applicable that can be implemented in a reasonable time. 

Some of the elements affecting the choice of a model are [8]:

- The accuracy of the model.
- The interpretability of the model.
- The complexity of the model.
- The scalability of the model.

- How long does it take to build, train, and test the model?

- How long does it take to make predictions using the model?

- Does the model meet the business goal?

4. Implement the machine learning algorithms.

Create an ML pipeline that compares the performance of each algorithm on the dataset using a set of evaluation criteria. 

5. Optimize hyperparameters. 

There are three main options for optimizing hyperparameters: grid search, random search, and Bayesian optimization.


### Choosing the right estimator

Often the hardest part of solving a machine learning problem can be finding the right estimator for the job [11].

Different estimators are better suited for different types of data and different problems.

The [flowchart](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) is designed to give users a bit of a rough guide on how to approach problems with regard to which estimators to try on your data [11].



------


## Artificial Intelligence

Artificial Intelligence = A technique which enables machines to mimic human behavior.

Machine Learning = A subset of AI which uses statistical methods to enable machines to improve with experience.

Deep Learning = A subset of ML which makes the computation of multi-layer neural networks feasible.


### How to Choose the Right AI Algorithm

- [Artificial Intelligence Algorithms: All you need to know](https://www.edureka.co/blog/artificial-intelligence-algorithms)

- [A Comprehensive Guide To Artificial Intelligence With Python](https://www.edureka.co/blog/artificial-intelligence-with-python/)

- [AI vs Machine Learning vs Deep Learning](https://www.edureka.co/blog/ai-vs-machine-learning-vs-deep-learning/)


------


### Artificial Intelligence Tutorial

- What is Artificial Intelligence?
- Importance of Artificial Intelligence
- Artificial Intelligence Applications
- Domains of Artificial Intelligence
- Different Job Profiles in AI
- Companies Hiring

[Artificial Intelligence Tutorial : All you need to know about AI](https://www.edureka.co/blog/artificial-intelligence-tutorial/)


------

K-Nearest Neighbors Algorithm in Python and Scikit-Learn (StackAbuse)

k-nearest neighbor algorithm in Python (GeeksforGeeks)

Building A Book Recommender System (DataScience+)


------


## Deep Learning

An _artificial neural network (ANN)_ is a computing system inspired by the biological neural networks that constitute animal brains. Such systems learn (progressively improve their ability) to do tasks by considering examples (generally without task-specific programming).

For example, in image recognition they might learn to identify images that contain cats by analyzing example images that have been manually labeled as "cat" or "no cat" and using the analytic results to identify cats in other images. They have found most use in applications difficult to express with a traditional computer algorithm using rule-based programming.

A _deep neural network (DNN)_ is an ANN with multiple layers between the input and output layers. The DNN finds the correct mathematical manipulation to turn the input into the output, whether it be a linear relationship or a non-linear relationship. The network moves through the layers calculating the probability of each output.

For example, a DNN that is trained to recognize dog breeds will go over the given image and calculate the probability that the dog in the image is a certain breed. The user can review the results and select which probabilities the network should display (above a certain threshold, etc.) and return the proposed label. Each mathematical manipulation as such is considered a layer, and complex DNN have many layers, hence the name "deep" networks.

In deep learning, a _convolutional neural network (CNN)_ is a class of deep neural networks that is usually used to analyze visual images. They are also known as shift invariant or space invariant artificial neural networks (SIANN) based on their shared-weights architecture and translation invariance characteristics. They have applications in image and video recognition, recommender systems, image classification, medical image analysis, natural language processing, and financial time series.

CNNs are regularized versions of _multilayer perceptrons_ which usually means fully connected networks -- each neuron in one layer is connected to all neurons in the next layer. However, the "fully-connectedness" of these networks makes them prone to overfitting data. Typical ways of regularization include adding some form of magnitude measurement of weights to the loss function. CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns. Therefore, on the scale of connectedness and complexity, CNNs are on the lower extreme.

### Challenges

As with ANNs, many issues can arise with naively trained DNNs. Two common issues are: overfitting and computation time.

DNNs are prone to overfitting because of the added layers of abstraction, which allow them to model rare dependencies in the training data. Regularization methods such as Ivakhnenko's unit pruning or weight decay or sparsity can be applied during training to combat overfitting. Alternatively dropout regularization randomly omits units from the hidden layers during training. This helps to exclude rare dependencies. Finally, data can be augmented via methods such as cropping and rotating such that smaller training sets can be increased in size to reduce the chances of overfitting.

DNNs must consider many training parameters, such as the size (number of layers and number of units per layer), the learning rate, and initial weights. Sweeping through the parameter space for optimal parameters may not be feasible due to the cost in time and computational resources. Various tricks, such as batching (computing the gradient on several training examples at once rather than individual examples) speed up computation. Large processing capabilities of many-core architectures (such as GPUs or the Intel Xeon Phi) have produced significant speedups in training, because of the suitability of such processing architectures for the matrix and vector computations

### Applications

- Automatic speech recognition
- Image recognition
- Visual art processing
- Natural language processing
- Drug discovery and toxicology
- Customer relationship management
- Recommendation systems
- Bioinformatics
- Medical Image Analysis
- Mobile advertising
- Image restoration
- Financial fraud detection
- Military


------


## Machine Learning for Humans

**Artificial intelligence** is the study of agents that perceive the world around them, form plans, and make decisions to achieve their goals [9].

Many fields fall under the umbrella of AI such as: computer vision, robotics, machine learning, and natural language processing.

Machine learning is a subfield of artificial intelligence. The goal of ML is to enable computers to learn on their own. 

An ML algorithm enables a machine to identify patterns in observed data, build models that explain the world, and predict things without having explicit pre-programmed rules and models.

The technologies discussed above are examples of **artificial narrow intelligence (ANI)** which can effectively perform a narrowly defined task.

We are also continuing to make foundational advances towards human-level **artificial general intelligence (AGI)** or _strong AI_. 

The definition of AGI is an artificial intelligence that can successfully perform _any intellectual task_ that a human being can: learning, planning and decision-making under uncertainty, communicating in natural language, making jokes, manipulating people, trading stocks, or reprogramming itself.


------


## Neural Networks and Deep Learning

Some extensions and further concepts worth noting [10]:

- Deep learning software packages. You will rarely need to implement all the parts of neural networks from scratch because of existing libraries and tools that make deep learning implementations easier. There are many of these: TensorFlow, Caffe, Torch, Theano, and more.

- **Convolutional neural networks (CNNs)** are designed specifically for taking images as input and are effective for computer vision tasks. They are also instrumental in deep reinforcement learning. CNNs are inspired by the way animal visual cortices work and they are the focus of the deep learning course we have been referencing throughout this article (Stanford’s CS231n).

- **Recurrent neural networks (RNNs)** have a kind of built-in memory and are well-suited for natural language problems. They are also important in reinforcement learning since they enable the agent to keep track of where things are and what happened historically even when those elements are not all visible at once. In fact, both RNNs and LSTMs are often used in the context of natural language problems.

- **Deep reinforcement learning** is one of the most exciting areas of deep learning research, at the heart of recent achievements in AI. We will dive deeper in Part 5, but essentially the goal is to apply all of the techniques in this post to the problem of teaching an agent to maximize reward. This can be applied in any context that can be gamified — from actual games like Counter Strike or Pacman, to self-driving cars, to trading stocks, to real life and the real world.


## Deep learning applications

Here are a few examples of the incredible things that deep learning can do:

- Facebook trained a neural network augmented by short-term memory to intelligently answer questions about the plot of "Lord of the Rings".

- Self-driving cars rely on deep learning for visual tasks like understanding road signs, detecting lanes, and recognizing obstacles.

- Predicting molecule bioactivity for drug discovery

- Face and object recognition for photo and video tagging

- Powering Google search results

- Natural language understanding and generation such as Google Translate

- The Mars explorer robot Curiosity is autonomously selecting inspection-worthy soil targets based on visual examination


------


## Gradient descent

Gradient descent is used in ML and DL to optimize the models in an iterative process by taking the gradient (derivative) of the _loss_ function. 

Gradient descent takes more computational resources to optimize, so SGD can be used which requires less computation power because it takes only one data point to optimize, but SGD requires a lot of iterations to reach the destination. 

Mini-batch gradient descent is the perfect balance between the above-spoken methods which takes fewer iterations than SGD to converge and less computation power than Gradient descent.


## What is online learning?

We define online learning as a training scenario in which the full dataset is never available to the model at the same time. 

The model is exposed to the portions of the dataset sequentially and expected to learn the full training task through such partial exposures. 

After being exposed to a certain portion of the dataset, the model is not allowed to re-visit this data later on. Otherwise, the model could simply loop over the dataset and perform a normal training procedure.


## References

[1] E. Alpaydin, Introduction to Machine Learning, 3rd ed., MIT Press, ISBN: 978-0262028189, 2014.

[2] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 3rd ed. Upper Saddle River, NJ, USA: Prentice Hall, 2010.

[3] aima-python: Python code for Artificial Intelligence: A Modern Approach. Accessed: June 14, 2020. [Online]. Available: https://github.com/aimacode/aima-python, 0.17.3. 2020.

[4] Y. Kosarenko, [The majority of business analytics and AI projects are still failing](https://medium.com/r/?url=https%3A%2F%2Fwww.datadriveninvestor.com%2F2020%2F04%2F30%2Fthe-majority-of-business-analytics-and-ai-projects-are-still-failing%2F), Data Driven Investor, April 30, 2020.

[5] A. DeNisco Rayome, [Why 85% of AI projects fail](https://medium.com/r/?url=https%3A%2F%2Fwww.techrepublic.com%2Farticle%2Fwhy-85-of-ai-projects-fail%2F), TechRepublic, June 20, 2019.

[6] J. F. DeFranco and J. Voas, "Reproducibility, Fabrication, and Falsification", IEEE Computer, vol. 54 no. 12, 2021.

[7] T. Shin, [4 Reasons Why You Shouldn't Use Machine Learning](https://towardsdatascience.com/4-reasons-why-you-shouldnt-use-machine-learning-639d1d99fe11), Towards Data Science, Oct 
5, 2021.


[8] [Do you know how to choose the right machine learning algorithm among 7 different types?](https://towardsdatascience.com/do-you-know-how-to-choose-the-right-machine-learning-algorithm-among-7-different-types-295d0b0c7f60

[9] [Machine Learning for Humans](https://medium.com/machine-learning-for-humans/why-machine-learning-matters-6164faf1df12)

[10] [Neural Networks and Deep Learning](https://medium.com/machine-learning-for-humans/neural-networks-deep-learning-cdad8aeae49b)

[11] [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

[12] [The Gap Between Deep Learning and Human Cognitive Abilities](https://www.kdnuggets.com/2022/10/gap-deep-learning-human-cognitive-abilities.html)



----------


[OpenAI Is Disbanding Its Robotics Research Team](https://lastfuturist.com/openai-is-disbanding-its-robotics-research-team/)

[Microsoft invests $1 billion in OpenAI to pursue holy grail of artificial intelligence](https://www.theverge.com/2019/7/22/20703578/microsoft-openai-investment-partnership-1-billion -azure-artificial-general-intelligence-agi)

