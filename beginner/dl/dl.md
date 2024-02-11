# Deep Learning

## Challenge of Training Deep Learning Neural Networks

[A Gentle Introduction to the Challenge of Training Deep Learning Neural Network Models](https://machinelearningmastery.com/a-gentle-introduction-to-the-challenge-of-training-deep-learning-neural-network-models/)

### Neural Networks Learn a Mapping Function

Deep learning neural networks learn a _mapping function_. 

Developing a model requires historical data from the domain that is used as training data which is comprised of observations or examples from the domain with input elements that describe the conditions and an output element that captures what the observation means.

- A problem where the output is a quantity would be described generally as a regression predictive modeling problem. 

- A problem where the output is a label would be described generally as a classification predictive modeling problem.

A neural network model uses the examples to learn how to map specific sets of input variables to the output variable. 

The NN must learn the mapping in such a way that this mapping works well for the training dataset but also works well on new examples not seen by the model during training which is called _generalization_. 


We can describe the relationship between the input variables and the output variables as a complex mathematical function. 

For a given model problem, we must believe that a true mapping function exists to best map input variables to output variables and that a neural network model can do a reasonable job at approximating the true unknown underlying mapping function.

Thus, we can describe the broader problem that neural networks solve as _function approximation_. 

A NN learns to approximate an unknown underlying mapping function given a training dataset by learning weights and the model parameters, given a specific network structure that we design.

However, finding the parameters for neural networks in general is hard.

In fact, training a neural network is the most challenging part of using the technique.

The use of nonlinear activation functions in the neural network means that the optimization problem that we must solve in order to find model parameters is not convex.

Solving this optimization is challenging since the error surface often contains many local optima, flat spots, and cliffs.


### Navigating the Non-Convex Error Surface

A NN model has a specific set of weights that can be evaluated on the training dataset and the average error over all training datasets can be thought of as the error of the model. 

A change to the model weights will result in a change to the model error. Therefore, we seek a set of weights that result in a model with a small error.

The process involves repeating the steps of evaluating the model and updating the model parameters in order to step down the error surface which is repeated until a set of parameters is found that is good enough or the search process gets stuck.

Thus, the process is a search or an optimization and we refer to optimization algorithms that operate in this way as gradient optimization algorithms sonce they naively follow along the error gradient. In practice, this is more art than science.

The algorithm that is most commonly used to navigate the error surface is called stochastic gradient descent (SGD).

Other global optimization algorithms designed for non-convex optimization problems could be used such as a genetic algorithm but stochastic gradient descent is more efficient since it uses the gradient information specifically to update the model weights via an algorithm called _backpropagation_.

Backpropagation refers to a technique from calculus to calculate the derivative (auch as the slope or the gradient) of the model error for specific model parameters which allows the model weights to be updated to move down the gradient.


### Components of the Learning Algorithm

Training a deep learning neural network model using stochastic gradient descent with backpropagation involves choosing a number of components and hyperparameters:

- Loss Function: The function used to estimate the performance of a model with a specific set of weights on examples from the training dataset.

- Weight Initialization: The procedure by which the initial small random values are assigned to model weights at the beginning of the training process.

- Batch Size: The number of examples used to estimate the error gradient before updating the model parameters.

- Learning Rate: The amount that each model parameter is updated per cycle of the learning algorithm.

- Epochs. The number of complete passes through the training dataset before the training process is terminated.


### Decrease Neural Network Size and Maintain Accuracy

[Decrease Neural Network Size and Maintain Accuracy](https://towardsdatascience.com/decrease-neural-network-size-and-maintain-accuracy-knowledge-distillation-6efb43952f9d)

Some neural networks are too big to use. There is a way to make them smaller but keep their accuracy.

1. Pruning
2. Knowledge distillation


### Number of hidden layers and nodes

The number of hidden layers depends on the complexity of the task: very complex tasks (such as large image classification or speech recognition) usually require networks with dozens of layers and a huge amount of training data. 

For the majority of the problems, we can start with just one or two hidden layers and gradually ramp up the number of hidden layers until we start overfitting the training set.

The number of hidden nodes should have a relationship to the number of input and output nodes, the amount of training data available, and the complexity of the function being modeled. As a rule of thumb, the number of hidden nodes in each layer should be somewhere between the size of the input layer and the size of the output layer, ideally the mean. 

The number of hidden nodes should not exceed twice the number of input nodes in order to avoid overfitting.


### Batch size

The batch size is the number of training examples in one forward/backward pass.

[Why Small Batch sizes lead to greater generalization in Deep Learning](https://medium.com/geekculture/why-small-batch-sizes-lead-to-greater-generalization-in-deep-learning-a00a32251a4f)

A batch size of 32 means that 32 samples from the training dataset will be used to estimate the error gradient before the model weights are updated. 

The higher the batch size, the more memory space that is needed.

There are some hyperparameterss that often have optimal values in base 2 such as `batch_size` mainly because it affects the data size that is fetched to/from memory by hardware that is base 2.


----------


## Encoder-Decoder

The Encoder-Decoder architecture is a way of organizing recurrent neural networks for sequence prediction problems that have a variable number of inputs, outputs, or both inputs and outputs.

The architecture involves two components: an encoder and a decoder.

- **Encoder:** The encoder reads the entire input sequence and encodes it into an internal representation, often a fixed-length vector called the context vector.

- **Decoder:** The decoder reads the encoded input sequence from the encoder and generates the output sequence.


## Attention

Attention is the idea of freeing the encoder-decoder architecture from the fixed-length internal representation.


## Transformer

The Transformer architecture follows an encoder-decoder structure but does not rely on recurrence and convolutions in order to generate an output. 

In a nutshell, the task of the encoder is to map an input sequence to a sequence of continuous representations which is fed into a decoder. 

The decoder receives the output of the encoder together with the decoder output at the previous time step, to generate an output sequence.

### The Transformer Attention Mechanism

Before the introduction of the Transformer model, the use of attention for neural machine translation was being implemented using RNN-based encoder-decoder architectures. 

The Transformer model revolutionized the implementation of attention by dispensing of recurrence and convolutions and relies only on a self-attention mechanism. 



## Transfer Learning

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

Transfer learning is usually done for tasks where your dataset has too little data to train a full-scale model from scratch.

A popular approach in deep learning is to use  pre-trained models as the starting point for computer vision and natural language processing tasks due to the vast compute and time resources required to develop neural network models on these problems.

Transfer learning is related to problems such as multi-task learning and concept drift and is not exclusively an area of study for deep learning.

This form of transfer learning used in deep learning is called inductive transfer. This is where the scope of possible models (model bias) is narrowed in a beneficial way by using a model fit on a different but related task.

### Transfer learning types

In transfer learning, there are three kinds of methods that can be used (depending on the problem statement):

- **Fixed feature extractor:** the pre-trained model is used as a feature extractor in which the weights in the feature extraction layer are frozen while the fully connected layer is removed

- **Fine-tuning:** it uses the architecture of a pre-trained model and initializes the weights in the feature extraction layer. It means the weights in the feature extraction layer are not frozen

- **Hybrid:** the combination between fixed feature extractor and fine-tuning — some layers are frozen while the others are trained (their weights are initialized)

How to choose the method to be used?


![Transfer learning quadrant|600xauto {Figure 1: Transfer learning quadrant}](https://miro.medium.com/max/1722/1*ZSx3ZsBxs3kE87ybpY4NnA.png)


- Quadrant 1: large data size but has a small data similarity. In this case, better if we develop the model from scratch

- Quadrant 2: large data size but has a high data similarity. In this case, we should consider training some layers in the feature extraction layer while the others are frozen. The number of layers is debatable, it depends on the needs

- Quadrant 3: small data size but has a small data similarity. Similar to the quadrant 2 scenario

- Quadrant 4: small data size but has a high data similarity. In this case, we can implement the fixed feature extractor method for transfer learning


## References

H. Tatsat, S. Puri, and B. Lookabaugh, "Machine Learning and Data Science Blueprints for Finance", O’Reilly Media, ISBN: 978-1-492-07305-5, 2021. 


[Better Deep Learning Performance](https://machinelearningmastery.com/start-here/#better)

[How To Improve Deep Learning Performance](https://machinelearningmastery.com/improve-deep-learning-performance/)

[How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

[Importance of Loss functions in Deep Learning](https://towardsdatascience.com/importance-of-loss-functions-in-deep-learning-and-python-implementation-4307bfa92810?gi=6295a1b1892)

[Understand the Impact of Learning Rate on Neural Network Performance](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)

[Estimating required sample size for model training](https://keras.io/examples/keras_recipes/sample_size_estimate/)


[Encoder-Decoder Models for Text Summarization in Keras](https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/)

[The Transformer Model](https://machinelearningmastery.com/the-transformer-model/)

[The Transformer Attention Mechanism](https://machinelearningmastery.com/the-transformer-attention-mechanism/)

[A Gentle Introduction to Transfer Learning for Deep Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)

