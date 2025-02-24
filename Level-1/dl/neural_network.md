# Neural Network

## Overview

Neural networks are a class of machine learning algorithms used to model complex patterns in datasets using multiple hidden layers and non-linear activation functions. 

A neural network takes an input, passes it through multiple layers of hidden neurons (mini-functions with unique coefficients that must be learned), and outputs a prediction representing the combined input of all the neurons.

Neural networks are trained iteratively using optimization techniques like gradient descent. 

After each cycle of training, an error metric is calculated based on the difference between prediction and target. 

The derivatives of this error metric are calculated and propagated back through the network using a technique called _backpropagation_.

Each neuron’s coefficients (weights) are then adjusted relative to how much they contributed to the total error. 

This process is repeated iteratively until the network error drops below an acceptable threshold.

## Weights

Weights are values that control the strength of the connection between two neurons which means inputs are typically multiplied by weights, and that defines how much influence the input will have on the output. 

Thus, when the inputs are transmitted between neurons, the weights are applied to the inputs along with an additional value (the bias)


## Bias

Bias terms are additional constants attached to neurons and added to the weighted input before the activation function is applied. 

Bias terms help models represent patterns that do not necessarily pass through the origin. 

For example, if all your features were 0, would your output also be zero? Is it possible there is some base value upon which your features have an effect? Bias terms typically accompany weights and must also be learned by your model.

## Weighted Input

A neuron’s input equals the sum of weighted outputs from all neurons in the previous layer. 

Each input is multiplied by the weight associated with the synapse connecting the input to the current neuron. 

If there are 3 inputs or neurons in the previous layer, each neuron in the current layer will have 3 distinct weights — one for each each synapse.


## Activation Functions

[How to Choose the Right Activation Function for Neural Networks](https://towardsdatascience.com/how-to-choose-the-right-activation-function-for-neural-networks-3941ff0e6f9c)

Activation functions are contained within neural network layers and modify the data they receive before passing it to the next layer. 

Activation functions give neural networks their power — allowing them to model complex non-linear relationships. 

By modifying inputs with non-linear functions neural networks can model highly complex relationships between features. Popular activation functions include relu and sigmoid.

Activation functions typically have the following properties:

- **Non-linear:** In linear regression we’re limited to a prediction equation that looks like a straight line. This is nice for simple datasets with a one-to-one relationship between inputs and outputs, but what if the patterns in our dataset were non-linear?

To model these relationships we need a non-linear prediction equation.¹ Activation functions provide this non-linearity.

- **Continuously differentiable:** To improve our model with gradient descent, we need our output to have a nice slope so we can compute error derivatives with respect to weights. If our neuron instead outputted 0 or 1 (perceptron), we wouldn’t know in which direction to update our weights to reduce our error.

- **Fixed Range:** Activation functions typically squash the input data into a narrow range that makes training the model more stable and efficient.


## Loss Functions

[How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

[Loss and Loss Functions for Training Deep Learning Neural Networks](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)

A loss or cost function is a wrapper around the model’s predict function that tells “how good” the model is at making predictions for a given set of parameters. 

The loss function has its own curve and its own derivatives. 

The slope of this curve tells us how to change our parameters to make the model more accurate! 

We use the model to make predictions. 

We use the cost function to update our parameters. 

Our cost function can take a variety of forms as there are many different cost functions available. 

Popular loss functions include: MSE (L2) and Cross-entropy Loss.

## Gradient Accumulation

Gradient accumulation is a mechanism to split the batch of samples used for training a neural network into several _mini-batches_ of samples that will be run sequentially.

Gradient accumulation is used to enable using large batch sizes that require more GPU memory than available by using mini-batches that require an amount of GPU memory that can be satisfied.

Gradient accumulation means running all mini-batches sequentially (generally on the same GPU) while accumulating their calculated gradients and not updating the model variables - the weights and biases of the model. 

The model variables must not be updated during the accumulation in order to ensure all mini-batches use the same model variable values to calculate their gradients. 

After accumulating the gradients of all those mini-batches, we generate and apply the updates for the model variables.

This results in the same updates for the model parameters as if we were to use the global batch.




# Complete Neural Network Example using Keras

This is a complete guide to multi-class classification problem using neural networks in which we will:

1. Transform the data
2. create the model
3. Evaluate with k-fold cross validation 
4. Compile and evaluate the model 
5. Save the model for later use. 
6. Load the model to make predictions without having to re-train



## References

[Machine Learning Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)

[The complete guide to Keras neural networks classification](https://shaun-enslin.medium.com/deep-dive-into-tensorflow-keras-with-a-real-life-neural-network-multi-class-problem-e50b1420432f)

[Neural Networks Basics: Activation Functions](https://medium.com/artificialis/neural-networks-basics-activation-functions-d75b67383da7)


