# Stochastic Gradient Descent Optimization

## What is Gradient Descent?

Gradient descent (GD) is an optimization algorithm often used for finding the _weights_ or coefficients of machine learning algorithms such as artificial neural networks and logistic regression.

GD works by having the model make predictions on training data and using the error on the predictions to update the model in such a way that it reduces the error.

The goal of the algorithm is to find model parameters (coefficients or weights) that minimize the error of the model on the training dataset. 

GD does this by making changes to the model that move it along a gradient or slope of errors down toward a minimum error value which gives the algorithm its name of “gradient descent.”

## Contrasting the 3 Types of Gradient Descent

Gradient descent can vary in terms of the number of training patterns used to calculate error which is in turn used to update the model.

The number of patterns used to calculate the error includes how stable the gradient is that is used to update the model. 

We will see that there is a tension in gradient descent configurations of computational efficiency and the fidelity of the error gradient.

The three main flavors of gradient descent are: batch, stochastic, and mini-batch.


### Stochastic Gradient Descent

Stochastic gradient descent (SGD) is a variation of the gradient descent algorithm that calculates the error and updates the model for each example in the training dataset.

The update of the model for each training example means that stochastic gradient descent is often called an **online** machine learning algorithm.

Cons:

- Updating the model so frequently is more computationally expensive than other configurations of gradient descent, taking significantly longer to train models on large datasets.

- The frequent updates can result in a noisy gradient signal, which may cause the model parameters and in turn the model error to jump around (have a higher variance over training epochs).

- The noisy learning process down the error gradient can also make it hard for the algorithm to settle on an error minimum for the model.

### Batch Gradient Descent

Batch gradient descent is a variation of the gradient descent algorithm that calculates the error for each example in the training dataset, but only updates the model after all training examples have been evaluated.

One cycle through the entire training dataset is called a training epoch. 

Therefore, it is often said that batch gradient descent performs model updates at the end of each training epoch.

### Mini-Batch Gradient Descent

Mini-batch gradient descent is a variation of the gradient descent algorithm that splits the training dataset into small batches that are used to calculate model error and update model coefficients (weights).

Implementations may choose to sum the gradient over the mini-batch which further reduces the variance of the gradient.

Mini-batch gradient descent seeks to find a balance between the robustness of stochastic gradient descent and the efficiency of batch gradient descent. 

It is the most common implementation of gradient descent used in the field of deep learning.

Pros:

- The model update frequency is higher than batch gradient descent which allows for a more robust convergence, avoiding local minima.
- The batched updates provide a computationally more efficient process than stochastic gradient descent.
- The batching allows both the efficiency of not having all training data in memory and algorithm implementations.

Cons:

- Mini-batch requires the configuration of an additional “mini-batch size” hyperparameter for the learning algorithm.
- Error information must be accumulated across mini-batches of training examples like batch gradient descent.


----------



# Gradient Descent Optimization Algorithms

http://ruder.io/optimizing-gradient-descent/
http://cs231n.github.io/neural-networks-3/

## Gradient Checks

In theory, performing a gradient check is as simple as comparing the analytic gradient to the numerical gradient. In practice, the process is much more involved and error prone. 

Here are some tips, tricks, and issues to watch out for:

**Use the centered formula.** 

The formula you may have seen for the finite difference approximation when evaluating the numerical gradient looks as follows:

    df/dx = (f(x+h) − f(x)) / h     (bad)

where h

is a very small number, in practice approximately 1e-5 or so. 

In practice, it turns out that it is much better to use the _centered_ difference formula of the form:

    df/dx = (f(x+h) − f(x−h)) / 2h  (good)

which requires you to evaluate the loss function twice to check every single dimension of the gradient (so it is about twice as expensive), but the gradient approximation turns out to be much more precise. 

To see this, you can use Taylor expansion of f(x+h) and f(x−h) and verify that the first formula has an error on order of O(h), while the second formula only has error terms on order of O(h2) (it is a second order approximation).

## Parameter Updates

SGD has trouble navigating areas where the surface curves much more steeply in one dimension than in another which are common around _local_ optima.

    # SGD update
    x = x - eta * dx

where `eta` is a hyperparameter (constant) and `dx` is the gradient. 

**Momentum** is a method that helps accelerate SGD in the relevant direction and dampens oscillations by adding a fraction gamma of the update vector of the past time step to the current update vector.

    # Momentum update
    v = mu * v - eta * dx # integrate velocity
    x += v # integrate position

where v is an additional variable that is initialized at zero and we add an additional hyperparamter mu that is referred to as momentum.

Nesterov accelerated gradient (NAG) enhances the Momentum method by computing the gradient at x + mu * v instead of the current position x. This prevents us from going too fast and results in increased responsiveness.

    x_ahead = x + mu * v
    # evaluate dx_ahead (the gradient at x_ahead instead of at x)
    v = mu * v - learning_rate * dx_ahead
    x += v


Momentum and NAG manipulate the learning rate globally and equally for all parameters. Adagrad is an algorithm that adapts the learning rate to the parameters, performing smaller updates for parameters associated with frequently occurring features, and larger updates for parameters associated with infrequent features. For this reason, it is well-suited for dealing with sparse data.

One of Adagrad's main benefits is that it eliminates the need to manually tune the learning rate. Most implementations use a default value of 0.01.

    # Assume the gradient dx and parameter vector x
    cache += dx**2
    x += - learning_rate * dx / (np.sqrt(cache) + eps)

where cache keeps track of the per-parameter sum of squared gradients and the smoothing term eps (usually in the range of 1e-4 to 1e-8) avoids division by zero.

Adagrad's main weakness is its accumulation of the squared gradients in the denominator: the accumulated sum keeps growing during training which causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge. 

Adadelta is an extension of Adagrad that seeks to overcome this weekness.



## References

[A Gentle Introduction to Mini-Batch Gradient Descent](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)


