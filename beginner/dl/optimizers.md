# Optimization Functions

The article [1] covers the mathematical expressions of common non-convex optimizers and their Python implementations from scratch .

Understanding the math behind these optimization algorithms can help enlighten your perspective when training complex machine learning models.

- Stochastic Gradient Descent (SGD)
- SGDMomentum
- AdaGrad
- RMSprop
- Adam


## Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is an iterative, non-convex, and first-order optimization algorithm over the differentiable error surfaces.

SGD is a stochastic estimation of gradient descent in which the training data is randomized that is a computationally-stable and mathematically well-established optimization algorithm.

The intuition behind SGD is that we take the partial derivative of the objective function with respect to the parameter that we can to optimize that yields its gradient which shows the increasing direction of the error loss. Thus, we take the negative of that gradient to step forward where the loss is not increasing.

To ensure stable and less-oscillatory optimization, we introduce the learning rate parameter ŋ then multiply the gradient with ŋ.

Finally, the obtained value is subtracted from the parameter that we can optimize in an iterative fashion.


## SGDMomentum

For SGD, rather than computing the exact derivate of our loss function, we are approximating it on small batches in an iterative fashion. Thus, it is not certain that the model learns in a direction where the loss is minimized.

For more stable, direction-aware, and faster learning, we introduce SGDMomentum which determines the next update as a linear combination of the gradient and the previous update. Thus, SGDMomentum also takes into account the previous updates.

In general, momentum stochastic gradient descent provides two advantages over classical SGD:

- Fast convergence
- Less oscillatory training

Here is the formula and the Python code for SGDMomentum:

where α is the momentum coefficient which takes values on [0,1].

Alpha (α) is an exponential decay factor that determines the relative contribution of the current gradient and earlier gradients to the weight change [1].

If α = 0, the formula is just pure SGD.

If α = 1, the optimization process takes into account the full history of the previous update.

There are some other widely used optimization algorithms such as AdaGrad, RMSprop, and Adam that are adaptive optimization algorithms which means they adapt the process of learning by rearranging the learning rate so that the model can reach a global minima more efficiently and faster.


## AdaGrad

The AdaGrad optimization algorithm keeps track of the sum of the squared gradients that decay the learning rate for parameters in propagation to their update history.

The mathematical expression is:

AdaGrad is stuck when close to convergence since the cumulative sum is increased gradually so that the overall update term is significantly decreased.


## RMSprop

The RMSprop optimization algorithms fixes the issue of AdaGrad by multiplying decaying rate by the cumulative sum and enables us to forget the history of cumulative sum after a certain point which depends on the decay term that helps to converge to global minima.

The mathematical expression for RMSprop is given by:

where 𝛼 is the decaying term.

This, the decaying term provides faster convergence and forgets the cumulative sum of gradient history that results in a more optimized solution.


## Adam

The Adam optimization algorithm is a more developed version of the RMSprop that takes the first and second momentum of the gradient separately.

Thus, Adam also fixes the slow convergence issue in converging to the global minima.

The mathematical expression for Adam is given by:

where 𝛿𝑀𝑖 is the first-moment decaying cumulative sum of gradients, 𝛿𝑉𝑖 is the second-moment decaying cumulative sum of gradients, the hat notation 𝛿𝑀, and 𝛿𝑉 are bias-corrected values of 𝛿𝑀 and 𝛿𝑉𝑖 and ŋ is the learning rate.

## Summary

Many researchers and data scientists use the Adam optimizer rather than SGD for training large deep neural networks since the Adam optimizer is adaptive to process and has both first and second-order momentum compared to SGD.


## References

[1: [Neural Network Optimizers from Scratch in Python](https://towardsdatascience.com/neural-network-optimizers-from-scratch-in-python-af76ee087aab?source=rss----7f60cf5620c9---4)
