# Machine Learning Cheatsheet

## Compute-centric vs Data-centric

Machine learning is bifurcating between two computing paradigms: compute-centric computing and data-centric computing. 

In the compute-centric paradigm, data is stockpiled and analyzed by instances in data centers. 

In the data-centric paradigm, the processing is done locally at the origin of the data. 

Although we appear to be quickly moving towards a ceiling in the compute-centric paradigm, work in the data-centric paradigm has only just begun.

**Tiny machine learning (tinyML)** is the intersection of machine learning and embedded internet of things (IoT) devices. The field is an emerging engineering discipline that has the potential to revolutionize many industries.


## Maximum Likelihood Estimation

In statistics, Maximum Likelihood Estimation (MLE) is a way of finding the best possible parameters which make the observed data most probable by finding parameters θ that maximize the likelihood function.

We can also define the loss function to be a measure of how bad our model is therefore we define loss function as -l(θ).

Now, the problem of MLE is the same as  finding the set of parameters which minimize the Negative Log Likelihood Loss for the Logistic Regression Model.


## Empirical risk minimization

In general, the risk cannot be computed because the distribution P(x,y) is unknown to the learning algorithm (referred to as agnostic learning). However, we can compute an approximation called empirical risk by averaging the loss function on the training set. 

The empirical risk minimization principle states that the learning algorithm should choose a hypothesis h which minimizes the empirical risk. 

Thus, the learning algorithm defined by the ERM principle consists in solving the above optimization problem.



## Neural Networks

### Layers

#### Convolution

In CNN, a _convolution_ is a linear operation that involves multiplication of weight (kernel/filter) with the input and it does most of the heavy lifting job.

Convolution layer consists of two major components: Kernel(Filter) and Stride

1. Kernel (Filter): A convolution layer can have more than one filter. 

   The size of the filter should be smaller than the size of input dimension which is intentional since it allows filter to be applied multiple times at difference point (position) on the input.
   
   Filters are helpful in understanding and identifying important features from given input. 
   
   Applying different filters (more than one filter) on the same input can be help to extract different features from the given input. 
   
   Output from multiplying filter with the input gives 2-dimensional array. 
   
   Thus, the output array from this operation is called a **Feature Map**.

2. Stride: This property controls the movement of filter over input. 

   When the value is set to 1, the filter moves 1 column at a time over input. 
   
   When the value is set to 2, the filter jump 2 columns at a time as filter moves over the input.

#### Dropout

A dropout layer takes the output of the previous layer’s activations and randomly sets a certain fraction (dropout rate) of the activatons to 0, cancelling or dropping them out.

Dropout is a common regularization technique used to prevent overfitting in Neural Networks.

The _dropout rate_ is the tunable hyperparameter that is adjusted to measure performance with different values. 

The dropout rate is typically set between 0.2 and 0.5 (but may be arbitrarily set).

Dropout is only used during training; At test time, no activations are dropped, but scaled down by a factor of dropout rate to account for more units being active during test time than training time.

The premise behind dropout is to introduce noise into a layer in order to disrupt any interdependent learning or coincidental patterns that may occur between units in the layer which are not significant.

#### Pooling

Pooling layers often take convolution layers as input. 

A complicated dataset with many objects will require a large number of filters, each responsible for finding patterns in an image, so the dimensionally of convolutional layer can get large which will cause an increase of parameters, leading to over-fitting. 

Pooling layers are methods for reducing this high dimensionally. 

Just like the convolution layer, there is kernel size and stride. 

The size of the kernel is smaller than the feature map. 

In most cases, the size of the kernel will be 2X2 with stride of 2. 

There are mainly two types of pooling layers:

1. Max pooling layer: this layer will take a stack of feature maps (convolution layer) as input. The value of the node in the max pooling layer is calculated using the maximum of the pixels contained in the window.

2. Average Pooling layer: this layer calculates the average of pixels contained in the window. This layer is not used often but you may see this used in applications for which smoothing an image is preferable.

### Loss Functions

#### Cross-Entropy

Cross-entropy loss or log loss measures the performance of a classification model whose output is a probability value between 0 and 1. 

Cross-entropy loss increases as the predicted probability diverges from the actual label. 

For example, predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. 

A perfect model would have a log loss of 0.

The graph above shows the range of possible loss values given a true observation (isDog = 1). 

As the predicted probability approaches 1, log loss slowly decreases. 

As the predicted probability decreases, the log loss increases rapidly. 

Log loss penalizes both types of errors, but especially those predictions that are confident and wrong!

Cross-entropy and log loss are slightly different depending on context, but in machine learning when calculating error rates between 0 and 1 they resolve to the same thing.

#### Mean Absolute Error (L1)

Mean Absolute Error or L1 loss.

#### MSE (L2)

Mean Squared Error or L2 loss. 


### Optimizers

#### What is Optimizer?

It is important to tweak the weights of the model during the training process, to make predictions as correct and optimized as possible.

But how exactly do you do that?

The best answer to the question is **optimizers**. 

Optimizers tie together the loss function and model parameters by updating the model in response to the output of the loss function. 

In simpler terms, optimizers shape and mold your model into its most accurate possible form by adjustijg the weights. 

The loss function is the guide during trainign which tells the optimizer when it is moving in the right or wrong direction.

Below are list of example optimizers:


### Regularization

#### What is overfitting?

Overfitting is the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably. 

#### What is Regularization?

Regularization is a technique for preventing overfitting and improving training.

- Data Augmentation
- Dropout
- Early Stopping
- Ensembling
- Injecting Noise
- L1 Regularization
- L2 Regularization


### Architectures

- Autoencoder
- CNN
- GAN
- MLP
- RNN
- VAE

#### Autoencoder

An _autoencoder_ is a neural network that is trained to copy its input to its output. 

Internally, it has a hidden layer h that describes a **code** used to represent the input. 

The network may be viewed as consisting of two parts: an encoder function h=f(x) and a decoder that produces a reconstruction r=g(h).

Figure 14.1  The general structure of an autoencoder, mapping an input x to an output (called reconstruction)r through an internal representation or code h. The autoencoder has two components: the encoder f (mapping x to h) and the decoder g (mapping h to r)
 
If an autoencoder succeeds in simply learning to set g(f(x)) = x everywhere, it is not especially useful. 

Thus, autoencoders are designed to be unable to learn to copy perfectly. 

Usually they are restricted in ways that allow them to copy only approximately, and to copy only input that resembles the training data. 

Because the model is forced to prioritize which aspects of the input should be copied, it often learns useful properties of the data.

Modern autoencoders have generalized the idea of an encoder and a decoder beyond deterministic functions to stochastic mappings: 

#### VAE

An atoencoder can encode an input image to a latent vector and decode it, but it cannot generate novel images. 

Variational Autoencoders (VAE) solve this problem by adding a constraint: the latent vector representation should model a unit gaussian distribution. 

The Encoder returns the mean and variance of the learned gaussian. 

To generate a new image, we pass a new mean and variance to the Decoder. 

In other words, we _sample a latent vector_ from the gaussian and pass it to the Decoder which also improves network generalization and avoids memorization. 

