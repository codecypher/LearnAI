# Convolutional Neural Networks (CNN)

## Overview

Welcome to Part 4 of Applied Deep Learning series:

- Part 1 was a hands-on introduction to Artificial Neural Networks, covering both the theory and application with a lot of code examples and visualization. 

- In Part 2 we applied deep learning to real-world datasets, covering the 3 most commonly encountered problems as case studies: binary classification, multiclass classification and regression. 
- 
- In Part 3 we explored a specific deep learning architecture: Autoencoders.

Now we will cover the most popular deep learning model: Convolutional Neural Networks.

## 1. Introduction

Convolutional Neural Networks (CNN) are perhaps the most popular deep learning architecture. The recent surge of interest in deep learning is due to the immense popularity and effectiveness of convnets. 

The interest in CNN started with AlexNet in 2012 and it has grown exponentially ever since. In just three years, researchers progressed from 8 layer AlexNet to 152 layer ResNet.

CNN is now the go-to model on every **image related problem**. In terms of accuracy, they blow the competition out of the water. 

CNN has been successfully applied to recommender systems, natural language processing, and more. 

The main advantage of CNN compared to its predecessors is that it automatically detects the important features without any human supervision. 

For example, given many pictures of cats and dogs it learns distinctive features for each class by itself.

CNN is also computationally efficient. It uses special convolution and pooling operations and performs parameter sharing. This enables CNN models to run on any device, making them universally attractive.

## 2. Architecture

All CNN models follow a similar architecture which is shown in the figure below:

```
    Input  =>  Conv  =>  Pool  =>  Conv  =>  Pool  =>  FC  => FC  =>  Softmax
```

There is an input image that we ae working with. 

We perform a series of convolution + pooling operations, followed by a number of fully connected layers. 

If we are performing multiclass classification, the output is softmax.


### 2.1 Convolution

The main building block of CNN is the _convolutional layer_. 

Convolution is a mathematical operation to merge two sets of information. 

Here, the convolution is applied on the input data using a _convolution filter_ to produce a _feature map_.

On the left side is the input to the convolution layer such as the input image. 

On the right is the convolution _filter_ or _kernel_.

The example above is called a 3x3 _convolution_ due to the shape of the filter.

We perform the convolution operation by sliding the filter over the input. 

At every location, we do element-wise matrix multiplication and sum the result. The sum goes into the feature map. 

The green area where the convolution operation takes place is called the _receptive field_. 

Due to the size of the filter the receptive field is also 3x3.

This was an example convolution operation shown in 2D using a 3x3 filter. In reality, the convolutions are performed in 3D where an image is represented as a 3D matrix with dimensions of height, width and depth where depth corresponds to color channels (RGB). 

A convolution filter has a specific height and width such as 3x3 or 5x5 and by design it covers the entire depth of its input so it needs to be 3D as well.

We perform multiple convolutions on an input, each using a different filter and resulting in a distinct feature map. Then we stack all these feature maps together which becomes the final output of the convolution layer. 

First, let us start simple and visualize a convolution using a single filter.

Supposee we have a 32x32x3 image and we use a filter of size 5x5x3 (the depth of the convolution filter matches the depth of the image, both being 3). When the filter is at a particular location it covers a small volume of the input, and we perform the convolution operation described above. 

The only difference is this time we do the sum of matrix multiply in 3D instead of 2D, but the result is still a scalar. 

We slide the filter over the input like above and perform the convolution at every location aggregating the result in a feature map. The feature map is of size 32x32x1, shown as the red slice on the right.

If we used 10 different filters, we would have 10 feature maps of size 32x32x1 and stacking them along the depth dimension would give us the final output of the convolution layer: a volume of size 32x32x10, shown as the large blue box on the right. 

Note that the height and width of the feature map are unchanged and still 32 which is due to padding which we will discuss shortly.

Below we can see how two feature maps are stacked along the depth dimension. The convolution operation for each filter is performed independently and the resulting feature maps are disjoint.

### 2.2 Non-linearity

For any neural network to be powerful, it needs to contain non-linearity. 

Both the ANN and autoencoder we saw before achieved this by passing the weighted sum of its inputs through an activation function and CNN is no different. 

We pass the result of the convolution operation through the ReLu activation function. The values in the final feature maps are not actually the sums, but the relu function applied to them. We have omitted this in the figures above for simplicity. 

Keep in mind that any type of convolution involves a relu operation or the network will not achieve its true potential.

### 2.3 Stride and Padding

_Stride_ specifies how much we move the convolution filter at each step. By default the value is 1, as you can see in the figure below.

We can have bigger strides if we want less overlap between the receptive fields. This also makes the resulting feature map smaller since we are skipping over potential locations. 

The following figure demonstrates a stride of 2 (the feature map is now smaller).

We see that the size of the feature map is smaller than the input because the convolution filter needs to be contained in the input. 

If we want to maintain the same dimensionality, we can use _padding_ to surround the input with zeros (see the animation below).

The gray area around the input is the padding. We either pad with zeros or the values on the edge. 

Now the dimensionality of the feature map matches the input. Padding is commonly used in CNN to preserve the size of the feature maps, otherwise it would shrink at each layer which is not desirable. 

The 3D convolution figures we saw above used padding which is why the height and width of the feature map was the same as the input (both 32x32) and only the depth changed.

### 2.4 Pooling

After a convolution operation, we usually perform _pooling_ to reduce the dimensionality which allows us to reduce the number of parameters which shortens the training time and prevents overfitting. 

Pooling layers downsample each feature map independently (reducing the height and width), keeping the depth intact.

The most common type of pooling is _max pooling_ which takes the max value in the pooling window. Unlike the convolution operation, pooling has no parameters. It slides a window over its input and simply takes the max value in the window. Similar to a convolution, we specify the window size and stride.

Here is the result of max pooling using a 2x2 window and stride 2 where each color denotes a different window. Since both the window size and stride are 2, the windows are not overlapping.

Note that this window and stride configuration halves the size of the feature map which is the main use case of pooling -- _downsampling_ the feature map while keeping the important information.

Now let us work out the feature map dimensions before and after pooling:

- If the input to the pooling layer has dimensions 32x32x10 then using the same pooling parameters described above, the result will be a 16x16x10 feature map. 

- Both the height and width of the feature map are halved, but the depth does not change because pooling works independently on each depth slice of the input.

By halving the height and the width, we have reduced the number of weights to 1/4 of the input. Considering that we typically deal with millions of weights in CNN architectures, this reduction is a pretty big deal.

In CNN architectures, pooling is typically performed with 2x2 windows, stride 2 and no padding. 

When convolution is done with 3x3 windows, stride 1 and padding are used.

### 2.5 Hyperparameters

Let us consider only a convolution layer (ignoring pooling) and go over the hyperparameter choices we need to make. 

We have 4 important hyperparameters to decide on:

- filter size: We typically use 3x3 filters but 5x5 or 7x7 are also used depending on the application. There are also 1x1 filters which we will explore in another article, it might look strange but they have interesting applications. Remember that these filters are 3D and have a depth dimension as well but since the depth of a filter at a given layer is equal to the depth of its input, we omit that.

- Filter count: This is the most variable parameter, it is a power of two anywhere between 32 and 1024. Using more filters results in a more powerful model but we risk overfitting due to increased parameter count. Usually we start with a small number of filters at the initial layers and progressively increase the count as we go deeper into the network.

- Stride: We keep it at the default value of 1.

- Padding: We usually use padding.

### 2.6 Fully Connected

After the convolution + pooling layers, we add a couple of fully connected layers to wrap up the CNN architecture. This is the same fully connected ANN architecture we talked about in Part 1.

Remember that the output of both convolution and pooling layers are 3D volumes but a fully connected layer expects a 1D vector of numbers. So we _flatten_ the output of the final pooling layer to a vector and that becomes the input to the fully connected layer. Flattening is simply arranging the 3D volume of numbers into a 1D vector.

### 2.7 Training

CNN is trained the same way like ANN, backpropagation with gradient descent. 

Due to the convolution operation it is more mathematically involved which is out of the scope of this article. If you are interested in the details refer here.


## 3. Intuition

A CNN model can be thought as a combination of **two components:** feature extraction part and the classification part. The convolution + pooling layers perform feature extraction. 

Example: Given an image, the convolution layer detects features such as two eyes, long ears, four legs, a short tail and so on. The fully connected layers then act as a classifier on top of these features and assign a probability for the input image being a dog.

The convolution layers are the main powerhouse of a CNN model:

Automatically detecting meaningful features given only an image and a label is not an easy task. The convolution layers learn such complex features by building on top of each other. The first layers detect edges, the next layers combine them to detect shapes, and the following layers merge this information to infer that this is a nose. 

The CNN does not know what a nose is. By seeing a lot of them in images, it learns to detect that as a feature. The fully connected layers learn how to use these features produced by convolutions in order to correctly classify the images.

All this might sound vague right now, but hopefully the visualization section will make everything more clear.


## 4. Implementation

After this lengthy explanation let us code up our CNN. 

We will use the **Dogs vs Cats** dataset from Kaggle to distinguish dog photos from cats.

We will use the following architecture: 4 convolution + pooling layers, followed by 2 fully connected layers. The input is an image of a cat or dog and the output is binary.

Structurally the code looks similar to the ANN we have been working on. 

However, there are 4  methods we have not seen before:

- Conv2D: this method creates a convolutional layer. The first parameter is the filter count,] and the second parameter is the filter size. 

  For example, in the first convolution layer we create 32 filters of size 3x3. We use relu non-linearity as activation, and we also enable padding. 

  In Keras there are two options for padding: same or valid. _Same_ means we pad with the number on the edge and _valid_ means no padding. 

  Stride is 1 for convolution layers by default so we do not change that. This layer can be customized further with additional parameters, you can see the documentation here.

- MaxPooling2D: creates a maxpooling layer, the only argument is the window size. We use a 2x2 window since it is the most common. By default, stride length is equal to the window size which is 2 here, so we do not change that.

- Flatten: After the convolution + pooling layers, we flatten their output to feed into the fully connected layers as we discussed above.

- Dropout: we will explain this in the next section.

### 4.1) Dropout

Dropout is by far the most popular _regularization_ technique for deep neural networks. Even the state-of-the-art models which have 95% accuracy get a 2% accuracy boost just by adding dropout which is a fairly substantial gain at that level of accuracy.

Dropout is used to prevent _overfitting_ and the idea is very simple. During training , at each iteration, a neuron is temporarily “dropped” or disabled with probability p. This means all the inputs and outputs to this neuron will be disabled during the current iteration. The dropped-out neurons are resampled with probability p at every training step, so a dropped out neuron at one step can be active at the next step. The hyperparameter p is called the dropout-rate and it is typically a number around 0.5, corresponding to 50% of the neurons being dropped out.

It is surprising that dropout works at all. We are disabling neurons on purpose and the network actually performs better. The reason is that dropout prevents the network from being too dependent on a small number of neurons and forces every neuron to be able to operate independently. This might sound familiar from constraining the code size of the autoencoder in Part 3, in order to learn more intelligent representations.

Let us visualize dropout, so it will be much easier to understand.

Dropout can be applied to input or hidden layer nodes but not the output nodes. The edges in and out of the dropped nodes are disabled. Remember that the nodes that are dropped out change at each training step. Also we do not apply dropout during testing after the network is trained, we onlydo so during training.

Almost all state-of-the-art deep networks now incorporate dropout. There is another very popular regularization technique called _batch normalization_ which we will cover in another article.

### 4.2 Model Performance

Let us now analyze the performance of our model. We will take a look at **loss and accuracy** curves, comparing training set performance against the validation set.

Training loss keeps going down, but the validation loss starts increasing after around epoch 10. This is the textbook definition of overfitting. The model is memorizing the training data, but it is failing to generalize to new instances, and that is why the validation performance gets worse.

We are overfitting despite the fact that we are using dropout. The reason is that we are training on very few examples, 1000 images per category. Usually we need at least 100K training examples to start thinking about deep learning. 

No matter which regularization technique we use, we will overfit on such a small dataset. But fortunately there is a solution to this problem which enables us to train deep models on small datasets, and it is called _data augmentation_.

### 4.3 Data Augmentation

Overfitting happens because of having too few examples to train on, resulting in a model that has poor generalization performance. If we had infinite training data, we would not overfit because we would see every possible instance.

The common case in most machine learning applications (especially in image classification tasks) is that obtaining new training data is not easy. Therefore, we need to make do with the training set at hand. 

Data augmentation is a way to generate more training data from our current set. It enriches or “augments” the training data by generating new examples via random transformation of existing ones. This way we artificially boost the size of the training set, reducing overfitting. So data augmentation can also be considered as a _regularization_ technique.

Data augmentation is done dynamically during training. We need to generate realistic images and the transformations should be learnable, simply adding noise will not help. 

**Common transformations are:** rotation, shifting, resizing, exposure adjustment, contrast change etc. This way we can generate a lot of new samples from a single training example. Also, data augmentation is only performed on the training data, we do not touch the validation or test set.

Visualization will help understanding the concept. Let us say this is our original image.

Using data augmentation we generate these artificial training instances. These are new training instances, applying transformations to the original image does not change the fact that this is still a cat image. We can infer it as a human, so the model should be able to learn that as well.

Data augmentation can boost the size of the training set by even 50x. It is a very powerful technique that is used in every single image-based deep learning model.

There are some data cleaning tricks that we typically use on images, mainly _whitening_ and _mean normalization_. More information about them is available here.

### 4.4 Updated Model

Now let us use data augmentation in our CNN model. The code for the model definition will not change at all, since we are not changing the architecture of our model. 

The only change is how we feed in the data, you can check the jupyter notebook here.

It is pretty easy to do data augmentation with Keras, it provides a class which does all the work for us, we only need to specify some parameters. The documentation is available here.

The loss and accuracy curves look as follows using data augmentation.

This time there is no apparent overfitting and validation accuracy jumped from 73% with no data augmentation to 81% with data augmentation, an 11% improvement which is a pretty big deal. 

There are two main reasons that the accuracy improved:

  1. We are training on more images with variety. 

  2. We made the model transformation invariant which means the model saw a lot of shifted/rotated/scaled images so it is able to recognize them better.


## 5. VGG Model

Let us now take a look at an example state-of-the art CNN model from 2014. 

VGG is a convolutional neural network from researchers at Oxford’s Visual Geometry Group, hence the name VGG. It was the runner up of the ImageNet classification challenge with 7.3% error rate. 

ImageNet is the most comprehensive hand-annotated visual dataset and they hold competitions every year where researchers from all around the world compete. All the famous CNN architectures make their debut at that competition.

Among the best performing CNN models, VGG is remarkable for its simplicity. 

Let us take a look at its architecture.


## 6. Visualization

Now comes the most fun and interesting part, visualization of CNN. 

Deep learning models are known to be very hard to interpret which is why they are usually treated as black boxes. But CNN models are actually the opposite and we can visualize various components. This will give us an in-depth look into their internal workings and help us understand them better.

We will visualize the 3 most crucial components of the VGG model:

- Feature maps
- Convnet filters
- Class output


## 7. Conclusion

CNN is a very fundamental deep learning technique. We covered a wide range of topics and the visualization section in my opinion is the most interesting. 

There are very few resources on the web which do a thorough visual exploration of convolution filters and feature maps, so I hope it was helpful.

The entire code for this article is available here if you want to hack on it yourself.


----------


## Basics of Convolutional Neural Networks

[Gentle Introduction to Convolutional Layers in CNNS](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)

[Gentle Introduction to Padding and Stride in CNNs](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/)

[Gentle Introduction to Pooling Layers in CNNs](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)


## CNN using PyTorch

[Pytorch [Basics] — Intro to CNN](https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29)

[Complete Guide to build CNN in Pytorch and Keras](https://medium.com/analytics-vidhya/complete-guide-to-build-cnn-in-pytorch-and-keras-abc9ed8b8160)


## References

[1] [Convolutional Neural Networks (CNN)](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)


