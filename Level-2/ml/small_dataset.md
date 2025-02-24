# Dealing With Small Dataset

## Overview

Keep these points in mind when dealing with small datasets [1]:

- Realize that your model will not generalize that well.

- Do some data augmentation.

- Generate some synthetic data.

- Beware of lucky splits: k-fold Cross-Validation is a better choice.

- Use transfer learning.

- Try an ensemble of weak learners such as SVM.



## How to address the problem of less data

The article [2] discusses how the size of the data set impacts traditional Machine Learning algorithms and few ways to mitigate these issues.

### Change the loss function

For classification problems, we often use cross-entropy loss and rarely use mean absolute error or mean squared error to train and optimize our model. 

With unbalanced data, the model becomes more biased towards the majority class since it has a larger influence on the final loss value and our model becomes less useful. Thus, we can add weights to the losses corresponding to different classes to even out this data bias. 

For example, if we have two classes with data in the ration 4:1, we can apply weights in the ratio 1:4 to the loss function calculation to make the data balanced which helps us easily mitigate the issue of unbalanced data and improves model generalization across different classes. There are libraries in both R and Python that can help in assigning weights to classes during loss calculation and optimization. 

Scikit-learn has a convenient utility function to calculate the weights based on class frequencies:

```py
  from sklearn.utils.class_weight import compute_class_weight
  from sklearn.linear_model import LogisticRegression
  
  # ['balanced', 'calculated balanced', 'normalized']
  class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)

  model = LogisticRegression(class_weight = class_weight)
  model.fit(X_train, y_train)
```


We can avoid the above calculation by using class_weight=`balanced` which does the same calculations to find class_weights.
We can also feed explicit class weights as per our requirements. 

```py
  from sklearn.linear_model import LogisticRegression

  # Create decision tree classifer object
  clf = LogisticRegression(random_state=0, class_weight='balanced')

  # Train model
  model = clf.fit(X_std, y)
```

### Anomaly/Change detection

In cases of highly imbalanced data sets such as fraud or machine failure, it is worth pondering if such examples can be considered as Anomaly or not. 

If the given problem meets the criterion of Anomaly, we can use models such as OneClassSVM, Clustering methods, or Gaussian Anomaly detection methods. 

These techniques require a shift in thinking where we consider the minor class as the outliers class which might help us find new ways to separate and classify. 

Change detection is similar to anomaly detection except we look for a change or difference instead of an anomaly. 

These might be changes in the behavior of a user as observed by usage patterns or bank transactions. 

### Up-sample or Down-sample

Since unbalanced data inherently penalizes majority class at different weight compared to a minority class, one solution to the problem is to make the data balanced which can be done either by increasing the frequency of minority class or by reducing the frequency of majority class through random or clustered sampling techniques. 

The choice of over-sampling vs under-sampling and random vs clustered is determined by business context and data size. 

Generally upsampling is preferred when the overall data size is small while downsampling is useful when we have a large amount of data. Similarly, random vs clustered sampling is determined by how well the data is distributed. 

Resampling can be easily done with the help of imblearn package as shown below:

```py
  from imblearn.under_sampling import RandomUnderSampler

  rus = RandomUnderSampler(random_state=42)
  X_res, y_res = rus.fit_resample(X, y)
```

### Generate Synthetic Data

Although upsampling or downsampling helps in making the data balanced, duplicate data increases the chances of overfitting. 

Another approach to address this issue is to generate synthetic data with the help of minority class data. 

Synthetic Minority Over-sampling Technique (SMOTE) and Modified- SMOTE are techniques that generate synthetic data. 

In short, SMOTE takes the minority class data points and creates new data points that lie between any two nearest data points joined by a straight line. The algorithm calculates the distance between two data points in the feature space, multiplies the distance by a random number between 0 and 1 and places the new data point at this new distance from one of the data points used for distance calculation. 

Note the number of nearest neighbors considered for data generation is also a hyperparameter and can be changed based on requirement.

### Ensembling Techniques

The idea of aggregating multiple weak learners/different models has shown great results when dealing with imbalanced data sets. 

Both Bagging and Boosting techniques have shown great results across a variety of problems and should be explored along with methods discussed above to get better results.


### Key factors in training Neural Nets with small dataset

Here are a few important factors which influence the network optimization process [2]:

- **Optimization Algorithm:** Adam, RMSprop, Adagrad, and Stochastic Gradient descent are a few variations of gradient descent which optimize the gradient update process and improve model performance.

- **Loss function:** Hinge loss is one example that makes training with small datasets possible. 

- **Parameter initialization:** The initial state of the parameters greatly influences the optimization process.

- **Data size:** Data size is a very crucial part of training neural networks. Larger datasets can help us better learn model parameters and improve the optimization process and imparts generalization.

Figure 1: Basic implications of fewer data and possible approaches and techniques to solve it


### Ways to overcome optimization difficulties

We discussed a few of the above techniques in Part 1. Here discuss the remaining techniques that are more relevant to deep learning [2].

1. Transfer learning

Transfer learning refers to the approach of using the learning from one task on to another task without the requirement of learning from scratch. It directly addresses the smart parameter initialization point for training neural networks. 

Using the Fast.ai library, we can build images classification model using transfer learning with just a few lines of code and a few hundred training images and still get state of the art results.

2. Problem Reduction

The problem reduction approach refers to modifying the new data or unknown problem to a known problem so that it can be easily solved using existing techniques.

3. Learning with less data

a) One Shot Learning: Humans have the ability to learn even with a single example and are still able to distinguish new objects with very high precision. However, deep neural networks require a huge amount of labeled data to train and generalize. This is a big drawback and one-shot learning is an attempt to train neural networks even with small data sets. 

There are two ways in which we can achieve this. We can either modify our loss function such that it can identify minute differences and learns a better representation of the data.

b) Siamese Network: Given a set of images, a Siamese Network tries to find how similar two given images are. The network has two identical sub-networks with same parameters and weights. 

The sub-networks consist of Convolutional blocks and have fully connected layer and extracts a feature vector(size 128) towards the end. The image set which needs to be compared are passed through the network to extract the feature vectors and we calculate the distance between the feature vectors. 

The model performance depends on training image pairs(closer the pairs better the performance) and model is optimized such that we get a lower loss for similar images and higher loss for different images. 

Siamese network is a good example of how we can modify the loss function and use fewer yet quality training data to train deep learning models. 

c) Memory Augmented Neural Networks: Neural Turing Machine (NTM) is a part of Memory augmented neural networks that tries to create an external memory for a neural network which can help in One-Short learning. 

NTM is fundamentally composed of a neural network called the controller and a 2D matrix called the memory bank. 

At each time step, the neural network receives some input from the outside world and sends some output to the outside world. However, the network also has the ability to read from memory locations and the ability to write to memory locations. Note that back-propagation will not work if we extract the memory using the index. 

Thus, the controller reads and writes using blurry operation -- it assigns different weights to each location while reading and writing. The Controller produces weightings over memory locations that allow it to specify memory locations in a differentiable manner.

d) Zero-Shot Learning: Zero-Shot learning refers to the method of solving tasks which were not a part of training data which can help us work with classes we did not see during training and reduces data requirements. 

There are various ways for formulating the task of zero short learning and I am going to discuss one such method. 

4. Better optimization techniques

**Meta-Learning (learning to learn)** is concerned with finding the best ways to learn from given data -- learning various optimization settings and hyper-parameters for the model. 

There are various ways of implementing Meta-Learning and we will discuss one such method. 

A meta-learning framework typically consists of a network which has two models:

- A neural network called Optimize or a Learner which is treated as a low-level network and is used for prediction.

- We have another neural network which is called Optimizer or Meta-Learner or High-Level model which updates the weights of the lower-level network.

This results in a two-way nested training process. 

We take multiple steps of the low-level network which forms a single step of meta-learner.

We also calculate a meta loss at the end of these steps of the low-level network and update the weights of the meta-learner accordingly. 

This process helps us figure out the best parameters to train on makes the learning process more efficient. 

### Addressing the lack of Generalization

1. Data Augmentation

Data augmentation (DA) can be an effective tool when dealing with a small dataset without overfitting. 

DA is also a good technique to make our model invariant to changes in size, translation, viewpoint, illumination, etc. 

We can achieve this by augmenting our data in a few of the following ways:

- Flip the image horizontally or vertically
- Crop and/or zoom images
- Change the brightness/sharpness/contrast of the image
- Rotate the image by some degree

Fast.ai has some of the best transform functions for data augmentation which makes the data augmentation task very easy using just a few lines of codes. 

2. Data Generation

- Semi-Supervised Learning: A lot of times we have a large corpus of data available but only a smart part of it is labeled. The large corpus can be any publically available data set or proprietary data. 

In such scenarios, semi-supervised learning can be a good technique to solve the problem of less labeled data.

- GAN: Generative adversarial networks are a type of generative models which can generate new data that looks very close to real data.


## References

[1]: [7 Tips for Dealing With Small Data](https://towardsdatascience.com/7-tips-for-dealing-with-small-data-7ffbd3d399a3)

[2]: [Breaking the curse of small data sets in Machine Learning: Part 1/2](https://towardsdatascience.com/breaking-the-curse-of-small-data-sets-in-machine-learning-part-2-894aa45277f4)

[3]: [Is a Small Dataset Risky?](https://towardsdatascience.com/is-a-small-dataset-risky-b664b8569a21)


[OpenAI Glow and the Art of Learning from Small Datasets](https://jrodthoughts.medium.com/openai-glow-and-the-art-of-learning-from-small-datasets-e6b0a0cd6fe4)

[Oversampling: A PCA-KNN Approach](https://towardsdatascience.com/oversampling-a-pca-knn-approach-f392ca232486)


[Explainability Is a Poor Band-Aid for Biased AI in Medicine](https://fperrywilson.medium.com/explainability-is-a-poor-band-aid-for-biased-ai-in-medicine-3db62a338857)
