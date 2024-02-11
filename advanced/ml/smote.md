# SMOTE for Imbalanced Classification

Imbalanced classification involves developing predictive models on classification datasets that have a severe class imbalance.

The challenge of working with imbalanced datasets is that most machine learning techniques will ignore and in turn have poor performance on the minority class whereas it is usually performance on the minority class that is most important.

## Tutorial Overview

The tutorial [1] is divided into five parts:

1. Synthetic Minority Oversampling Technique
2. Imbalanced-Learn Library
3. SMOTE for Balancing Data
4. SMOTE for Classification
5. SMOTE With Selective Synthetic Sample Generation
   - Borderline-SMOTE
   - Borderline-SMOTE SVM
   - Adaptive Synthetic Sampling (ADASYN)


## Synthetic Minority Oversampling Technique

A problem with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary.

One way to solve this problem is to **oversample** the examples in the minority class which can be achieved by simply duplicating examples from the minority class in the training dataset prior to fitting a model which can balance the class distribution but does not provide any additional information to the model.

An improvement on duplicating examples from the minority class is to _synthesize_ new examples from the minority class which is a type of data augmentation for tabular data and can be very effective.

Perhaps the most widely used approach to synthesizing new examples is called the **Synthetic Minority Oversampling TEchnique  (SMOTE).**

SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space, and drawing a new sample at a point along that line.

This procedure can be used to create as many synthetic examples for the minority class as are required. 

The paper suggests first using random undersampling to trim the number of examples in the majority class then use SMOTE to oversample the minority class to balance the class distribution.

The combination of SMOTE and under-sampling performs better than plain under-sampling.

The approach is effective because new synthetic examples from the minority class are created that are plausible which means they relatively close in feature space to existing examples from the minority class.

A general downside of the approach is that synthetic examples are created without considering the majority class, possibly resulting in ambiguous examples if there is a strong overlap of the classes.

## SMOTE for Balancing Data

In this section, we will develop an intuition for the SMOTE by applying it to an imbalanced binary classification problem.

The original paper on SMOTE suggested combining SMOTE with **random undersampling** of the majority class.

The `imbalanced-learn` library supports random undersampling via the `RandomUnderSampler` class.


## SMOTE for Classification

In this section, we will look at how we can use SMOTE as a data preparation method when fitting and evaluating machine learning algorithms in scikit-learn.

First, we use our binary classification dataset from the previous section then fit and evaluate a decision tree algorithm.


## SMOTE With Selective Synthetic Sample Generation

We can be selective about the examples in the minority class that are oversampled using SMOTE.

In this section, we will review some extensions to SMOTE that are more selective regarding the examples from the minority class that provide the basis for generating new synthetic examples.

### Adaptive Synthetic Sampling (ADASYN)

Another approach involves generating synthetic samples inversely proportional to the density of the examples in the minority class.

This, we generate more synthetic examples in regions of the feature space where the density of minority examples is low and fewer or none where the density is high.

We can implement this procedure using the ADASYN class in the imbalanced-learn library.

Unlike Borderline-SMOTE, we can see that the examples that have the most class overlap have the most focus. 

On problems where these low density examples might be outliers, the ADASYN approach may put too much attention on these areas of the feature space which may result in worse model performance.

It may help to remove outliers prior to applying the oversampling procedure which might be a helpful heuristic to use in general. 


## Refrences

[1] [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

[2] [5 SMOTE Techniques for Oversampling your Imbalance Data](https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5?gi=29e5140d8e06)

[3] [How to handle Multiclass Imbalanced Data? Not SMOTE](https://towardsdatascience.com/how-to-handle-multiclass-imbalanced-data-say-no-to-smote-e9a7f393c310)


