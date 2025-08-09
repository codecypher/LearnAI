# Interpretability and Explainability (XAI)


## Overview

A ML model is _interpretable_ (trace input to output) if we can inspect the actual model and understand why it got a particular answer for a given input and how the answer would change when the input changes.

A ML model is _explainable_ (trace output to input) when it can help you understand why an output produced for a given input

Here, interpretability derives from inspecting the actual model whereas explainability can be provided by a separate process which means the model itself can be a hard-to-understand black box but an explanation module can summarize what the model does. 

Thus, a system is _interpretable_ if we can inspect the source code of the model and see what it is doing and a model is _explainable_ if we can make up a story about what it is doing - even if the system itself is an uninterpretable black box.



## Verification and Validation

To earn trust, any engineered systems must go through a _verification and validation (V&V)_ process. 

_Verification_ means that the product satisfies the specifications. 

_Validation_ means ensuring that the specifications actually meet the needs of the user and other affected parties. 

We have an elaborate V&V methodology for engineering in general and traditional software development done by human coders; much of which is applicable to AI systems. However, machine learning systems are different and demand a different V&V process which has not yet been fully developed. 

We need to verify the data that these systems learn from; we need to verify the accuracy and fairness of the results, even in the face of uncertainty that makes an exact result unknowable; and we need to verify that adversaries cannot unduly influence the model nor steal information by querying the resulting model.



## Testing vs XAI

Which would you trust: An experimental aircraft that has never flown before but has a detailed explanation of why it is safe (XAI) or an aircraft that safely completed 100 previous flights and has been carefully maintained but comes with no guaranted explanation (tested)?

Thus, testing may provide more confidence and trust than XAI. 


E. Alpaydin, Introduction to Machine Learning, 3rd ed., MIT Press, ISBN: 978-0262028189, 2014.

- confidence interval estimation
- cross-validation and bootstraping
- hypothesis testing
- comparing classification algorithms (expected error rate, ANOVA)
- comparing algorithms over multiple datasets
- multivariate tests


When comparing misclassifications errors, it is assumed that all misclassifications have the same cost. When this is not the case, the hypothesis tests should be based on risks taking a suitable loss function into account. Unfortunately, not much work has been done in this area. 

Similarly, these tests should be generalized from classification to regression to be able to assess the mean square errors of regression algorithms or to be able to compare the errors of two regression algorithms. 


In comparing two classification algorithms,  we are only testing whether they have the same expected error rate. 
 
LIME builds interpretable linear models that approximate whatever machine learning system you have.
 
SHAP (Shapley Additive exPlanations) uses the notion of a Shapley value (p. 628) to determine the contribution of each feature.


## Predictive vs Explanatory Tradeoff

Uncertainty quantification (UQ) is the science of quantitative characterization and reduction of uncertainties in both computational and real world applications. 

UQ tries to determine how likely certain outcomes are if some aspects of the system are not exactly known. 

An example would be to predict the acceleration of a human body in a head-on crash with another car: even if the speed was exactly known, small differences in the manufacturing of individual cars, how tightly every bolt has been tightened, etc. will lead to different results that can only be predicted in a statistical sense.


Uncertainty is sometimes classified into two categories which are primarily seen in medical applications: aletoric and epistemic (A&E). 

Aleatoric uncertainty is also known as statistical uncertainty, and is representative of unknowns that differ each time we run the same experiment. 

Epistemic uncertainty is also known as systematic uncertainty and is due to things one could in principle know but does not in practice which may be die to a measurement is not accurate, the model neglects certain effects, or particular data have been deliberately hidden.

To evaluate aleatoric uncertainties can be relatively straightforward where traditional (frequentist) probability is the most basic form using techniques such as the Monte Carlo method. 

To evaluate epistemic uncertainties, the efforts are made to understand the (lack of) knowledge of the system, process, or mechanism. Epistemic uncertainty is generally understood  via Bayesian probability where probabilities are interpreted as indicating how certain a rational person could be regarding a specific claim.

In mathematics, uncertainty is often characterized in terms of a probability distribution. Thus, epistemic uncertainty means not being certain what the relevant probability distribution is and aleatoric uncertainty means not being certain what a random sample drawn from a probability distribution will be.



The A&E approach can be more of an art form than rigorous statistics when applied to ML models since 1) many ML models rarely have a closed form analytical solution and 2) ML models are stochastic in nature. However, you can instead try to apply probabilistic models to the datasets (in a sense). 

A&E appears to be one of many techniques used in an attempt to evaluate DL models for XAI. However, A&E makes a lot of assumptions about the model and the datasets in the process. 

There are other more popular techniques that seem to be more interpretable and easier for the layperson to understand.

A&E may stem from the model-centric vs data-centric schism in the theoretical realm, so not sure how applicable it is in AI engineering. The ultimate goal of A&E is XAI but to do that requires you to look at the model rather than the data. 

In short, there is a predictive vs explanatory tradeoff.



## References

S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Upper Saddle River, NJ: Prentice Hall, ISBN: 978-0-13-604259-4, 2021.

E. Alpaydin, Introduction to Machine Learning, 3rd ed., MIT Press, ISBN: 978-0262028189, 2014.


[Uncertainty Quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification?wprov=sfti1)

[Uncertainty in Deep Learning — Aleatoric Uncertainty and Maximum Likelihood Estimation](https://towardsdatascience.com/uncertainty-in-deep-learning-aleatoric-uncertainty-and-maximum-likelihood-estimation-c7449ee13712)

[Predicting vs Explaining](https://towardsdatascience.com/predicting#-vs-explaining-69b516f90796)
