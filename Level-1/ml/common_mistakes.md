# Common Machine Learning Mistakes

## Overview

There are five common mistakes that programmers make when getting started in machine learning:

- Don’t put machine learning on a pedestal
- Don’t write machine learning code
- Don’t do things manually
- Don’t reinvent solutions to common problems
- Don’t ignore the math

The proper technique to learning ML is to take a top-down approach. It is best not to waste time trying to hand code algorithms (bottom-up approach) when you are just getting started. 

Once you have mastered using ML algorithms then go back and try to understand how they work internally by coding them by hand. 

In short, it is best to take a data-centric rather than model-centric approach. 


## Put Machine Learning on a pedestal

A mindset shift is required to be effective at machine learning from technology to processsng from precision to “good enough”.

It is also best to take a black-box view of ML algorithms and avoid the need to understand how they work internally. Most ML algorithms have very little in common internally. 


## Write Machine Learning Code

You do not have to implement machine learning algorithms when getting started in machine learning.

Starting in machine learning by writing code can make things difficult because it means that you are solving at least two problems rather than one: 1) how a technique works so that you can implement it and 2) how to apply the technique to a given problem.

Thus, it is better to learn how to use machine learning algorithms before implementing them.

Implementing an algorithm can be treated as a separate project to be completed at a later time such as a learning exercise. 

## Doing Things Manually

Automation is a big part of modern software development for builds, tests, and deployment. There is great advantage in scripting data preparation, algorithm testing and tuning, and the preparation of results in order to gain the benefits of rigor and speed of improvement.

## Reinvent Solutions to Common Problems

Hundreds and thousands of people have likely implemented the algorithm you are implementing before you or have solved a problem type similar to the problem you are solving, so exploit their lessons learned.

## Ignoring the Math

You do not need the mathematical theory to get started but mathematics is a big part of machine learning. The reason is that math provides the most efficient and unambiguous way to describe problems and the behaviors of systems.

Ignoring the mathematical treatments of algorithms can lead to problems such as having a limited understanding of a method or adopting a limited interpretation of an algorithm. 

For example, many machine learning algorithms have an optimization at their core that is incrementally updated. Knowing about the nature of the optimization being solved (the function convex) allows you to use efficient optimization algorithms that exploit this knowledge.

Internalizing the mathematical treatment of algorithms is slow and comes with mastery. Particularly if you are implementing advanced algorithms from scratch including the internal optimization algorithms, take the time to learn the algorithm from a mathematical perspective.



## Python Mistakes To Avoid

### Always looping over ranges

Ranges are often used just as well to get the elements of multiple lists at once but it is usually better to use either `enumerate()` or `zip()`. 

### Import *

```py
  # create an alias instead
  import numpy as np
```

### String Concatenation

if you are concetanating a lot of values, it is best to use f-strings which  allow you to interject strings with some level of interpolation.

### Using bare excepts

Using bare exceptions inside of try/except blocks is a bad praxtice because these exceptions are open-ended. 

Whenever an exception is open-ended, we leave room for just about any exception to come about. We could end up processing an interrupt exception under the block and our code will run for no reason because it was interrupted. Instead, we should use a real exception.

### Mutable Defaults

Providing mutable defaults for positional arguments. 

Argument defaults are made whenever a given function is defined, not when it is ran.

```py
def hello(x : int, y = []):
```

This means every call to this function is now sharing y, so if we were to append something to y then every subsequent call to the `hello()` function would use this y as a default, so we would be using a y that is now littered with values. 

Instead, we should set y to `None` as a placeholder and then change it with a conditional.

```py
def hello(X : int, y = None):
    if y is None:
        y = []
```

### Checking types with ==

There are some cases where you might want to check your types like this, but in general it should be avoided.

```py
if typeof(x) == tuple:
    pass
```

Instead, we should probably use the `isinstance()` method which is able to check directly if a given pyobject is an instance of a constructor.

```py
if isinstance(x, tuple):
```

### Equality for singletons

Checking equality rather than identity. 

The bitwise equality operator == is familiar and something we know plus it works, so it makes sense why a lot of people do end up doing this. 

If we want to check if something is true, false, or is defined as nothing, we should be using `is`. 

```py
    if x is None
```



## References

[Stop Coding Machine Learning Algorithms From Scratch](https://machinelearningmastery.com/dont-implement-machine-learning-algorithms/)

[5 Mistakes Programmers Make when Starting in Machine Learning](https://machinelearningmastery.com/mistakes-programmers-make-when-starting-in-machine-learning/)

[10 Atrocious Python Mistakes To Avoid](https://towardsdatascience.com/10-atrocious-python-mistakes-to-avoid-12fb228d60a1)

