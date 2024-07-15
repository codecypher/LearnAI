# Design Patterns

## What is a Design Pattern?

A _design pattern_ is defined by four essential elements:

**Name:** A way of defining the general context and vocabulary of how and what the DP is used to solve.

**Problem:** Describes when to apply the pattern.

**Solution:** The way classes, interfaces and objects are designed to respond to a problem.

**Consequences:** The trade-offs we have to consider once we decide to apply a given DP.


## How to Classify Design Patterns?

Design patterns are categorized according to the type of reuse:

**Behavioral:** How responsibility is sharedand information is propagated through different objects.

**Structural:** Concerns about the interaction processes between the different objects and classes.

**Creational:** Allows decoupling and optimizing the creation steps of different objects.


## Common Design Patterns

1. Strategy

Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

2. Mediator

Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly, and it lets you vary their interaction independently.

3. State

Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.

4. Builder

Separate the construction of a complex object from its representation so that the same construction process can create different representations.

5. Prototype

Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.

6. Adapter

Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn’t otherwise because of incompatible interfaces.

7. Decorator

Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.



## Design Pattern Cheatsheet

The _decorator_ pattern is a software design pattern that allows us to dynamically add functionality to classes without creating subclasses and affecting the behavior of other objects of the same class [2]. 

By using the decorator pattern, we can easily generate different permutations of functionality that we might want without creating an increasing number of subclasses which making the code more complex and bloated.

Decorators are usually implemented as sub-interfaces of the main interface that we want to implement and store an object of the main interface’s type. 

The decorator will modify the methods that it wants to add functionality by overriding the methods in the original interface and calling the methods from the stored object.

With Python, we are able to simplify many design patterns due to dynamic typing along with functions and classes being first-class objects as well as Python’s decorator syntax. 

Figure 1: UML class diagram for the decorator pattern [2].


Data Mesh Observability Pattern: real-time understanding of the state of a Data Mesh, how data moves within it, who is using it, and how they are using it.



## Software Design Patterns

[What Primitive Obsession Is and Why You Already Ruin Your Code Secretly](https://medium.com/codex/what-primitive-obsession-is-and-why-you-already-ruin-your-code-secretly-87120f8acaae)

[12 Front End Performance Patterns You Need to Know](https://medium.com/geekculture/12-front-end-performance-patterns-you-need-to-know-def550620464)

[10 Common Software Architectural Patterns in a nutshell](https://towardsdatascience.com/10-common-software-architectural-patterns-in-a-nutshell-a0b47a1e9013)


## Data Design Patterns

[Data Mesh Observability Pattern](https://towardsdatascience.com/data-mesh-observability-pattern-467438627572)


## Machine Learning Design Patterns

[Design Patterns for Machine Learning](https://towardsdatascience.com/design-patterns-for-machine-learning-410be845c0db)

[Understand Machine Learning Through 7 Software Design Patterns](https://betterprogramming.pub/machine-learning-through-7-design-patterns-35a8d5844cf6)

[Understand Machine Learning through More Design Patterns](https://towardsdatascience.com/understand-machine-learning-through-more-design-patterns-9c8430fd2ae8)


## MLOps Design Patterns

[Design Patterns in Machine Learning for MLOps](https://towardsdatascience.com/design-patterns-in-machine-learning-for-mlops-a3f63f745ce4)

[How (not) to do MLOps](https://towardsdatascience.com/how-not-to-do-mlops-96244a21c35e)

[Serving ML Models in Production: Common Patterns](https://www.kdnuggets.com/2021/10/serving-ml-models-production-common-patterns.html)



## References

[1] Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides. Design Patterns: Elements of Reusable Object-Oriented Software. ISBN: 978-0201633610. 1994.

[2]: [A Gentle Introduction to Decorators in Python](https://machinelearningmastery.com/a-gentle-introduction-to-decorators-in-python/)
