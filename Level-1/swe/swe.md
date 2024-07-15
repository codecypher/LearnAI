# Software Engineering


## Overview

Here are some key software engineering principles:

> Speed, Security, and Scalability, pick two. 

> The myth of small incremental improvements. 

> Premature optimization is the root of all evil. 


## Pareto Principle

The **Pareto Principle** states that a vital few inputs/causes (20% of inputs) directly influence a significant majority of the outputs/effects (80% of outputs) which is also known as the 80–20 rule. 

The Pareto Principle states that (for a wide variety of situations) about 80% of the outcome is caused by about 20% of the inputs.

Pareto principle has been found to be applicable in almost every aspect of our life:

- 20% of the code has 80% of the errors. 

- 80% of a piece of software can be written in 20% of the total allocated time. 

- the hardest 20% of the code takes 80% of the time.


## Rule of Six

The brain has three memory functions that work together to understand code [8]:

- **Long-term memory (LTM):** Stores information for long-term retrieval, such as keywords, syntax, and commonly used idioms and patterns.

- **Short-term memory (STM):** Stores new information for short-term retrieval (less than 30 seconds!), such as variable names and special values.

- **Working memory (WM):** Processes information from LTM and STM to draw conclusions and derive new knowledge.

STM and WM are small. Both can only store about 4 to 6 things at a time! Overload them and you’ve got a recipe for confusion.


> **Rule of six:** A line of code containing 6+ pieces of information should be simplified.

There are two strategies which can be used to break up code: SPLIT and MORF

SIMPLE: Split Into MultiPle LinEs

MORF: Move out and Rewrite as a Function



## SOLID Design Principles

SOLID is an acronym for five software development principles which are guidelines to follow when building software so that it is easier to scale and maintain, made popular by Robert C. Martin.

SOLID is a mnemonic acronym for the following five principles [1]:

1. Single-Responsibility Principle (SRP)
2. Open-Closed Principle (OCP)
3. Liskov Substitution Principle (LSP)
4. Interface Segregation Principle (ISP)
5. Dependency Inversion Principle (DIP)

Understanding and applying the five principle is the hallmark for good software engineering. Thus, it is any aspiring software engineer should be acquainted with them.


Cohesion refers to the interconnection between functions in a class. While coupling refers to the interdependence between classes [1].

The goal is to develop software with high cohesion and low coupling.

High cohesion means we want functions in a class to be performing tasks that are related to each other. 

Low coupling means classes should not depend too much on each other.


## Software Engineering Principles

- Key Principles of Functional Programming for Data Engineering
- Myths Programmers Believe about CPU Caches
- Hints to Avoid Over-complexity in Software Projects
- Understand Temporal Coupling in Code
- Understand Load vs Stress Tests

- Code Smell — When to Refactor
- Design for Services, Not Microservices

- Systems Design Crash Course for ML Engineers
- A Primer on Architectural Patterns
- Speed, Security, and Scalability: Choose 2

- An Introduction to Event-Driven Architecture
- Comparison of the Modes of Event-Driven Architecture




## The Single-responsibility Principle

The single-responsibility principle helps us achieve the goal of high cohesion and low coupling.

A class should only have a single purpose. 

If the functionality of the class needs to be changed there should only be one reason to do so.



## Dependency Inversion Principle

When designing a software system, the most important thing is the domain [6]. 

The domain is a fancy word for the problem you are trying to solve which are real-world rules that do not have anything to do with web server, web framework, or database.

Here we need to be aware of the concepts of high-level modules, low-level modules, and abstractions.

High-level modules are usually the code designed to solve the problem which contains the application's important policy decisions and business models — the application’s identity.

Low-level modules contain detailed implementations of individual mechanisms needed to realize the business model such as communication with a database.

An abstraction is used to hide background details or any unnecessary implementation of the data so that users only see the required information.

The domain should not depend on whether your data is stored in a SQL or NoSQL database. It also should not depend on the fact that the data is stored in a database or a FileSystem. 

The domain is only interested that the data is in persistent storage and is accessible when needed.

Higher level modules should not depend on lower level modules: both should depend on abstractions.

The database is merely an IO device that happens to provide some useful tools for sorting, querying, and reporting but those are ancillary to the system architecture - Robert C. Martin

To achieve all this, smart software engineers created patterns and techniques such as the repository pattern and dependency injection. 

**Dependency Injection**

Dependency injection is giving an object its instance variables. 

We provide the variables to the instance and make the instance communicate with its dependencies via abstractions (interfaces) rather than constructing them.

We can apply the dependency inversion principle here [6]. 

```py
def create_new_job(data, blob_storage_client: BlobStorageClient):
    file_url = blob_storage_client.save_data(data)
    return requests.post(
        url='some_url',
        json={'file_url': file_url}
        )
```

Rather than directly importing S3Client and initializing it inside the function, we define it as an argument. 

```py
class BlobStorageClient(ABC):
    @abstractmethod
        def save_data(self, data) -> str:
        raise NotImplementedError()
```

In fact, `BlobStorageClient` is an abstract class in Python with an abstract method called `save_data`. Now our S3Client will implement this interface.

Now, we can do the same thing to the database. Rather than domain models being dependent on the database, the database must be dependent on our domain models. 

We will invert the dependency using the repository pattern.

**Repository Pattern**

Repositories are classes or components that encapsulate the logic required to access data sources. Simply put, the repositories are responsible for communicating with the DB or whatever persistence storage solution you have [6]. 

Rather than communicating with the database directly, we abstract it through pattern repositories.

Most of the problems in software engineering can be solved by adding another layer of abstraction.
The repository pattern is nothing more than an abstraction for the database.

We create a class with all the operations we need to perform on the database for a particular domain entity. It is better to have a repository for every single entity which keeps them small and simple. 

Here is an example for the User entity in Python using SQLAlchemy [6]:

```py
class UserRepo:
    def __init__(self, session):
        self.session = session
    def get_by_username(self, username: str) -> User | None:
        return self.session.query(User).filter_by(
        username=username).first()
```

### Testing

The obvious benefits are testability, single responsibility, modularity, etc. 

Normally, when we test `create_new_job`, we do not want to actuallt call the S3 service. We could mock out both methods but mocking every single dependency creates a code smell on the tests.

With our patterns and principles implemented, we can leverage a different technique to deal with third parties while testing: We can inject fake implementations.

We can implement another `BlobStorageClient` as follows [6]:

```py
class TestBlobStorageClient(BlobStorageClient):
    is_called = False
    def save_data(self, data):
    self.is_called = True
```

When we test, we can inject this test client rather than rhe s3 client:

```py
def test_create_new_job():
    test_client = TestBlobStorageClient()
    create_new_job('some-data', test_client)
    
    assert test_client.is_called
```

### Apply the Dependency Inversion Principle

If we apply the DIP by taking some help from SQLAlchemy, here is the result [6]:

Thus, the ORM depends on our domain model which is a pure Python class without any external dependencies. 

When we call `start_mappers`, SQLAlchemy will attach some private attributes to the `User` class such as `__table__` so when we use it inside a query with SQLAlchemy session, it knows that this class represents the user table in the database.

All business logic related to the User entity will go inside the User class. 

The business logic that will use more than one entity will go inside Services or use cases depending on what you decide to use. 

The important part is that now we have inverted the dependency which means the database is now dependent on the domain models rather than the other way around.



## Inversion of Control vs Dependency Injection

IoC is a generic term meaning that rather than having the application call the methods in a framework, the framework calls implementations provided by the application.

DI is a form of IoC where implementations are passed into an object through constructors/setters/service lookups which the object will "depend" on in order to behave correctly.

IoC without using DI would be the Template pattern because the implementation can only be changed through sub-classing.

DI Frameworks are designed to make use of DI and can define interfaces (or Annotations in Java) to make it easy to pass in the implementations.

IoC Containers are DI frameworks that can work outside of the programming language. In some you can configure which implementations to use in metadata files (e.g. XML) which are less invasive. With some you can do IoC that would normally be impossible like inject an implementation at pointcuts.



Inversion of Control (IoC) means that objects do not create other objects on which they rely to do their work. Instead, they get the objects that they need from an outside source (e.g. a container or an xml config file).

Dependency Injection (DI) means that this is done without the object intervention usually by a framework component that passes constructor parameters and set properties.


### Advantages of IoC

Here are some advantages of Inversion Of Control (IoC) [9]:

- Inversion of control makes your code loosely coupled

- Inversion of control makes it easy for the programmer to write great unit tests




----------



## Software Engineering Myths

Incremental changes do not provide emergency exits for a failing system. It also does not recognize when a tool is not useful. 

The  small incremental improvements concept of agile methodology is flawed.


**There is No Incremental Change in New Apps**

All the objections hurled at an offending pull request are usually completely absent when an app first comes online. 

The seeds for failure are sown early on: poor decisions about application boundaries or data modeling can be practically impossible to fix down the line, so if you really want to play the “small incremental improvement” game, you really need to evaluate new applications very carefully before they come online. Amateur developers make things that work, senior developers make things that scale… and most management cannot tell the difference.

**Insufficient Feedback Loops**

Business and product folks are focused on features and finances, but your app will suffer if feedback is limited to only these talking points.

Developers, better than anyone else in an organization, can identify software inefficiencies and correct them. 

**Efficiency Wins**

In a global marketplace of ideas and solutions, we would do well to remember that _efficiency wins_.

The most efficient software with the fewest requirements in terms of hardware or maintainers will have a clear advantage. So listen carefully when the developers try to improve the efficiency of the code, even in cases when thousands of files get changed or when absolutely nothing changes for the customer.

**You Cannot Unboil a Frog**

This happens all the time: a company with “mostly functional” software attempts to transition from scrappy start-up to Respectable Profitability™. 

Endless man-hours are spent pumping blood through the legacy code while incremental changes are introduced with surgical slowness. Because the existing implementations are so far off from where they should be, it will take years of diligent development to escape the wilderness and arrive at the promised land. 

Instead of saving the project, new changes can introduce more and more debt because there is no incremental way to break out of the flawed paradigms on which it was built. Amidst this madness, the captain dictates that absolutely no man-hours are available for rewriting the system in a way that would avoid the icebergs and sinking ships altogether.

In practical terms, you cannot unboil a frog: you have to make a new one.

With software systems, it is less work to rebuild a new system than it is to perform CPR on the legacy system.  




## API vs Web Service

All web services are also APIs because they expose the data and/or functionality of an application, however not all APIs are web services. 




## References

[1]: [SOLID Coding in Python](https://towardsdatascience.com/solid-coding-in-python-1281392a6a94)

[2]: [The S.O.L.I.D Principles in Pictures()](https://medium.com/backticks-tildes/the-s-o-l-i-d-principles-in-pictures-b34ce2f1e898)

[3]: [The Single Responsibility Principle Explained in Python](https://betterprogramming.pub/the-single-responsibility-principle-explained-in-python-622e2d996d86)

[4]: [Code Smell — When to Refactor](https://betterprogramming.pub/code-smell-when-to-refactor-e18f1dca2f01)

[5]: [Design for Services; Not Microservices](https://betterprogramming.pub/design-for-services-not-microservices-e339883946d7)

[6]: [The Database Is Not the Most Important Part](https://betterprogramming.pub/the-database-is-not-the-most-important-part-b87d8af01959)

[7]: [The Myth of Small Incremental Improvements](https://betterprogramming.pub/the-myth-of-small-incremental-improvements-fd0bfd5e1977)

[8]: [Want Cleaner Code? Use the Rule of Six](https://betterprogramming.pub/want-cleaner-code-use-the-rule-of-six-c21635ee2185)

[9]: [Inversion of Control vs Dependency Injection](https://stackoverflow.com/questions/6550700/inversion-of-control-vs-dependency-injection)


[CUPID for better coding](https://medium.com/codex/cupid-for-better-coding-4c6eb401c8f5)

[Systems Design Crash Course for ML Engineers](https://towardsdatascience.com/systems-design-crash-course-for-ml-engineers-aafae1cf1890)

[A Primer on Architectural Patterns](https://towardsdatascience.com/a-primer-on-architectural-patterns-fd1b22a4389d)

[Speed, Security, and Scalability: Pick Only 2!](https://betterprogramming.pub/speed-security-and-scalability-pick-only-2-5e61c637b08e)


[3 Key Principles of Functional Programming for Data Engineering](https://towardsdatascience.com/3-key-principles-of-functional-programming-for-data-engineering-67d2b82c7483)

[Myths Programmers Believe about CPU Caches](https://software.rajivprab.com/2018/04/29/myths-programmers-believe-about-cpu-caches/)

[10 Hints to Avoid Over-complexity in Software Projects](https://betterprogramming.pub/10-hints-to-avoid-over-complexity-in-software-projects-51a25bf51853)

[Understand Temporal Coupling in Code](https://betterprogramming.pub/temporal-coupling-in-code-e74899f7a48f)

[Understanding Load vs Stress Tests](https://betterprogramming.pub/load-vs-stress-tests-ee49ae110b1d)


[An Introduction to Event-Driven Architecture](https://aws.plainenglish.io/event-driven-architecture-2436055f64b1?gi=62c6bfcf207)

[The Comparison of the Modes of Event-Driven Architecture](https://medium.com/geekculture/the-comparison-of-the-modes-of-event-driven-architecture-1742711d79bb)
