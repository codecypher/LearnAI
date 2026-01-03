# Python Tips

Here are some tips and tricks for using Python.

## Vectorization

**Vectorization** is the technique of implementing (NumPy) array operations on a dataset.

Vectorization is a very fast alternative to loops in Python. 

In the background, NumPy applies the operations to all the elements of an array or series at once rather than one row at a time using for loop. 

### Finding the Sum of numbers

Using Loops

```py
    import time 
    start = time.time()

     
    # iterative sum
    total = 0
    # iterating through 1.5 Million numbers
    for item in range(0, 1500000):
        total = total + item


    print('sum is:' + str(total))
    end = time.time()

    print(end - start)

    # 1124999250000
# 0.14 Seconds
```

Using Vectorization

```py
    import numpy as np

    start = time.time()

    # vectorized sum - using numpy for vectorization
    # np.arange create the sequence of numbers from 0 to 1499999
    print(np.sum(np.arange(1500000)))

    end = time.time()

    print(end - start)


    # 1124999250000
    # 0.008 Seconds
```

### Mathematical Operations (on DataFrame)

Create the DataFrame

The DataFrame is tabular data in the form of rows and columns.

We create a pandas DataFrame having 5 Million rows and 4 columns filled with random values between 0 and 50.

```py
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(np.random.randint(0, 50, size=(5000000, 4)), columns=('a','b','c','d'))
    df.shape
    # (5000000, 5)
    df.head()
```

We create a new column ‘ratio’ to find the ratio of the column ‘d’ and ‘c’.

Using Loops

```py
    import time 
    start = time.time()

    # Iterating through DataFrame using iterrows
    for idx, row in df.iterrows():
        # creating a new column 
        df.at[idx,'ratio'] = 100 * (row["d"] / row["c"])  
    end = time.time()
    print(end - start)
    # 109 Seconds
```

Using Vectorization

```py
    start = time.time()
    df["ratio"] = 100 * (df["d"] / df["c"])

    end = time.time()
    print(end - start)
# 0.12 seconds
```

### If-else Statements (on DataFrame)

Suppose we want to create a new column ‘e’ based on some conditions on the exiting column ‘a’.

Using Loops

```py
    import time 
    start = time.time()

    # Iterating through DataFrame using iterrows
    for idx, row in df.iterrows():
        if row.a == 0:
            df.at[idx,'e'] = row.d    
        elif (row.a <= 25) & (row.a > 0):
            df.at[idx,'e'] = (row.b)-(row.c)    
        else:
            df.at[idx,'e'] = row.b + row.c

    end = time.time()

    print(end - start)
    # Time taken: 177 seconds
```

Using Vectorization
 
```py
    start = time.time()
    df['e'] = df['b'] + df['c']
    df.loc[df['a'] <= 25, 'e'] = df['b'] -df['c']
    df.loc[df['a']==0, 'e'] = df['d']end = time.time()
    print(end - start)
    # 0.28007707595825195 sec
```

### Solving Deep Learning Networks

Create  the Data

```py
    import numpy as np
    # setting initial values of m 
    m = np.random.rand(1,5)

    # input values for 5 million rows
    x = np.random.rand(5000000,5)
```


Using Loops

```py
    import numpy as np
    m = np.random.rand(1,5)
    x = np.random.rand(5000000,5)

    total = 0
    tic = time.process_time()

    for i in range(0,5000000):
        total = 0
        for j in range(0,5):
            total = total + x[i][j]*m[0][j] 
            
        zer[i] = total 

    toc = time.process_time()
    print ("Computation time = " + str((toc - tic)) + "seconds")
    # Computation time = 28.228 seconds
```

Using Vectorization

Figure: Dot Product of 2 matrices

```py
    tic = time.process_time()

    # dot product 
    np.dot(x,m.T) 

    toc = time.process_time()
    print ("Computation time = " + str((toc - tic)) + "seconds")
    # Computation time = 0.107 seconds
```

The np.dot implements Vectorized matrix multiplication in the backend which is much faster compared to loops in python.


## Iterables

An _iterable_ is any Python object that is capable of returning its members one at a time, permitting it to be iterated over in a loop.

There are _sequential_ iterables that arrange items in a specific order, such as lists, tuples, string and dictionaries.

There are _non-sequential_ collections that are iterable. For example, a set is an iterable, despite lacking any specific order.

Iterables are fundamental to Python and manuy other programming language, so knowing how to efficiently use them will have an impact on the quality of your code.

In general, iterables can be processed using a for-loop that allows for successively handling each item that is part of the iterable.

You may find that you often create programming loops that are similar to one another. Therefore, there are plenty of great built-in Python functions that can help to process iterables without reinventing the wheel.

Python iterables are fast, memory-efficient, and when used properly make your code more concise and readable.

> Python’s built-in functions are written in C, so they are very fast and efficient. 


The functions included in the article [4] are:

- Function 1: all
- Function 2: any
- Funciton 3: enumerate
- Function 4: filter
- Function 5: map
- Function 6: min
- Function 7: max
- Function 8: reversed
- Function 9: sum
- Function 10: zip
- collections.Counter



## Pythonic Loops

### Iterate in Parallel Over Multiple Iterables with zip

The `zip()` function is used to iterate over multiple iterables in parallel by pairing corresponding elements of different iterables together [12]. 

Suppose we need to loop through both names and scores list:

```py
names = ["Alice", "Bob", "Charlie"]
scores = [95, 89, 78]

for i in range(len(names)):
    print(f"{names[i]} scored {scores[i]} points.")
```

Here is a more readable loop with the zip() function:

```py
names = ["Alice", "Bob", "Charlie"]
scores = [95, 89, 78]

for name, score in zip(names, scores):
    print(f"{name} scored {score} points.")
```

The Pythonic version using zip() is more elegant and avoids the need for manual indexing—making the code cleaner and more readable.
    

## List and Dictionary Comprehension

In Python, list comprehensions and dictionary comprehensions are concise one-liners to create lists and dictionaries which can also include conditional statements to filter items based on certain conditions [12].

```py
    numbers = [1, 2, 3, 4, 5]
    squared_numbers = [num ** 2 for num in numbers]
    print(squared_numbers)
```

Here, the list comprehension creates a new list containing the squares of each number in the numbers list.

### List Comprehension with Conditional Filtering

We can add filtering conditions within the list comprehension expression [12]:

```py
    numbers = [1, 2, 3, 4, 5]
    odd_numbers = [num for num in numbers if num % 2 != 0]
    print(odd_numbers)
```

Suppose we have a fruits list and we want create a dictionary with `fruit:len(fruit)` key-value pairs.

We can do this with a for loop:

```py
    fruits = ["apple", "banana", "cherry", "date"]
    fruit_lengths = {}

    for fruit in fruits:
        fruit_lengths[fruit] = len(fruit)

    print(fruit_lengths)
```

We can also write the dictionary comprehension equivalent:

```py
    fruits = ["apple", "banana", "cherry", "date"]
    fruit_lengths = {fruit: len(fruit) for fruit in fruits}
    print(fruit_lengths)
```

The dictionary comprehension creates a dictionary where keys are the fruits and values are the lengths of the fruit names.

### Dictionary Comprehension with Conditional Filtering

Let us modify our dictionary comprehension expression to include a condition [12]:

```py
    fruits = ["apple", "banana", "cherry", "date"]
    long_fruit_names = {fruit: len(fruit) for fruit in fruits if len(fruit) > 5}
```

The dictionary comprehension creates a dictionary with fruit names as keys and their lengths as values, but only for fruits with names longer than 5 characters.


## Python Tips

### Use Context Managers for Effective Resource Handling

Context managers in Python help you manage resources efficiently [12]. 

The simplest and the most common example of context managers is in file handling.

Now consider the following version using the `with` statement that supports `open` function which is a context manager:

```py
filename = 'somefile.txt'
with open(filename, 'w') as file:
    file.write('Something')

print(file.closed)
```

We use the `with` statement to create a context in which the file is opened whicv ensures that the file is properly closed when the execution exits the with block - even if an exception is raised during the operation.

Now it is not necessary to implement exception handling. 

Here are a few more examples using context managers [13]:

- Handling Database Connections 
- Managing Python Subprocesses
- High-Precision Floating-Point Arithmetic


### Use Generators for Memory-Efficient Processing

Generators provide an elegant way to work with large datasets or infinite sequences to improve code efficiency and reducing memory consumption [12]. 

_Generators_ are functions that use the `yield` keyword to return values one at a time, preserving their internal state between invocations [12]. 

Unlike regular functions that compute all values at once and return a complete list, generators compute and yield values on-the-fly as they are requested which makes them suitable for processing large sequences.

Avoid Returning Lists From Functions

It is common to write functions that generate sequences such as a list of values. But we can rewrite them as generator functions. 

Generators use lazy evaluation which means they yield elements of the sequence on demand rather than computing all the values ahead of time.

- Using generators can be more efficient for large input sizes. 

- We can chain generators together to create efficient data processing pipelines.


### Cache Expensive Function Calls

Caching can improve performance by storing the results of expensive function calls and reusing them when the function is called again with the same inputs [16]. 

Suppose we are coding k-means clustering algorithm from scratch and want to cache the Euclidean distances computed. 

We can cache function calls with the `@cache` decorator:

```py
from functools import cache
from typing import Tuple
import numpy as np

@cache
def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def assign_clusters(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    clusters = np.zeros(data.shape[0])
    for i, point in enumerate(data):
        distances = [euclidean_distance(tuple(point), tuple(centroid)) for centroid in centroids]
        clusters[i] = np.argmin(distances)
    return clusters
```

Here is a sample function call:

```py
data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [8.0, 9.0], [9.0, 10.0]])
centroids = np.array([[2.0, 3.0], [8.0, 9.0]])

print(assign_clusters(data, centroids))

# Output >>> [0. 0. 0. 1. 1.]
```

Python provides built-in support for caching through the functools module: the  decorators `@cache` and `@lru_cache` [17]. 

Here are some reasons why caching function calls can improve performance [17]:

- Performance improvement: When a function is called with the same arguments multiple times, caching the result can eliminate redundant computations. Instead of recalculating the result every time, the cached value can be returned, leading to faster execution.

- Reduction of resource usage: Some function calls may be computationally intensive or require significant resources (such as database queries or network requests). Caching the results reduces the need to repeat these operations.

- Improved responsiveness: In applications where responsiveness is crucial, such as web servers or GUI applications, caching can help reduce latency by avoiding repeated calculations or I/O operations.

The `@lru_cache` decorator is similar to `@cache` but allows you to specify the maximum size of the cache. Once the cache reaches this size, the least recently used items are discarded which can used to limit memory usage [17]. 

We can use the `timeit` function from the `timeit` module to compare the execution times. 


### Leverage Collection Classes

There are two less know but useful collection classes [12]:

#### More Readable Tuples with NamedTuple

In Python, a namedtuple in the collections module is a subclass of the built-in tuple class. But it provides named fields. Which makes it more readable and self-documenting than regular tuples.

#### Use Counter to Simplify Counting

The `Counter` is a class in the collections module that is designed for counting the frequency of elements in an iterable such as a list or a string). It returns a Counter object with {element:count} key-value pairs.

Suppose we are counting character frequencies in a long string.

We can achieve the same task using the Counter  class using the syntax Counter(iterable):

```py
from collections import Counter

word = "incomprehensibilities"

# Count character frequencies using Counter
char_counts = Counter(word)

print(char_counts)

# Find the most common character
most_common = char_counts.most_common(1)

print(f"Most Common Character: '{most_common[0][0]}' (appears {most_common[0][1]} times)")
```

Thus, Counter provides a much simpler way to count character frequencies without the need for manual iteration and dictionary management.


### Write Shorter Conditionals using Dictionaries

Dictionaries are a concise alternative to the classic If-Else statement and the new Match-Case statement [5]. 

If we were to use an If-Else:

```py
    def month(idx):
        if idx == 1:
            return "January"
        elif idx == 2:
            return "February"
        # ...
        else:
        return "not a month"
```

Another approach would be to use the recently released Match-Case statement:

```py
    def month(idx):
        match idx:
            case 1:
                return "January"
            case 2:
                return "February"
            # ...
            case _:
        return "Not a month"
```

There is nothing inherently wrong with either of these approaches. In terms of code clarity, there is more to be gained using dictionaries.

Dictionary Approach

The first step is to return a dictionary that uses the index of the months as keys, and their corresponding names as values.

Next, we use the `.get()` method to obtain the name of the month that actually belongs to the number that we provided as function argument.

The great thing about this method is that we can also specify a default return value for when the requested key is not part of the dictionary (“not a month”).

```py
    def month(idx):
        return {
            1: "January",
            2: "February",
            # ...
            }.get(idx, "not a month")
```

While our dictionary conditional has the least-best performance, it is important to understand where this difference comes from.

Since we define our dictionary _within_ the `month()` function, it has to be constructed once for every function call which is inefficient. 

If we define the dictionary _outside_ the function and rerun the experiment, we achieve the fastest runtime.

```py
    dt = {
        1: "January",
        2: "February",
        # ...
        11: "November",
        12: "December"
    }

    def month(idx):
        return dt.get(idx, "not a month")
```


### Display Pandas DataFrame in table style

```py
    # importing the modules
    from tabulate import tabulate
    import pandas as pd
      
    # creating a DataFrame
    dict = {'Name':['Martha', 'Tim', 'Rob', 'Georgia'],
            'Maths':[87, 91, 97, 95],
            'Science':[83, 99, 84, 76]}
    df = pd.DataFrame(dict)
      
    # displaying the DataFrame
    print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
```


## Regex Library

Here are some common re libraries functions [6]:

- re.search(): This function scans a string for the first location where the regular expression pattern matches.

- re.match(): This function is used to find a regular expression pattern that results in a match at the beginning of the string.

NOTE: Search will loop through the string to find the first appearance of the pattern while match only checks the beginning of the string. For example, the match will return none while the search will return the position.

- re.split(): This function is used to split a string based on a regular expression pattern.

- re.findall(): This function returns a list of character(s) that match the RegEx pattern.

- re.sub(): This function is used to replace any character(s) that match the RegEx pattern with another character(s).



## Utility Classes

Here are some examples for starting a library of reusable utility classes. 

### Enumeration

We can create an `Enumeration` class to hold related members of the same concept such as compass directions (north, south, east, and west) or seasons [3]. 

In the Python standard library, the `enum` module provides the core features needed to create an enumeration class.

```py
    from enum import Enum
    
    class Season(Enum):
        SPRING = 1
        SUMMER = 2
        FALL = 3
        WINTER = 4
```

```py
    spring = Season.SPRING
    spring.name
    spring.value
    
    fetched_season_value = 2
    matched_season = Season(fetched_season_value)
    matched_season
    # <Season.SUMMER: 2>
    
    list(Season)
    
    [x.name for x in Season]
```

### Data Classes

We can create a class to hold data using the `dataclass` decorator [3]. 

```py
    from dataclasses import dataclass
    
    @dataclass
    class Student:
        name: str
        gender: str
```

```py
    student = Student("John", "M")
    student.name
    student.gender
    
    repr(student)   # __repr__
    # "Student(name='John', gender='M')"
    
    print(student)  # __str__
    Student(name='John', gender='M')
```


## Better Error Handling

Clean up your code by creating your own custom exceptions [9] and [19].

### Create custom exceptions

```py
class InvalidCredentialsException(Exception):
    def __str__(self):
        return f"Invalid email-password combination"

class UserNotFoundException(Exception):
    email:str
    def __init__(self, email:str, *args, **kwargs):
        self.email = email
        super().__init__(args, kwargsp)

    def __str__(self):
        return f"Could not find an account associated with email '{self.email}'"
```

### Cleanup the login function

In addition to looking much nicer the function is more clean and pure; it is only responsible for logging in, so it does not need to know anything about redirecting and popups. 

This kind of logic should be limited to a few places in your project and should not be littered throughout the application. 

```py
def login(email: str, password: str) -> None:
    """ Logs in a user """

    if (not userexists(email=email)):
        raise UserNotFoundException(email=email)

    if (not credentials_valid(email=email, password=password)):
        raise InvalidCredentialsException()
```

In `main.py` file we can now call the login function:

```py
try:
    login_clean(email=my_email, password=my_pass)
    show_popup("logged in!")
    redirect(target_page='my_account')
except UserNotFoundException as e:
    show_popup(f"Unable to log in: {e}")
    redirect(target_page='/register')
except InvalidCredentialsException as e:
    show_popup(f"Unable to log in: {e}")
except Exception as e:
    show_popup("Something went wrong: try again later")
```

### Be Specific

Try to be specific and only catch generic exceptions for debugging. In some cases, it can be useful to add a `finally` block [10]. 

```py
try:
    result = do_something()
except FileNotFoundError:
    log_error("File not found.")
except ValueError:
    log_error("Invalid input.")
except Exception as e:
    # catch any other unexpected exceptions and log them for debugging
    log_error(f"An unexpected error occurred: {str(e)}")
finally:
    # this runs always, whether an exception occurred or not
    close_resources()
```

You can also create custom exception classes to provide more specific error messages and help distinguish different types of errors [10].

```py
class CustomError(Exception):
    def __init__(self, message, *args):
        self.message = message
        super(CustomError, self).__init__(message, *args)

def some_function():
    if some_condition:
        raise CustomError("This is a custom error message.")

try:
    some_function()
except CustomError as ce:
log_error(str(ce))
```


## Python Hacks

Here are some practical Python hacks that can simplify your daily life [14], [15]. 

- Automate File Organization
- Schedule Tasks with Python
- Backing Up a Directory
- Manage and Monitor System Resources
- Renaming Multiple Files
- Converting DOCX to PDF
- Backing Up a Directory



## Best Practices for ML Code

Unlike traditional software engineering projects, ML codebases tend to lag behind in code quality due to their complex and evolving nature, leading to increased technical debt and difficulties in collaboration [10].

The following section will show common examples from ML codebases and explain how to handle those properly.

### Handling Multiple Return Values

Adding parameters to return statements in functions can make Python code more difficult to maintain. Every time you change something, you need to update all the calling code. 

Python provides an elegant solution for handling multiple return values: `namedtuple` from the Python collections module [10].

```py
from collections import namedtuple

def calculate_statistics(numbers):
    total = sum(numbers)
    mean = total / len(numbers)
    maximum = max(numbers)
    Statistics = namedtuple('Statistics', ['sum', 'mean', 'maximum'])
    return Statistics(sum=total, mean=mean, maximum=maximum)

# example
data = [12, 5, 8, 14, 10]
result = calculate_statistics(data)

print("Sum:", result.sum)         # output: Sum: 49
print("Mean:", result.mean)       # output: Mean: 9.8
print("Maximum:", result.maximum) # output: Maximum: 14
```

### Large Conditional Logic Trees

The complexity of (business) logic can quickly escalate [10]. 

One solution is to refactor the processing for the if-else tree using a dictionary:

```py
def process_input(x):
    mapping = {
        'A': 'Apple',
        'B': 'Banana',
        'C': 'Cherry',
        'D': 'Date',
        # ... and so on for many more cases
    }
    return mapping.get(x, 'Unknown')
```

It is best to define the mapping outside of the function in a separate settings or configuration file, but this approach can still become quite a large dictionary if you are not careful.

Another way to handle this is by using polymorphism.

- Create a base class with a common interface, and then implement subclasses for each specific case. 

- Each subclass will handle its unique logic.


Beware of code repetition. If you have multiple models and you want to use their outputs to get to a final score, avoid nested conditional logic trees by using multiple small functions to get to the final score.

Suppose your final grade for math will be calculated out of your attendance percentage and your exam percentage score [10].

```py
from typing import List

def map_score(score, score_ranges: List[float]) -> int:
    for i, threshold in enumerate(score_ranges):
        if score < threshold:
            return 2*i
    return 2*(i+1)

def final_score(parameter_scores: List[float], base_score: int = 4) -> int:
    parameter_range = [0.25, 0.5, 0.75]
    scores = [map_score(parameter_score, parameter_range)/len(parameter_scores) for parameter_score in parameter_scores]
    return sum(scores) + base_score
  
# example
attendance = 0.6
exam = 0.85
result = final_score([attendance, exam])
print("Final Score:", result)
```

### Annotations

Code annotations are more than mere comments: they are a standardized way to highlight areas of code that require attention or improvement [11].

Here are the typical code annotations:

TODO: Indicates tasks, improvements, or features that need to be implemented. It’s one of the most common annotations, probably the most frequently used.

- NOTE: Used to highlight an important piece of information about a module, class, method or function (if located in a docstring) or about the code fragment before which it’s placed (if used as an inline comment). It can be something related to the implementation, usage, or context that developers should be aware of.

- BUG: Marks a bug within the code. It should be accompanied by a description of the bug and other significant information. This can be a specific bug that you know of, or an indication of a code fragment that has an unknown bug.

- FIXME: Marks an issue in the code that needs to be fixed. It’s different from BUG, however. BUG indicates an actual mistake in the code while FIXME rather indicates problems that aren’t mere bugs but could be related to, e.g., lousy or inefficient implementation, incorrect implementation of the business logic, unclear code.

- REVIEW: Signals that a reviewer should pay attention to a particular piece of code. Therefore, the context should be clearly explained.

In addition to these, from time to time we can use two custom code annotations:

- THINK ABOUT THIS: Marks a code fragment or an idea (if located in a docstring) that requires in-depth thinking, for whatever reasons. Most often, this won’t have anything to do with the technical side of the code but with the business logic.

- RECONSIDER: Signals a code fragment to be reconsidered. You must provide reasons and/or ideas, since without explanation, this annotation could be more confusing than helpful.


```py
"""This is an example annotation module.

TODO: Install the extension.
RECONSIDER: Should the different annotations be formatted
    the same way?
"""
def foo(x, y, z):
    """Create a tuple of three elements.
    
    Examples:
    >>> foo(1, 2, 3)
    (1, 2, 3)
    >>> foo('1', '2', 3)
    ('1', '2', 3)

    TODO: Use *args to allow for more arguments.
    RECONSIDER: Is this function really important?
    """
    return x, y, z

def bar(x: int) -> tuple[int, float, str]:
    """Create a three-type int tuple.
    
    TODO: Explain what "a three-type int tuple" is.
    TODO: Add doctests.
    """
    # FIXME: Remove the direct call to tuple:
    return tuple(x, float(x), str(x))

def baz(x: float) -> tuple[int, float, str]:
    """Create a three-type int tuple from float.
    
    TODO: Explain what "a three-type int tuple" is.
    TODO: Add doctests.
    """
    # BUG: x is float, so it should be converted to
    #    a float (see the first position of the tuple)
    # FIXME: Remove the direct call to tuple:
    return tuple(x, x, str(x))
```


## Python Debug

While mistakes are unavoidable, getting better at debugging can save a lot of time and frustration  

### Avoid Mutable Default Argument Pitfalls
 
One classic Python gotcha involves default arguments in functions, especially mutable ones such as lists or dictionaries [18]. 

```py
def add_item(item, my_list=[]):
    my_list.append(item)
    return my_list

print(add_item("apple"))  # ['apple']
print(add_item("banana")) # ['apple', 'banana']
```

You might have expected each call to start with an empty list, but instead the list “remembers” past calls. 

Python evaluates default arguments only once when the function is defined, not each time it is called which means the same list is reused. 

A safer pattern is to use None and then create a new list inside:

```py
def add_item(item, my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(item)
    return my_list
```

Now each call without an explicit list gets a fresh list. 


## References

[1]: [Binary Image Classification in PyTorch](https://towardsdatascience.com/binary-image-classification-in-pytorch-5adf64f8c781)

[2]: [Learn Python By Example: 10 Python One-Liners That Will Help You Save Time](https://medium.com/@alains/learn-python-by-example-10-python-one-liners-that-will-help-you-save-time-ccc4cabb9c68)

[3]: [3 Alternatives for Regular Custom Classes in Python]()

[4]: [Iterator Functions](https://itnext.io/iterator-functions-33265a99e5d1)

[5]: [Write Shorter Conditionals (Using Dictionaries)](https://itnext.io/write-shorter-conditionals-using-dictionaries-python-snippets-4-f92c8ce5eb7)


[6]: [Understanding Regular Expression for Natural Language Processing](https://heartbeat.comet.ml/understanding-regular-expression-for-natural-language-processing-ce9c4e272a29)

[7]: [Regular Expressions Clearly Explained with Examples](https://towardsdatascience.com/regular-expressions-clearly-explained-with-examples-822d76b037b4)

[8]: [Regular Expression (RegEx) in Python: The Basics](https://pub.towardsai.net/regular-expression-regex-in-python-the-basics-b8f2cd041bdb)

[9]: [Why and how custom exceptions lead to cleaner, better code](https://towardsdatascience.com/why-and-how-custom-exceptions-lead-to-cleaner-better-code-2382216829fd)

[10]: [Software Engineering Best Practices for Writing Maintainable ML Code](https://towardsdatascience.com/software-engineering-best-practices-for-writing-maintainable-ml-code-717934bd5590)

[11]: [Enhancing Readability of Python Code via Annotations](https://towardsdatascience.com/enhancing-readability-of-python-code-via-annotations-09ce4c9b3729)

[12]: [How To Write Efficient Python Code: A Tutorial for Beginners](https://www.kdnuggets.com/how-to-write-efficient-python-code-a-tutorial-for-beginners)

[13]: [3 Interesting Uses of Python’s Context Managers](https://www.kdnuggets.com/3-interesting-uses-of-python-context-managers)

[14]: [Python Hacks for Everyday Life: Streamline Your Tasks with Simple Scripts](https://medium.com/codex/python-hacks-for-everyday-life-streamline-your-tasks-with-simple-scripts-1b54751180dc)

[15]: [Automate the Boring Stuff: Practical Python Solutions for Daily Tasks](https://medium.com/codex/automate-the-boring-stuff-practical-python-solutions-for-daily-tasks-07d81c65ab27)

[16]: [5 Python Tips for Data Efficiency and Speed](https://www.kdnuggets.com/5-python-tips-for-data-efficiency-and-speed)

[17]: [How To Speed Up Python Code with Caching](https://www.kdnuggets.com/how-to-speed-up-python-code-with-caching)

[18]: [7 Python Debugging Techniques Every Beginner Should Know](https://www.kdnuggets.com/7-python-debugging-techniques-every-beginner-should-know)

[19]: [5 Error Handling Patterns in Python (Beyond Try-Except)](https://www.kdnuggets.com/5-error-handling-patterns-in-python-beyond-try-except)


[Python: Pretty Print a Dict (Dictionary) – 4 Ways](https://datagy.io/python-pretty-print-dictionary/)

[Name of a Python function](https://medium.com/@vadimpushtaev/name-of-python-function-e6d650806c4)

[Python's assert: Debug and Test Your Code Like a Pro](https://realpython.com/python-assert-statement/)

[6 Must-Know Methods in Python’s Random Module](https://medium.com/geekculture/6-must-know-methods-in-pythons-random-module-338263b5f927)
