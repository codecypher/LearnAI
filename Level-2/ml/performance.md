# Performance

Here are some resources to improve Python performance, also see [Memory Usage](,/memory_usage.md)


## Why Python is so slow

Slowness vs waiting

1. CPU-tasks
2. I/O-tasks


Examples of I/O tasks are writing a file, requesting some data from an API, printing a page; they involve waiting.

Although I/O can cause a program to take more time to execute, this is not Python’s fault. Python is just waiting for a response; a faster language cannot wait faster.

Thus, I/O slowness is not what we are trying to solve here.

Here, we figure out why Python executes CPU-tasks more slowly than other languages.

- Compiled vs Interpreted
- Garbage collection and memory management
- Single-thread vs multi-threaded


### Single-thread vs multi-threaded

Python is single-threaded on a single CPU by design.

The mechanism that makes sure of this is called the GIL: the Global Interpreter Lock. The GIL makes sure that the interpreter executes only one thread at any given time.

The problem the GIL solves is the way Python uses reference counting for memory management. A variable’s reference count needs to be protected from situations where two threads simultaneously increase or decrease the count which can cause all kinds of weird bugs to to memory leaks (when an object is no longer necessary but is not removed) or incorrect release of the memory (a variable gets removed from the memory while other variables still need it).

In short: Because of the way garbage collection is designed, Python has to implements a GIL to ensure it runs on a single thread. There are ways to circumvent the GIL though, read this article, to thread or multiprocess your code and speed it up significanly.

### How to speed things up

Thus, we can conclude that the main problems for execution speed are:

- **Interpretation:** compilation and interpretation occurs during runtime due to the dynamic typing of variables. For the same reason we have to create a new PyObject, pick an address in memory and allocate enough memory every time we create or “overwrite” a “variable” we create a new PyObject for which memory is allocated.

- **Single thread:** The way garbage-collection is designed forces a GIL: limiting all executing to a single thread on a single CPU

How do we remedy these problems:

- Use built-in C-modules in Python such as `range()`.

- I/O-tasks release the GIL so they can be threaded; you can wait for many tasks to finish simultaneously.

- Run CPU-tasks in parallel by multiprocessing.

- Create and import your own C-module into Python; you can extend Python with pieces of compiled C-code that are 100x faster than Python.

- Write Python-like code that Cython compiles to C and then neatly packages into a Python package which offers the readability and easy syntax of Python with the speed of C.


## Improve Python Performance

The rule of thumb is that a computer is a sum of its parts or weakest link. In addition, the basic performance equation reminds us that there are always tradeoffs in hardware/software performance.

Thus, there is no silver bullet hardware or software technology that will magically improve computer performance.

### Scalability 

_Scalability_ is the property of a system to handle a growing amount of work [5].

- Horizontal scaling adds more machines to increase capacity without disruptions for applications that need load distribution. 

- Vertical scaling upgrades the hardware of existing machines when more processing power is required but cannot be easily distributed.

Hardware upgrades (vertical scaling) usually provide only marginal improvement in performance. 

We can achieve as much as 30-100x performance improvement using software libraries and code refactoring to improve parallelization (horizontal scaling) [2] [3].

> There is no silver bullet to improve performance.

In general, improving computer performance is a cumulative process of several or many different approaches primarily software related.

> NOTE: Many of the software libraries to improve pandas performance also enhance numpy performance as well.


----------


## Lightning Fast Iteration

Here are some tips to improve Python loop/iteration performance [2].

### Zip

```py
    z = list([1, 2, 3, 4, 5, 6])
    z2 = list([1, 2, 3, 4, 5, 6])

    # create a new zip iterator class
    bothlsts = zip(z, z2)

    for i, c in bothlsts:
        print(i + c)

    # call the zip() class directly
    for i, c in zip(z, z2):
        print(i + c)
```

### Itertools

For more complex looping patterns, the Python `itertools` library provides highly efficient looping constructs [6]. 

```py
    import itertools as its

    def fizz_buzz(n):
        fizzes = its.cycle([""] * 2 + ["Fizz"])
        buzzes = its.cycle([""] * 4 + ["Buzz"])
        fizzes_buzzes = (fizz + buzz for fizz, buzz in zip(fizzes, buzzes))
        result = (word or n for word, n in zip(fizzes_buzzes, its.count(1)))

        for i in its.islice(result, 100):
            print(i)
```

`itertools` provides efficient tools for handling complex looping scenarios which makes code faster and more readable [6]. 

```py

import itertools

temperatures = [15, 22, 30]
precipitation = [0, 5, 10]

# Generate all combinations of temperatures and precipitation
combinations = list(itertools.product(temperatures, precipitation))

print("Combinations of temperatures and precipitation:", combinations)
```


### Stop Nesting

Avoid writing nested for loops.

If you need an index to call, you can use the `enumerate()` on your iterator in a similar fashion to how we used `zip()` above.

### Do not Zip dicts!

There is no need to use zip() with dictionaries.

```py
    dct = {"A" : [5, 6, 7, 8], "B" : [5, 6, 7, 9]}

    for i in dct:
        print(i)
    # A, B

    for i in dct:
        print(dct[i])
    # [5, 6, 7, 8]

    # only work with the values
    for i in dct.values():
        print(i)
```

### Filter

The built-in Python `filter()` method can be used to eliminate portions of an iterable with minimal performance cost.

```py
    people = [{"name": "John", "id": 1}, {"name": "Mike", "id": 4},
              {"name": "Sandra", "id": 2}, {"name": "Jennifer", "id": 3}]

    # filter out some of the unwanted values prior to looping
    for person in filter(lambda i: i["id"] % 2 == 0, people):
        print(person)

    # {'name': 'Mike', 'id': 4}
    # {'name': 'Sandra', 'id': 2}
```




## Optimize Python Code

Tips to improve performance of Python code [3].

### Leverage NumPy for numerical operations

When handling large datasets with numerical data, the built-in Python loops and arithmetic operations can become slow [6]. 

NumPy provides optimized arrays and functions for numerical operations and high-performance data processing.

### Use built-in functions rather than coding them from scratch

Some built-in functions in Python like map(), sum(), max(), etc. are implemented in C so they are not interpreted during the execution which saves a lot of time.

For example, if you want to convert a string into a list you can do that using the `map()` function  instead of appending the contents of the strings into a list manually.

```py
    string = ‘Australia’
    U = map(str, s)
    print(list(string))
    # [‘A’, ‘u’, ‘s’, ‘t’, ‘r’, ‘a’, ‘l’, ‘i’, ‘a’]
```

Also, the use of f-strings while printing variables in a string instead of the traditional ‘+’ operator is also very useful in this case.

### Focus on Memory Consumption During Code Execution

Reducing the memory footprint in your code definitely make your code more optimized.

Check if unwanted memory consumption is occuring.

Example: str concatenation using + operator will generate a new string each time which will cause unwanted memory consumption. Instead of using this method to concatenate strings, we can use the function `join()` after taking all the strings in a list.

### Using C libraries/PyPy to Get Performance Gain

If there is a C library that can do the job then it is probably better to use that to save time when the code is interpreted.

The best way to make us of C libraries in Python  is to use the ctype library in python, but there is also the CFFI library which provides an elegant interface to C.

If you do not want to use C then using the PyPy package which makes use of the JIT (Just In Time) compiler can give a significant performance boost to Python code.

### Proper Use of Data Structures and Algorithms

Taking time to consider other data structures and algorithms can provide a considerable performance boost by improving the time complexity of code.

### Memoization in Python

Those who know the concept of dynamic programming are well versed with the concept of memorization.

In memorization, the repetitive calculation is avoided by storing the values of the functions in the memory.

Although more memory is used, the performance gain is significant. Python comes with a library called `functools` that has an LRU cache decorator that can give you access to a cache memory that can be used to store certain values.


### Avoid using + for string concatenation

```py
    s = ", ".join((a, b, c))
```

### Use tuple packing notation for swapping two variables

```py
    a, b = b, a
```

### Use list comprehensions rather than loops to construct lists

```py
    b = [x*2 for x in a]
```

### Use chained comparisons

If you need to compare a value against an upper and lower bound, you can (and should) used operator chaining:

```py
    if 10 < a < 100:
        x = 2 * x

    if 10 < f(x) < 100:
        x = f(x) + 10
```

### Use the in operator to test membership

If you want to check if a particular value is present in a list, tuple, or set, you should use the in operator:

```py
    k = [1, 2, 3]
    if 2 in k:
        # ...
```


### Avoid global variables

A global variable is a variable that is declared at the top level so that it can be accessed by any part of the program.

While it can be very convenient to be able to access shared data from anywhere, it usually causes more problems than it solves, mainly because it allows any part of the code to introduce unexpected side effects. So globals are generally to be avoided. But if you need an extra reason to not use them, they are also slower to access.

### Use enumerate if you need a loop index

If for some reason you really need a loop index, you should use the enumerate function which is faster and clearer:

```py
    for i, x in enumerate(k):
        print(i, x)
```

### Use the latest release of Python

New versions of Python are released quite frequently (at the time of writing Python 3.9 has been updated 8 times in the last year). It is worth keeping up to date as new versions often have bug fixes and security fixes, but they sometimes have performance improvements too.


Here are three techniques and recommended approaches for creating memory-efficient Python classes: slots, lazy initialization, and generators [9]. 

### Use slots

Using the Python `__slots__` magic method, we can explicitly define the attributes that a class can contain which can help optimize the memory usage of classes [9].

By default, each instance of a Python classe stores its attributes in a private dictionary `__dict__` which allows for a lot of flexibility, but this comes at the cost of memory overhead.

When using `__slots__`, Python uses only a fixed amount of storage space for the specified attributes rather than using the default dictionary.

### Use Lazy Initialization

Lazy initialization is the technique in which we delay initialization of an attribute until it is actually needed [9].

By implementing lazy initialization, we can reduce the memory footprint of Python objects since only the necessary attributes will be initialized at runtime.

In Python, we can implement lazy initialization by using the `@cached_property` decorator.

### Use Generators

Python generators are a type of iterable which generate values on the fly as needed rather than all at once (lists and tuples) [9].

Generators are very memory-efficient when dealing with large datasets.

Generators allow us to lazily evaluate data and generats elements only when needed which can significantly improve performance [6].

```py
# Generator for generating large climate data on-the-fly
def generate_climate_data():
    for i in range(1000000):
        yield {"temperature": i % 100, "humidity": i % 70}

# Processing climate data without loading it all into memory
for data in generate_climate_data():
    if data["temperature"] > 90:
        print(f"High temperature: {data['temperature']}°C")
        break
```

Using a generator prevents the entire dataset from being loaded into memory at once which improves speed and memory usage.


## Scikit-learn Performance

Sometimes scikit-learn models can take a long time to train. How can we create the best scikit-learn model in the least amount of time? [4]

There are a few approaches to solving this problem:

- Changing your optimization function (solver).

- Using different hyperparameter optimization techniques (grid search, random search, early stopping).

- Parallelize or distribute your training with joblib and Ray.


----------


## Optimizing Memory Usage

Find out why Python apps are using too much memory and reduce RAM usage with these simple tricks and efficient data structures [7], [10]. 

### Find Bottlenecks

First we need to find the bottlenecks in the code that are hogging memory [10].

The `memory_profiler` tool measures memory usage of specific function on line-by-line basis.

We also install the `psutil` package which significantly improves the profiler performance.

The `memory_profiler` shows memory usage/allocation on line-by-line basis for the decorated function (here the `memory_intensive` function) which intentionally creates and deletes large lists.

Now that we are able to find specific lines that increase memory consumption, we can see how much each variable is using.

If we were to use `sys.getsizeof` to measure to measure variables, we woll get questionable information for some types of data structures. 

For integers or bytearrays we will get the real size in bytes, for containers such as list though, we will only get size of the container itself and not its contents.

A better approach is to use the `pympler` tool that is designed for analyzing memory behaviour which can help us obtain a realistic view of Python object sizes [10].

```py
    from pympler import asizeof

    print(asizeof.asizeof([1, 2, 3, 4, 5]))
    # 256

    print(asizeof.asized([1, 2, 3, 4, 5], detail=1).format())
    # [1, 2, 3, 4, 5] size=256 flat=96
    #     1 size=32 flat=32
    #     2 size=32 flat=32
    #     3 size=32 flat=32
    #     4 size=32 flat=32
    #     5 size=32 flat=32

    print(asizeof.asized([1, 2, [3, 4], "string"], detail=1).format())
    # [1, 2, [3, 4], 'string'] size=344 flat=88
    #     [3, 4] size=136 flat=72
    #     'string' size=56 flat=56
    #     1 size=32 flat=32
    #     2 size=32 flat=32
```

Pympler provides `asizeof` module with function of same name which correctly reports size of the list as well all values it contains and the `asized` function which can give a more detailed size breakdown of individual components of the object.

Pympler has many more features including tracking class instances or identifying memory leaks.

### Saving Some RAM

Now we need to find a way to fix memory issues. The quickest and easiest solution can be switching to more memory-efficient data structures [10].

Python lists are one of the more memory-hungry options when it comes to storing arrays of values:

Here we have used the `array` module which can store primitives such as integers or characters.

We can see that in this case memory usage peaked at just over 100MiB which is a huge difference in comparison to a list.

We can further reduce memory usage by choosing appropriate precision:

One major downside of using array as data container is that it does not support very many types.

If we need to perform a lot of mathematical operations on the data then should use NumPy arrays instead:

We can also improve the size of individual objects defined by Python classes using the `__slots__` class attribute which is used to explicitly declare class properties.

Declaring `__slots__` on a class also prevents creation of `__dict__` and `__weakref__` attributes which can be useful:

How do we store strings depends on how we want to use them. 

If we are going to search through a huge number of string values then using `list` is a bad choice.

The best option may be to use an optimized data structure such as tree, especially for static datasets used for querying [10]. 

There is a library for using tree-like data structures called [pytries](https://github.com/pytries) [10]. 

### Not Using RAM At All

Perhaps the easiest way to save RAM is to not use memory in a first place. 

We cannot avoid using RAM completely, but we can avoid loading the full dataset at once and work with the data incrementally, if possible.

The simplest method is using generators which return a lazy iterator that computes elements on demand rather than all at once [10].

An even stronger tool that we can leverage is _memory-mapped files_ which allows us to load only parts of data from a file [10].

The Python standard library provides `mmap` module that can be used to create memory-mapped files which behave like both files and bytearrays that can be used with file operations such as read, seek, or write as well as string operations [10]:

```py
    import mmap

    with open("some-data.txt", "r") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as m:
            print(f"Read using 'read' method: {m.read(15)}")
            # Read using 'read' method: b'Lorem ipsum dol'
            m.seek(0)  # Rewind to start
            print(f"Read using slice method: {m[:15]}")
            # Read using slice method
```

Loading and reading memory-mapped files is rather simple:

Most of the time, we will probably want to read the file as shown above but we also write to the memory-mapped file:

If we are performing computations in NumPy, it may he better to use its `memmap` feature which is suitable for NumPy arrays stored in binary files.


Here are some tips to optimize memory usage in Python [7].

### Cache it

In general, we want to cache anything that we download unless we know that we will not  need it again (or it will expire before we need it again) [7].

- A classical approach to caching is to organize a directory for storing the previously obtained objects by their identifiers. 

  The identifiers could be object URLs, tweet ids, or database row numbers; anything related to the object source.

- The next step is to convert an identifier to a unique file name. 

  We can write the conversion function ourselves or use the standard library. 
  
  We can start by encoding the identifier which is presumably a string.

Apply one of the hashing functions such as the `hashlib.md5()` or `hashlib.sha256()` which is faster to get a HASH object.

The functions do not produce totally unique file names but the likelihood of getting two identical file namescalled a _hash collision_ is very low.

- Obtain a hexadecimal digest of the object which is a 64-character ASCII string: a perfect file name that has no resemblance to the original object identifier.

Assuming the directory cache has already been created and is writable, we can use pickle to save our objects [7].

```py
    import hashlib

    source = "https://lj-dev.livejournal.com/653177.html"
    hash = hashlib.sha256(source.encode())
    filename = hash.hexdigest()
    print(hash, filename)

    # First, check if the object has already been pickled.
    cache = f'cache/{filename}.p'
    try:
      with open(cache, 'rb') as infile:
          # Has been pickled before! Simply unpickle
          object = pickle.load(infile)
    except FileNotFoundError:
        # Download and pickle
        object = 'https://lj-dev.livejournal.com/653177.html'
        with open(cache, 'wb') as outfile:
          pickle.dump(outfile, object)
    except:
        # Things happen...
```

### Sort big in place

Sorting and searching are arguably the two most frequent and important operations in modern computing [7]. 

Python has two functions for sorting lists: `list.sort()` and `sorted()`.

- `sorted()` sorts any iterable while `list.sort()` sorts only lists.

- `sorted()` creates a sorted copy of the original iterable.

- The `list.sort()` method sorts the list in place.

The `list.sort()` method  shuffles the list items around without making a copy. If we could load the list into memory, we could surely afford to sort it. However, list.sort() ruins the original order.

If the list is large then sort it in place using `list.sort()`. 

If the list is moderately sized or needs to preserve the original order, we can use `sorted()` to retrieve a sorted copy.

### Garbage collector

The C and C++ languages require that we allocate and deallocate memory ourselves, but Python manages allocation and deallocation itself [7].

- Each Python object has a reference count which is the number of variables and other objects that refer to this object. 

- When we create an object and do not assign it to a variable, the object has zero references.

-  When we redefine a variable, it no longer points to the old object and the reference count decreases.

```py
    'Hello, world!'         # An object without references
    s3 = s2 = s1 = s        # Four references to the same object!
    s = 'Goodbye, world!'   # Only three references remain
    strList = [s1]
    s1 = s2 = s3 = None     # Still one reference
```

When the reference count becomes zero, an object becomes unreachable. A part of Python runtime called garbage collector automatically collects and discards unreferenced objects. There is rarely a need to mess with garbage collection [7].

Suppose we work with big data — something large enough to stress the computer RAM.

We start with the original dataset and progressively apply expensive transformations and record the intermediate results. 

An intermediate result may be used in more than one subsequent transformation. 

Eventually, our computer memory will be clogged with large objects, some that are still needed and some that are not.

We can help Python by explicitly marking variables and objects associated with them for deletion using the `del` operator [7].

```py
    bigData = ...
    bigData1 = func1(bigData)
    bigData2 = func2(bigData)
    del bigData # Not needed anymore
```

The `del` does not remove the object from memory but merely marks the object as unreferenced and destroys its identifier. The garbage collector still must intervene and collect the garbage.

We can force garbage collection immediately in anticipation of heavy memory use [7].

```py
    import gc # Garbage Collector
    gc.collect()
```

NOTE: Garbage collection takes a long time, so we should only let it happen only when necessary.



## How to Profile Memory Usage

Here we learn how to quickly check the memory footprint of your machine learning function/module with one line of command [1].

Monitor line-by-line memory usage of functions with the `memory_profiler` module.

```bash
  pip install -U memory_profiler

  python -m memory_profiler some-code.py
```

It is easy to use this module to track the memory consumption of the function. `@profile` decorator that can be used before each function that needs to be tracked which will track the memory consumption line-by-line in the same way as of line-profiler [1].

```py
    from memory_profiler import profile

    @profile
    def my_func():
        a = [1] * (10 ** 6)
        b = [2] * (2 * 10 ** 7)
        c = [3] * (2 * 10 ** 8)
        del b
        return a

    if __name__=='__main__':
        my_func()
```

We can use pthe Python `cProfile` and `timeit` modules for profiling and benchmarking code [4].

```py
import cProfile

def process_climate_data():
    temperatures = [15, 22, 30, 18, 25] * 1000000  # Large dataset
    converted = [temp * 9/5 + 32 for temp in temperatures]
    max_temp = max(converted)
    return max_temp

# Profile the function
cProfile.run('process_climate_data()')
```


## References

[1]: [Why Python is so slow and how to speed it up](https://towardsdatascience.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)

[2]: [Lightning Fast Iteration Tips For Python Programmers](https://towardsdatascience.com/lightning-fast-iteration-tips-for-python-programmers-61d4f72bf4f0)

[3]: [5 Tips To Optimize Your Python Code](https://towardsdatascience.com/try-these-5-tips-to-optimize-your-python-code-c7e0ccdf486a?source=rss----7f60cf5620c9---4)

[4]: [How to Speed up Scikit-Learn Model Training](https://medium.com/distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1)

[5]: [Horizontal vs Vertical Scaling: Which One Should You Choose?](https://www.finout.io/blog/horizontal-vs-vertical-scaling)

[6]: [Are you optimizing your Python code or ignoring performance?](https://medium.com/codex/are-you-optimizing-your-python-code-or-ignoring-performance-0053a6867b68)

----------

[7]: [Optimizing memory usage in Python code](https://medium.com/geekculture/optimising-memory-usage-in-python-code-d50a9c2a562b)

[8]: [Optimizing Memory Usage in Python Applications](https://towardsdatascience.com/optimizing-memory-usage-in-python-applications-f591fc914df5)

[9]: [How to Write Memory-Efficient Classes in Python](https://towardsdatascience.com/how-to-write-memory-efficient-classes-in-python-beb90811abfa)

[10]: [How Much Memory is your ML Code Consuming?](https://towardsdatascience.com/how-much-memory-is-your-ml-code-consuming-98df64074c8f)

[11]: [Profile Memory Consumption of Python functions in a single line of code](https://towardsdatascience.com/profile-memory-consumption-of-python-functions-in-a-single-line-of-code-6403101db419)

[12]: [Are you optimizing your Python code or ignoring performance?](https://medium.com/codex/are-you-optimizing-your-python-code-or-ignoring-performance-0053a6867b68)


[4 easy-to-implement, high-impact tweaks for supercharging your Python code’s performance](https://towardsdatascience.com/4-easy-to-implement-high-impact-tweaks-for-supercharging-your-python-codes-performance-eb0652d942b7)
