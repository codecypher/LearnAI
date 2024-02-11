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

Hardware upgrades (vertical scaling) usually provide only marginal improvement in performance. However, we can achieve as much as 30-100x performance improvement using software libraries and code refactoring to improve parallelization (horizontal scaling) [2] [3].

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


### Use slots

Using the Python `__slots__` magic method, we can explicitly define the attributes that a class can contain which can help optimize the memory usage of classes [7]. 

By default, each instance of a Python classe stores its attributes in a private dictionary `__dict__` which allows for a lot of flexibility, but this comes at the cost of memory overhead. 

When using `__slots__`, Python uses only a fixed amount of storage space for the specified attributes rather than using the default dictionary.

### Use Lazy Initialization

Lazy initialization is the technique in which we delay initialization of an attribute until it is actually needed [7]. 

By implementing lazy initialization, we can reduce the memory footprint of Python objects since only the necessary attributes will be initialized at runtime.

In Python, we can implement lazy initialization by using the `@cached_property` decorator. 

### Use Generators

Python generators are a type of iterable which generate values on the fly as needed rather than all at once (lists and tuples) [7]. 

Generators are very memory-efficient when dealing with large amounts of data.



## Optimize Memory Usage

Here are some tips to optimize memory usage in python [4].

### Cache it

In general, we want to cache anything that we download unless we know for certian that we will not  need it again or it will expire before we need it again.

- A classical approach to caching is to organize a directory for storing the previously obtained objects by their identifiers. The identifiers may be, for example, objects’ URLs, tweet ids, or database row numbers; anything related to the objects’ sources.

- The next step is to convert an identifier to a uniform-looking unique file name. We can write the conversion function ourselves or use the standard library. Start by encoding the identifier, which is presumably a string. 

Apply one of the hashing functions such as the hashlib.md5() or hashlib.sha256() which is faster to get a HASH object. 

The functions do not produce totally unique file names but the likelihood of getting two identical file names (called a hash collision) is so low that we can ignore it for all practical purposes. 

- Obtain a hexadecimal digest of the object which is a 64-character ASCII string: a perfect file name that has no resemblance to the original object identifier.

Assuming that the directory cache has already been created and is writable, we can pickle our objects into it.

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

Sorting and searching are arguably the two most frequent and important operations in modern computing. 

Sorting and searching are so important that Python has two functions for sorting lists: list.sort() and sorted().

- `sorted()` sorts any iterable while `list.sort()` sorts only lists.
- `sorted()` creates a sorted copy of the original iterable.
- The `list.sort()` method sorts the list in place. 

The `list.sort()` method  shuffles the list items around without making a copy. If we could load the list into memory, we could surely afford to sort it. However, list.sort() ruins the original order. 

In summary, if our list is large then sort it in place with `list.sort()`. If our list is moderately sized or needs to preserve the original order, we can call `sorted()` and retrieve a sorted copy.

### Garbage collector

Python is a language with implicit memory management. The C and C++ languages require that we allocate and deallocate memory ourselves, but Python manages allocation and deallocation itself. 

When we define a variable through the assignment statement, Python creates the variable and the objects associated with it.

Each Python object has a reference count, which is the number of variables and other objects that refer to this object. When we create an object and do not assign it to a variable, the object has zero references.

When we redefine a variable, it no longer points to the old object and the reference count decreases.

```py
    'Hello, world!'         # An object without references
    s3 = s2 = s1 = s        # Four references to the same object!
    s = 'Goodbye, world!'   # Only three references remain
    strList = [s1]
    s1 = s2 = s3 = None     # Still one reference
```

When the reference count becomes zero, an object becomes unreachable. For most practical purposes, an unreachable object is a piece of garbage. A part of Python runtime called garbage collector automatically collects and discards unreferenced objects. There is rarely a need to mess with garbage collection, but here is a scenario where such interference is helpful.

Suppose we work with big data — something large enough to stress our computer’s RAM. 

We start with the original data set and progressively apply expensive transformations to it and record the intermediate results. An intermediate result may be used in more than one subsequent transformation. Eventually, our computer memory will be clogged with large objects, some of which are still needed while some aren’t. 

We can help Python by explicitly marking variables and objects associated with them for deletion using the `del` operator.

```py
    bigData = ...
    bigData1 = func1(bigData) 
    bigData2 = func2(bigData)
    del bigData # Not needed anymore
```

Bear in mind that del doesn’t remove the object from memory. It merely marks it as unreferenced and destroys its identifier. The garbage collector still must intervene and collect the garbage. 

We may want to force garbage collection immediately in anticipation of heavy memory use.

```py
    import gc # Garbage Collector
    gc.collect()
```

NOTE: Do not abuse this feature. Garbage collection takes a long time, so we should only let it happen only when necessary.



----------



## Scikit-learn Performance

Sometimes scikit-learn models can take a long time to train. How can we create the best scikit-learn model in the least amount of time? [6]

There are a few approaches to solving this problem:

- Changing your optimization function (solver).

- Using different hyperparameter optimization techniques (grid search, random search, early stopping).

- Parallelize or distribute your training with joblib and Ray.




## References

[1] [Why Python is so slow and how to speed it up](https://towardsdatascience.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)

[2] [Lightning Fast Iteration Tips For Python Programmers](https://towardsdatascience.com/lightning-fast-iteration-tips-for-python-programmers-61d4f72bf4f0)

[3] [5 Tips To Optimize Your Python Code](https://towardsdatascience.com/try-these-5-tips-to-optimize-your-python-code-c7e0ccdf486a?source=rss----7f60cf5620c9---4)

[4] [Optimizing memory usage in Python code](https://medium.com/geekculture/optimising-memory-usage-in-python-code-d50a9c2a562b)


[6] [How to Speed up Scikit-Learn Model Training](https://medium.com/distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1)

[7] [How to Write Memory-Efficient Classes in Python](https://towardsdatascience.com/how-to-write-memory-efficient-classes-in-python-beb90811abfa)


[4 easy-to-implement, high-impact tweaks for supercharging your Python code’s performance](https://towardsdatascience.com/4-easy-to-implement-high-impact-tweaks-for-supercharging-your-python-codes-performance-eb0652d942b7)

