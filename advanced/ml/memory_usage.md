# Memory Usage Tips

Here are some resources to evaluate Python memory usage, also see [Performance](./performance.md). 


## Optimizing Memory Usage in Python Applications

Fins out why your Python apps are using too much memory and reduce their RAM usage with these simple tricks and efficient data structures

### Find Bottlenecks

First we need to find the bottlenecks in the code that are hogging memory.

The `memory_profiler` tool measures memory usage of specific function on line-by-line basis. 

We also install the `psutil` package which significantly improves the profiler performance.

The memory_profiler shows memory usage/allocation on line-by-line basis for the decorated function (here the memory_intensive function) which intentionally creates and deletes large lists.

Now that we are able to find specific lines that increase memory consumption, we can see how much each variable is using. 

If we were to use `sys.getsizeof` to measure to measure variables, we woll get questionable information for some types of data structures. For integers or bytearrays we will get the real size in bytes, for containers such as list though, we will only get size of the container itself and not its contents. 

A better approach is to the pympler tool that is designed for analyzing memory behaviour which can help you get more realistic view of Python object sizes. 

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

Now we need to find a way to fix memory issues. The quickest and easiest solution can be switching to more memory-efficient data structures.

Python lists are one of the more memory-hungry options when it comes to storing arrays of values:

In this example we used the `array` module which can store primitives such as integers or characters. 

We can see that in this case memory usage peaked at just over 100MiB which is a huge difference in comparison to list. 

We can further reduce memory usage by choosing appropriate precision:

One major downside of using array as data container is that it doesn't support that many types.

If we need to perform a lot of mathematical operations on the data then should use NumPy arrays instead:

The above optimizations help with overall size of arrays of values but we can also improvem the size of individual objects defined by Python classes using `__slots__` class attribute which is used to explicitly declare class properties. 

Declaring `__slots__` on a class also prevents creation of `__dict__` and `__weakref__` attributes which can be useful:

How do we store strings depends on how we want to use them. If we are going to search through a huge number of string values then using `list` is a bad idea. 

The best option may be to use optimized data structure such as trie, especially for static data sets which you use for example for querying and there is a library for that as well as for many other tree-like data structures called [pytries](https://github.com/pytries).

### Not Using RAM At All

Perhaps the easiest way to save RAM is to not use memory in a first place. We obviously cannot avoid using RAM completely but we can avoid loading the full data set at once and work with the data incrementally when possible. 

The simplest way to achieve this is using generators which return a lazy iterator which computes elements on demand rather than all at once.

An even stronger tool that we can leverage is _memory-mapped files_ which allows us to load only parts of data from a file. 

The Python standard library provides `mmap` module which can be used to create memory-mapped files that behave  like both files and bytearrays that can be used with file operations such read, seek, or write as well as string operations:

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


----------



## How to Profile Memory Usage

Learn how to quickly check the memory footprint of your machine learning function/module with one line of command. Generate a nice report too [2].

Monitor line-by-line memory usage of functions with memory profiler module. 

```bash
  pip install -U memory_profiler
  
  python -m memory_profiler some-code.py
```

It is easy to use this module to track the memory consumption of the function. `@profile` decorator that can be used before each function that needs to be tracked which will track the memory consumption line-by-line in the same way as of line-profiler.

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



## References

[1] [Profile Memory Consumption of Python functions in a single line of code](https://towardsdatascience.com/profile-memory-consumption-of-python-functions-in-a-single-line-of-code-6403101db419)

[2] [How Much Memory is your ML Code Consuming?](https://towardsdatascience.com/how-much-memory-is-your-ml-code-consuming-98df64074c8f)

[3] [Optimizing Memory Usage in Python Applications](https://towardsdatascience.com/optimizing-memory-usage-in-python-applications-f591fc914df5)

