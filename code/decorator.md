# The Decorator Pattern

<!-- MarkdownTOC -->

- Python decorator attributes
    - @staticmethod
    - @classmethod
    - @property
- Decorator Code Snippets
    - Timer
    - Measure Function Performance
    - Repeat
    - Show prompt
    - Try/Catch
    - Convert Data
    - Memoization
    - Function Catalog
- Separation of Concerns
    - Logging
    - Timing
    - Counting
    - Checking parameter types
    - Checking parameter values
    - Exception handling
    - Authentication
    - Ignoring functions
    - Switching functions
    - Changing arguments and return values
    - Memoization
    - Dynamic application of decorators
- New Type Annotation Features in Python 3.11
    - Self — the Class Type
    - Arbitrary Literal String
    - Varying Generics
    - TypedDict — Flexible Key Requirements
- References

<!-- /MarkdownTOC -->

## Python decorator attributes

A _decorator_ is a function that takes another function as input, extends its behavior, and returns a new function as output which is possible becaus functions are first-class objects in Python which means they can be passed as arguments to functions and also be returned from functions just like other types of objects such as string, int, or float. Thus, a decorator can be used to decorate a function or a class.

Here we discuss three special decorators: @staticmethod, @classmethod, and @property which are “magical” decorators that can be very handy for our development work and make your code more clean [2].

### @staticmethod

A static method is a method that does not require the creation of an instance of a class. 

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        def get_number(self):
            return self.number
          
        @staticmethod
        def get_emergency_number():
            return "911"
          
    Cellphone.get_emergency_number()
    # '911'
```

### @classmethod

A class method requires the class itself as the first argument which is written as cls. 

A class method normally works as a factory method and returns an instance of the class with supplied arguments. However, it does not have to work as a factory class and return an instance.

We can create an instance in the class method and do whatever you need without having to return it.

Class methods are very commonly used in third-party libraries.

Here, it is a factory method here and returns an instance of the Cellphone class with the brand preset to “Apple”.

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        def get_number(self):
            return self.number
          
        @staticmethod
        def get_emergency_number():
            return "911"
          
        @classmethod
        def iphone(cls, number):
            _iphone = cls("Apple", number)
            print("An iPhone is created.")
            return _iphone
         
    iphone = Cellphone.iphone("1112223333")
    # An iPhone is created.
    iphone.get_number()
    # "1112223333"
    iphone.get_emergency_number()
    # "911"
```

If you use class methods properly, you can reduce code redundancy dramatically and make your code more readable and more professional. 

The key idea is that we can create an instance of the class based on some specific arguments in a class method, so we do not have to repeatedly create instances in other places (DRY).


### @property

In the code snippet above, there is a function called `get_number` which returns the number of a Cellphone instance. 

We can optimize the method a bit and return a formatted phone number.


In Python, we can also use getter and setter to easily manage the attributes of the class instances.


```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        @property
        def number(self):
            _number = "-".join([self._number[:3], self._number[3:6],self._number[6:]])
            return _number
        
        @number.setter
        def number(self, number):
            if len(number) != 10:
                raise ValueError("Invalid phone number.")
            self._number = number

    cellphone = Cellphone("Samsung", "1112223333")
    print(cellphone.number)
    # 111-222-3333

    cellphone.number = "123"
    # ValueError: Invalid phone number.
```


Here is the complete example using the three decorators in Python: `@staticmethod`, `@classmethod`, and `@property`:

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        @property
        def number(self):
            _number = "-".join([self._number[:3], self._number[3:6],self._number[6:]])
            return _number

        @number.setter
        def number(self, number):
            if len(number) != 10:
                raise ValueError("Invalid phone number.")
            self._number = number
        
        @staticmethod
        def get_emergency_number():
            return "911"
        
        @classmethod
        def iphone(cls, number):
            _iphone = cls("Apple", number)
            print("An iPhone is created.")
            return _iphone
```


## Decorator Code Snippets

The beauty of decorators is that they are easy to apply but provide a lot of extra functionalities for your code. 

Here we disucss some decorators that we can easily apply to real-world problems when debugging  code [3]. 

### Timer

```py
    def timer(func):
      """
      Display time it took for our function to run. 
      """
      @wraps(func)
      def wrapper(*args, **kwargs):
        start = time.perf_counter()
    
        # Call the actual function
        res = func(*args, **kwargs)
    
        duration = time.perf_counter() - start
        print(f'[{wrapper.__name__}] took {duration * 1000} ms')
        return res
        return wrapper
```

```py
    @timer
    def isprime(number: int):
      """ Check if a number is a prime number """
      isprime = False
      for i in range(2, number):
        if ((number % i) == 0):
          isprime = True
          break
          return isprime
```

### Measure Function Performance

```py
    def performance_check(func):
        """ Measure performance of a function """
        @wraps(func)
        def wrapper(*args, **kwargs):
          tracemalloc.start()
          start_time = time.perf_counter()
          res = func(*args, **kwargs)
          duration = time.perf_counter() - start_time
          current, peak = tracemalloc.get_traced_memory()
          tracemalloc.stop()
    
          print(f"\nFunction:             {func.__name__} ({func.__doc__})"
                f"\nMemory usage:         {current / 10**6:.6f} MB"
                f"\nPeak memory usage:    {peak / 10**6:.6f} MB"
                f"\nDuration:             {duration:.6f} sec"
                f"\n{'-'*40}"
          )
          return res
          return wrapper
```

```py
    @performance_check
    def is_prime_number(number: int):
        """Check if a number is a prime number"""
        # ....rest of the function
```

### Repeat

```py
    def repeater(iterations:int=1):
      """ Repeat the decorated function [iterations] times """
      def outer_wrapper(func):
        def wrapper(*args, **kwargs):
          res = None
          for i in range(iterations):
            res = func(*args, **kwargs)
          return res
        return wrapper
        return outer_wrapper
```

```py
    @repeater(iterations=2)
    def sayhello():
      print("hello")
```

### Show prompt

```py
    def prompt_sure(prompt_text:str):
      """ Show prompt asking you whether you want to continue. Exits on anything but y(es) """
      def outer_wrapper(func):
        def wrapper(*args, **kwargs):
          if (input(prompt_text).lower() != 'y'):
            return
          return func(*args, **kwargs)
        return wrapper
        return outer_wrapper
```

```py
    @prompt_sure('Sure? Press y to continue, press n to stop')
    def say_hi():
      print("hi")
```

### Try/Catch

```py
    def trycatch(func):
      """ Wraps the decorated function in a try-catch. If function fails print out the exception. """
      @wraps(func)
      def wrapper(*args, **kwargs):
        try:
          res = func(*args, **kwargs)
          return res
        except Exception as e:
          print(f"Exception in {func.__name__}: {e}")
          return wrapper
```

```py
    @trycatch
    def trycatchExample(numA:float, numB:float):
      return numA / numB
```

### Convert Data

```py
    import numpy as np
    import pandas as pd
     
    # function decorator to ensure numpy input
    # and round off output to 4 decimal places
    def ensure_numpy(fn):
        def decorated_function(data):
            array = np.asarray(data)
            output = fn(array)
            return np.around(output, 4)
        return decorated_function
     
    @ensure_numpy
    def numpysum(array):
        return array.sum()
     
    x = np.random.randn(10,3)
    y = pd.DataFrame(x, columns=["A", "B", "C"])
     
    # output of numpy .sum() function
    print("x.sum():", x.sum())
    print()
     
    # output of pandas .sum() funuction
    print("y.sum():", y.sum())
    print(y.sum())
    print()
     
    # calling decorated numpysum function
    print("numpysum(x):", numpysum(x))
    print("numpysum(y):", numpysum(y))
```

### Memoization

There are some function calls that we do repeatedly but the values rarely change. 

This could be calls to a server where the data is relatively static or as part of a dynamic programming algorithm or computationally intensive math function. 

We might want to memoize these function calls -- storing the value of their output on a virtual memo pad for reuse later.

A decorator is the best way to implement a memoization function. 

Here, we implement the `memoize()` to work using a global dictionary MEMO such that the name of a function together with the arguments becomes the key and the function’s return becomes the value. 

When the function is called, the decorator will check if the corresponding key exists in MEMO, and the stored value will be returned. Otherwise, the actual function is invoked and its return value is added to the dictionary.

```py
    import pickle
    import hashlib
     
     
    MEMO = {} # To remember the function input and output
     
    def memoize(fn):
        def _deco(*args, **kwargs):
            # pickle the function arguments and obtain hash as the store keys
            key = (fn.__name__, hashlib.md5(pickle.dumps((args, kwargs), 4)).hexdigest())
            # check if the key exists
            if key in MEMO:
                ret = pickle.loads(MEMO[key])
            else:
                ret = fn(*args, **kwargs)
                MEMO[key] = pickle.dumps(ret)
            return ret
        return _deco
     
    @memoize
    def fibonacci(n):
        if n in [0, 1]:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
     
    print(fibonacci(40))
    print(MEMO)
    ```
    
Memoization is very helpful for expensive functions whose outputs do not change frequently such as reading stock market data from the Internet. 
    
```py
    import pandas_datareader as pdr
     
    @memoize
    def get_stock_data(ticker):
        # pull data from stooq
        df = pdr.stooq.StooqDailyReader(symbols=ticker, start="1/1/00", end="31/12/21").read()
        return df
     
    #testing call to function
    import cProfile as profile
    import pstats
     
    for i in range(1, 3):
        print(f"Run {i}")
        run_profile = profile.Profile()
        run_profile.enable()
        get_stock_data("^DJI")
        run_profile.disable()
        pstats.Stats(run_profile).print_stats(0)
```

Python 3.2 or later shipped you the decorator `lru_cache` from the built-in library functools. 

The `lru_cache` implements LRU caching which limits its size to the most recent calls (default 128) to the function. In Python 3.9, there is a @functools.cache as well, which is unlimited in size without the LRU purging.

```py
    import functools
    import pandas_datareader as pdr
     
    # memoize using lru_cache
    @functools.lru_cache
    def get_stock_data(ticker):
        # pull data from stooq
        df = pdr.stooq.StooqDailyReader(symbols=ticker, start="1/1/00", end="31/12/21").read()
        return df
     
    # testing call to function
    import cProfile as profile
    import pstats
     
    for i in range(1, 3):
        print(f"Run {i}")
        run_profile = profile.Profile()
        run_profile.enable()
        get_stock_data("^DJI")
        run_profile.disable()
        pstats.Stats(run_profile).print_stats(0)
```


### Function Catalog

Another example is to register functions in a catalog which allows us to associate functions with a string and pass the strings as arguments for other functions. 

A function catalog is the start to making a system to allow user-provided plug-ins such `activate()`.



## Separation of Concerns

We can use decorators to easily apply the _separation of concerns_ principle [4].

This can be very useful but it can also be a double-edged sword. By their nature, decorators drag in hidden functions from distant corners of the code base which can cause some serious problems. 

It is usually best to limit decorators to areas where we can derive the most benefit, and generally use decorators that perform a conceptually simple function (such as logging or authenticating a user).

Basically, avoid making decorators your go to solution for every problem. Use decorators sparingly where the benefits definitely outweigh the downsides. In the right scenario, decorators are an extremely useful tool.

### Logging

```py
    def log(function):

        def wrapped_function():
            print("Entering", function.__name__)
            function()
            print("Leaving", function.__name__)

        return wrapped_function

    @log
    def show1():
        print("show1")

    @log
    def show2():
        print("show2")

    show1()
    show2()
```

The `log` function is a special type of construct called a closure which does the following:

- The log function accepts a parameter function.

- Inside the log function we declare another function wrapped_function that implements the required logging functionality around whatever function we pass into log. wrapped_function is called an inner function of log.

- The log function then returns wrapped_fnction.

The way the closure works is that when we call `log(show1)` it returns a brand new function (the inner function) that executes show1 with the logging code around it.


### Timing

Another use of decorators is to provide profiling information. 

We could easily modify the previous example to check the time at the start and end of each call and store the timing data in some structure that can be accessed and analysed when the program ends.


### Counting

We can also use decorators to count how many times each function is called. 

```py
    def count(function):

        call_count = 0

        def wrapped_function():
            nonlocal call_count
            call_count += 1
            print("Call", call_count, "of", function.__name__)
            function()

        return wrapped_function

    @count
    def show1():
        print("show1")

    @count
    def show2():
        print("show2")

    show1()
    show2()
    show1()
    show2()
```

### Checking parameter types

We could add code to the function itself to check the type of the input parameter, but it adds a bit of extra code that distracts form the main purpose of the function. Thus, we can add a decorator to check the parameter.

```py
    def accepts_string(function):

        def wrapped_function(s):
            if not isinstance(s, str):
                raise TypeError("String parameter required")
            return function(s)

        return wrapped_function

    @accepts_string
    def show_upper(s):
        print(s.upper())

    show_upper("abc")
    show_upper(1)
```

The decorator checks the parameter and throws an exception if the parameter is not a string. 

We could enhance this by maybe throwing a custom exception and perhaps logging a message too but for the sake of simplicity we just throw a `TypeError`. 

The decorator adds this extra check without messing up the main code of the function and also allows us to reuse the same decorator on other functions without duplicating code.

### Checking parameter values

In the same manner, we can use a decorator to check the value of a parameters

```py
    def check_value(function):

        def wrapped_function(s):
            if not s:
                raise ValueError("Non-empty string parameter required")
            return function(s)

        return wrapped_function

    @check_value
    def show_upper(s):
        print(s.upper())

    show_upper("abc")
    show_upper("")
```

This works the same way to the previous example but instead of checking the type of the argument it checks the value. In a real application we would probably want to check the type and the value which can be combined in a single decorator.

### Exception handling

Certain operations can cause runtime exceptions which is usually done using a `try` block to surround the code that might throw an exception but that adds more boilerplate code to the function.

Once again, we can use a decorator to handle the error case, so the function itself can stick to its main concern of executing the happy path and ignoring the secondary concern of handling error cases.

```py
    def handle_exception(function):

        def wrapped_function(a, b):
            try:
                return function(a, b)
            except Exception:
                return None

        return wrapped_function

    @handle_exception
    def divide(a, b):
       return(a / b)

    print(divide(1, 2))
    print(divide(3, 0))
```

In this case our function performs a divide operation which can potentially raise an exception if the divisor is zero.

The `handle_exception` decorator works by catching this exception and returning `None` which means  our code does not need to bother with exception handling, but any code that calls divide will need to take account of the fact that a return value of `None` indicates an error.

There are other possibilities, we could log the exception or rethrow the exception with additional information. In some cases, it might be preferable to ignore the exception if it is non-critical but that is not usually a good idea.

### Authentication

This example shows how we can avoid boilerplate code using decorators. Suppose we have code that requires user authentication for certain operations:

```py
    def authenticate(function):

        def wrapped_function():
            if do_authenticate():
                function()

        return wrapped_function

    @authenticate
    def secure_function(s):
        print("Authorised user")
```

Here, `secure_function` is some function that should only be performed if the current user is authenticated. The `do_authenticate` is a function that performs authentication and returns `true` or `false`.

This implementation simply ignores the call is the user is not known, but we would probably want to provide some kind of notification or perhaps throw an exception which should be implemented within the `wrapped_function`.

### Ignoring functions

We may not want to do this very often but it is useful to know that it is possible.

This simple decorator causes a function to not be called:

```py
    def ignore(function):

        def wrapped_function():
            pass

        return wrapped_function

    @ignore
    def show():
        print("Executing show")

    show() # Doesn't execute show()
```

Here `wrapped_function` does not call the original function when we apply the ignore decorator to `show`.

### Switching functions

We may not want to do this very often but it is useful to know that it is possible.

This decorator causes a different function to be called:

```py
def switch(function):

    def wrapped_function():
        other_show()

    return wrapped_function

@switch
def show():
    print("Executing show")

def other_show():
    print("Executing other show")

show() # Actually executes other_show
```

Here `wrapped_function` always calls `other_show` regardless of which function is wrapped. When we apply the `switch` decorator to show, then we can call `show` but `other_show` will be executed.

### Changing arguments and return values

We can use decorators to affect the function’s arguments and return values.

This example implements byte subtraction where byte values are limited to the range 0 to 255. 

In this example `subtract_bytes` does a simple subtraction, but the `clamp` decorator clamps the input values to the range 0 to 255 before calling the function and clamps the result to the same range after the function returns. 

By clamping we mean that values less than 0 are set to 0 and values greater than 255 are set to 255.

```py
    def clamp(function):

        def inner(a, b):
            a = min(255, max(a, 0))
            b = min(255, max(b, 0))
            r = function(a, b)
            return min(255, max(r, 0))

        return inner


    @clamp
    def subtract_bytes(a, b):
        return a - b


    print(subtract_bytes(500, 50))
    print(subtract_bytes(120, 130))
```

In the first call, we pass in values of 500 and 50 but 500 is clamped to 255 so the result (255–50) is 205. 

In the second example we pass in 120 and 130, so the result (120–130) is -10 but that is clamped to 0 by the decorator.

### Memoization

Suppose we have a function that takes a long time to execute and it might be called many times with the same input values. Therefore, we may want to avoid calling the function more than once with the same input values.

This only works for pure functions (a pure function is a function that always returns the same value for a given set of inputs and has no other side effects).

For example, the `negate(x)` function returns the negative of x. So if we call `negate(1)` it returns -1. If we call `negate(1)` again later, we do not need to repeat the calculation, so we can simply remember the last time it was called with value 1 and return the same result which is called `memoization`.

```py
    def memoize(function):

        cache = dict()

        def inner(a):
            if a not in cache:
                cache[a] = function(a)
            return cache[a]

        return inner

    @memoize
    def negate(a):
        print("negating", a)
        return -a

    print(negate(1))
    print(negate(2))
    print(negate(1))
```

The `memoize` decorator maintains a dictionary cache that holds previous results. 

The inner function checks if the result is known and only calculates the result if it has not been calculated already. 

Here is the output from this code:

```
    negating 1
    -1
    negating 2
    -2
    -1
```

Although we call `negate(1)` twice and correctly print the result both times, the inner negate function is only called once (so the string "negating 1" is only printed once).

With a complex and time-consuming functions memoization can provide good performance improvements. 

Note that this implementation uses the parameter value as a dictionary key, so our simple decorator only works with values that are suitable keys such as numbers or strings. The `functools` module has a decorator called `lru_cache` that provides a full implementation.

### Dynamic application of decorators

The @ notation is really just syntactic sugar but we do not have to use it. 

We can apply the decorator manually which allows us to create different versions of the same function. 

```py
    def log(function):

        def wrapped_function():
            print("Entering", function.__name__)
            function()
            print("Leaving", function.__name__)

        return wrapped_function

    def show():
        print("show")

    logged_show = log(show)
```

Since `log(show)` returns a function that executes `show()` surrounded by the logging code, so we can call `logged_show()` to execute `show()` with logging, and we can also call `show()` to execute the function directly with no logging.



----------



## New Type Annotation Features in Python 3.11

The improvement of type annotations in Python 3.11 can help to write bug-free code [5].

### Self — the Class Type

The following code does not use type hints which may cause problems. 

```py
    class Box:
        def paint_color(self, color):
            self.color = color
            return self
```

We can use Self to indicate that the return value is an object in the type of “Self" which is interpreted as the Box class.

```py
    from typing import Self
    class Box:
        def paint_color(self, color: str) -> Self:
            self.color = color
            return self
```

### Arbitrary Literal String

When we want a function to take a string literal, we must specify the compatible string literals. 
        
Python 3.11 introduces a new general type named `LiteralString` which allows the users to enter any string literals. 

```py
    from typing import LiteralString

    def paint_color(color: LiteralString):
        pass

    paint_color("cyan")
    paint_color("blue")
```

The `LiteralString` type gives the flexibility of using any string literals instead of specific string literals when we use the `Literal` type. 

### Varying Generics

We can use `TypeVar` to create generics with a single type, as we did previously for Box. When we do numerical computations (such as array-based operations in NumPy and TensorFlow), we use arrays that have varied dimensions and shapes.

When we provide type annotations to these varied shapes, it can be cumbersome to provide type information for each possible shape which requires a separate definition of a class since the exiting TypeVar can only handle a single type at a time.

Python 3.11 is introducing the TypeVarTuple that allows you to create generics using multiple types. Using this feature, we can refactor our code in the previous snippet, and have something like the below:

```py
    from typing import Generic, TypeVarTuple
    Dim = TypeVarTuple('Dim')
    class Shape(Generic[*Dim]):
        pass
```

Since it is a tuple object, we can use a starred expression to unpack its contained objects which is a variable number of types. 

The above Shape class can be of any shape which has more flexibility and eliminates the need of creating separate classes for different shapes.

### TypedDict — Flexible Key Requirements

In Python, dictionaries are a powerful data type that saves data in the form of key-value pairs. 

The keys are arbitrary and you can use any applicable keys to store data. However, sometimes we may want to have a structured dictionary that has specific keys and the values of a specific type which means using TypedDict. 

```py
    from typing import TypedDict
    class Name(TypedDict):
        first_name: str
        last_name: str
```

We know that some people may have middle names (middle_name) and some do not. 

There are no direct annotations to make a key optional and the current workaround is creating a superclass that uses all the required keys while the subclass includes the optional keys. 

Python 3.11 introduces NotRequired as a type qualifier to indicate that a key can be potentially missing for TypedDict. The usage is very straightforward. 

```py
    from typing import TypedDict, NotRequired
    class Name(TypedDict):
        first_name: str
        middle_name: NotRequired[str]
        last_name: str
```

If we have too many optional keys, we can specify those keys that are required using `Required` instead of specifying those optional as not required. 

Thus, the alternative equivalent solution for the above issue:

```py
    from typing import TypedDict, Required
    class Name(TypedDict, total=False):
        first_name: Required[str]
        middle_name: str
        last_name: Required[str]
```

Note in the code snippet we specify `total=False` which makes all the keys optional. In the meantime, we mark these required keys as `Required` which means that the other keys are optional.



## References

[1] [A Gentle Introduction to Decorators in Python](https://machinelearningmastery.com/a-gentle-introduction-to-decorators-in-python/)

[2] [How to Use the Magical @staticmethod, @classmethod, and @property Decorators in Python](https://betterprogramming.pub/how-to-use-the-magical-staticmethod-classmethod-and-property-decorators-in-python-e42dd74e51e7?gi=8734ec8451fb)

[3] [5 real handy python decorators for analyzing/debugging your code](https://towardsdatascience.com/5-real-handy-python-decorators-for-analyzing-debugging-your-code-c22067318d47)
 
[4] [12 Ways to Use Function Decorators to Improve Your Python Code](https://medium.com/geekculture/12-ways-to-use-function-decorators-to-improve-your-python-code-f35515a45e3b)

[5] [4 New Type Annotation Features in Python 3.11](https://betterprogramming.pub/4-new-type-annotation-features-in-python-3-11-84e7ec277c29)

