# Python Decorator Code Snippets

Here are some useful code snippets using Python decorators.

## Separation of Concerns

We can use decorators to easily apply the _separation of concerns_ principle [2].

This can be very useful but it can also be a double-edged sword. By their nature, decorators drag in hidden functions from distant corners of the code base which can cause some serious problems.

It is usually best to limit decorators to areas where we can derive the most benefit, and generally use decorators that perform a conceptually simple function (such as logging or authenticating a user).

Be sure to use decorators sparingly. In the right scenario, decorators can be an extremely useful tool.

### Simple Logging

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

The `log` function is a special type of construct called a _closure_ which does the following:

- The log function accepts a parameter function.

- Inside the log function we declare another function `wrapped_function` that implements the required logging functionality around whatever function we pass into log. The wrapped_function is called an _inner_ function of log.

- The log function then returns wrapped_fnction.

The way the closure works is that when we call `log(show1)` it returns a brand new function (the inner function) that executes `show1` with the logging code around it.

### Simple Timing

Another use of decorators is to provide profiling information [1].

We could easily modify the previous example to check the time at the start and end of each call and store the timing data in some structure that can be accessed and analysed when the program ends.

### Simple Counting

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


## Utility Decorators

The beauty of decorators is that they are easy to apply but provide a lot of extra functionalities for your code.

Here we disucss some decorators that we can easily apply to real-world problems when debugging code [1].

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

A decorator is the best way to implement a memoization function [1].

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

## More Decorators

Python decorator functions act as wrappers around the target functions to provide a higher-order function that takes in the original function as an argument and returns a new function that incorporates the desired modifications [3].

Decorators promote code reuse and maintainability by separating cross-cutting concerns and promoting modular and composable code.

The power of decorators lies in their ability to modify function behavior transparently without modifying the existing code which results in cleaner and highly reusable code.

Decorators enable a wide range of use cases, including logging, timing, caching, access control, validation, and much more.

- Retry decorators
- Timing decorators
- Timing decorators for async coroutines
- Caching decorators
- Logging decorators
- Send email decorators

### Data Preprocessing

This decorator handles data preprocessing tasks such as scaling and feature extraction before passing the data to your function [4].

```py
    def preprocess_data(func):
        def wrapper(*args, **kwargs):
            data = args[0]  # Assuming the first argument is the data
            # Data preprocessing code here
            preprocessed_data = data  # Replace with actual preprocessing logic
            return func(preprocessed_data, **kwargs)
        return wrapper

    @preprocess_data
    def train_model(data, learning_rate=0.01):
        # Training code here
        pass
```

### Experiment Tracking

This decorator logs experiment details including hyperparameters and performance metrics [4].

```py
    def track_experiment(experiment_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                experiment_details = {
                    'name': experiment_name,
                    'hyperparameters': kwargs,
                    'performance_metrics': result  # Replace with actual metrics
                }
                # Log experiment details to a tracking system (e.g., MLflow)
                print(f"Logged experiment: {experiment_details}")
                return result
            return wrapper
        return decorator

    @track_experiment('experiment_1')
    def train_model(data, learning_rate=0.01, batch_size=32):
        # Training code here
        pass
```

### Logging

This decorator logs function inputs, outputs, and exceptions [4].

```py
    import logging

    def log_function(func):
        logging.basicConfig(filename='ml_engineer.log', level=logging.INFO)

        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                logging.info(f"{func.__name__}({args}, {kwargs}) returned {result}")
                return result
            except Exception as e:
                logging.error(f"{func.__name__}({args}, {kwargs}) raised an exception: {e}")
                raise

        return wrapper

    @log_function
    def train_model(data, epochs=10):
        # Training code here
        pass
```

### Memoization

Memoization caches the results of expensive function calls and reuses them when the same inputs occur again which can drastically improve the efficiency of your ML pipelines [4].

```py
    def memoize(func):
        cache = {}

        def wrapper(*args):
            if args not in cache:
                cache[args] = func(*args)
            return cache[args]

            return wrapper
```

### Model Persistance

After a model has been trained, this decorator automatically saves the trained model to a specified file path [4].

```py
    import joblib

    def save_model(model_path):
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                model = args[0]  # Assuming the first argument is the trained model
                joblib.dump(model, model_path)
                return result
            return wrapper
        return decorator

    @save_model('my_model.pkl')
    def train_model(data, epochs=10):
        # Training code here
        pass
```

### Retry

In ML, we often deal with external data sources or external APIs.

This decorator retries a function a specified number of times if it fails [4].

```py
    import random

    def retry(max_retries):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for _ in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        print(f"Error: {e}")
                        wait_time = random.uniform(0.1, 1.0)
                        time.sleep(wait_time)
                raise Exception(f"Max retries ({max_retries}) exceeded.")
            return wrapper
        return decorator

    @retry(max_retries=3)
    def fetch_data():
        # Data fetching code here
        pass
```

### Timing

Timing your code is crucial in ML, especially when optimizing algorithms [4].

The following decorator calculates the execution time of a function.

```py
    import time

    def timing(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} took {end - start} seconds to run.")
            return result

        return wrapper

    @timing
    def train_model(data):
        # Training code here
        pass
```

### Input Validation

This decorator adds input validation to your functions to make sure you are working with the correct data types [4].

```py
    def validate_input(*types):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for i, arg in enumerate(args):
                    if not isinstance(arg, types[i]):
                        raise TypeError(f"Argument {i+1} should be of type {types[i]}")
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @validate_input(int, list)
    def train_model(iterations, data):
        # Training code here
        pass
```

### Parameter Validation

This decorator ensures that the hyperparameters passed to your functions are within acceptable ranges [4].

```py
    def validate_hyperparameters(param_ranges):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for param, value in kwargs.items():
                    if param in param_ranges:
                        min_val, max_val = param_ranges[param]
                        if not (min_val <= value <= max_val):
                            raise ValueError(f"{param} should be between {min_val} and {max_val}")
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @param_validate({'learning_rate': (0.001, 0.1), 'batch_size': (16, 128)})
    def train_model(data, learning_rate=0.01, batch_size=32):
        # Training code here
        pass
```

### Performance Profiling

This decorator profiles your code and provides insights into its execution which is crucial for optimization [4].

```py
    import cProfile

    def profile_performance(func):
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            result = profiler.runcall(func, *args, **kwargs)
            profiler.print_stats()
            return result
        return wrapper

    @profile_performance
    def train_model(data, epochs=10):
        # Training code here
        pass
```


## Model Runtime and Debugging

The article [3] discusses how to define and apply function wrappers for profiling machine learning model runtime for a simple classification model.

- We will use this function wrapper to monitor the runtime of the data preparation, model fit and model predict steps in a simple machine learning workflow.

- We will also see how to define and apply function wrappers for debugging these same steps.

The `functools` module in Python makes defining custom decorators easy which can “wrap” (modify/extend) the behavior of another function.

The process for defining timer and debugger function wrappers follows similar steps.

### Define a timer function wrapper

```py
    def runtime_monitor(input_function):
      @functools.wraps(input_function)
      def runtime_wrapper(*args, **kwargs):
         start_value = time.perf_counter()
         return_value = input_function(*args, **kwargs)
         end_value = time.perf_counter()
         runtime_value = end_value - start_value
         print(f"Finished executing {input_function.__name__} in {runtime_value} seconds")
     return return_value
```

We can then use `runtime_monitor` to wrap our data_preparation, fit_model, predict, and model_performance functions.

```py
    @runtime_monitor
    def fit_model(X_train,y_train):
       model = RandomForestClassifier(random_state=42)
       model.fit(X_train,y_train)
       return model

       model = fit_model(X_train,y_train)
       # Finished executing fit_model in 0.545468124000763 seconds

    @runtime_monitor
    def predict(X_test, model):
       y_pred = model.predict(X_test)
       return y_pred

       y_pred = predict(X_test, model)
       # Finished executing predict in 0.05903794700134313 seconds

    @runtime_monitor
    def model_performance(y_pred, y_test):
       print("f1_score", f1_score(y_test, y_pred))
       print("accuracy_score", accuracy_score(y_test, y_pred))
       print("precision_score", precision_score(y_test, y_pred))

       model_performance(y_pred, y_test)

    # f1_score 0.5083848190644307
    # accuracy_score 0.7604301075268817
    # precision_score 0.5702970297029702
# Finished executing model_performance in 0.0057055420002143364 seconds
```

### Define a debugger function wrapper

```py
    def debugging_method(input_function):
        @functools.wraps(input_function)
        def debugging_wrapper(*args, **kwargs):
            arguments = []
            keyword_arguments = []
            for a in args:
               arguments.append(repr(a))
            for key, value in kwargs.items():
               keyword_arguments.append(f"{key}={value}")
            function_signature = arguments + keyword_arguments
            function_signature = "; ".join(function_signature)
            print(f"{input_function.__name__} has the following signature: {function_signature}")
            return_value = input_function(*args, **kwargs)
            print(f"{input_function.__name__} has the following return: {return_value}")
            return return_value
            return debugging_wrapper
```

We can now wrap our functions with the `debugging_method`:

```py
    @debugging_method
    @runtime_monitor
    def fit_model(X_train,y_train):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train,y_train)
        return model

        model = fit_model(X_train,y_train)
```

## References

[1]: [5 real handy python decorators for analyzing/debugging your code](https://towardsdatascience.com/5-real-handy-python-decorators-for-analyzing-debugging-your-code-c22067318d47)

[2]: [12 Ways to Use Function Decorators to Improve Your Python Code](https://medium.com/geekculture/12-ways-to-use-function-decorators-to-improve-your-python-code-f35515a45e3b)

[3]: [Python decorators: 5 + 1 useful decorators to adopt immediately](https://itnext.io/python-decorators-5-1-useful-decorators-to-adopt-immediately-1594e8d438e4)

[4]: [Enhancing Efficiency: 10 Decorators I Use Daily as a Tech MLE](https://mindfulmodeler.hashnode.dev/enhancing-efficiency-10-decorators-i-use-daily-as-a-tech-mle)
