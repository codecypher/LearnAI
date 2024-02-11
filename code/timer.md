# Execution Times in Python


## Timer Decorator

Here is a decorator that we can easily apply to real-world problems when debugging code [1].

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



## Logging Method Execution Time in Python

```py
import logging
import time

from functools import wraps

logging.basicConfig()
logger = logging.getLogger("my-logger")
logger.setLevel(logging.DEBUG)


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

        return wrapper
```

```py
from timer import timed

@timed
def my_printer(max):
    sum = 0
    for i in range(max):
        sum += i
if __name__ == "__main__":
my_printer(100000000)
```



## Building a custom Timer class

Since Python 3.7, many functions in the time module with the suffix `_ns` return integer nanoseconds while the original float (seconds) versions can be called using the functions of the same name without the suffix [2]. 

#### Use perf_counter for precision

The `time.perf_counter_ns` function is the preferred way to time execution. However, absolute value is meaningless since only the subtraction of times makes sense.

```py
import time

start_counter_ns = time.perf_counter_ns()
do_complex_stuff()
end_counter_ns = time.perf_counter_ns()

timer_ns = end_counter_ns - start_counter_ns
print(timer_ns)
```

#### Use process_time for CPU time

Sometimes we want to measure the CPU time of the process, regardless of thread sleep (other processes doing other things). In such cases, we should use `time.process_time_ns`. 

```py
import time

start_time_ns = time.process_time_ns()
do_complex_stuff()
end_time_ns = time.process_time_ns()

timer_ns = end_time_ns - start_time_ns
print(timer_ns)
```

#### Use a monotonic clock for long processes

When timing longer processes, system time changes can occur such as an NTP sync service or daylight saving time.

Therefore, we need a clock that does not change even when the system time is changed called `time.monotonic_ns` which is suitable for timing longer processes. 

```py
import time

start_time_ns = time.monotonic_ns()
do_complex_stuff()
end_time_ns = time.monotonic_ns()

timer_ns = end_time_ns - start_time_ns
print(timer_ns)
```

#### Disable garbage collector for accurate timing

One of the features of Python is automatic garbage collection which is basically memory deallocation for unused objects.

Garbage collection can impact execution times, so it can happen during the timing of your code block and the time it consumes is not negligible. 

Therefore, it is better to disable GC while measuring time. 

#### Custom Timer class

We can create a custom `Timer` class by using what we now know about the garbage collector, the different clocks we can use, and the need for a more Pythonic timer [2].

Here is what we want from the timer [2]:

- the possibility of turning off garbage collection

- to select the type of clock we want

- use time in nanoseconds except when specifically converting to seconds

```py
import time
import gc
from typing import Literal, Optional, NoReturn


class Timer:
    """
    timer type can only take the following string values:
    - "performance": the most precise clock in the system.
    - "process": measures the CPU time, meaning sleep time is not measured.
    - "long_running": it is an increasing clock that do not change when the
        date and or time of the machine is changed.
    """

    _counter_start: Optional[int] = None
    _counter_stop: Optional[int] = None

    def __init__(
        self,
        timer_type: Literal["performance", "process", "long_running"] = "performance",
        disable_garbage_collect: bool = True,
    ) -> None:
        self.timer_type = timer_type
        self.disable_garbage_collect = disable_garbage_collect

    def start(self) -> None:
        if self.disable_garbage_collect:
            gc.disable()
        self._counter_start = self._get_counter()

    def stop(self) -> None:
        self._counter_stop = self._get_counter()
        if self.disable_garbage_collect:
            gc.enable()

    @property
    def time_nanosec(self) -> int:
        self._valid_start_stop()
        return self._counter_stop - self._counter_start  # type: ignore

    @property
    def time_sec(self) -> float:
        return self.time_nanosec / 1e9

    def _get_counter(self) -> int:
        counter: int
        match self.timer_type:
            case "performance":
                counter = time.perf_counter_ns()
            case "process":
                counter = time.process_time_ns()
            case "long_running":
                counter = time.monotonic_ns()
        return counter

    def _valid_start_stop(self) -> Optional[NoReturn]:
        if self._counter_start is None:
            raise ValueError("Timer has not been started.")
        if self._counter_stop is None:
            raise ValueError("Timer has not been stopped.")
            return None
```

To instantiate the class, we need to specify the timer_type: if we will use the performance clock, process clock, or the long-running clock (monotonic). We can also determine whether to disable garbage collection.

```py
import time

timer = Timer(timer_type="long_running")

timer.start()
do_complex_stuff()
timer.stop()

print(timer.time_nanosec, timer.time_sec)
```


#### Timing context manager

The context manager is one of Python’s top syntactic sugar features such as when we read code with the keyword “with”. 

Simple context managers are not hard to create, especially when using the standard module named `contextlib`.

So we can create a context manager that uses our Timer class and takes care of starting and stopping it. 

```py
from contextlib import contextmanager
from typing import Literal

@contextmanager
def timing(
    timer_type: Literal["performance", "process", "long_running"] = "performance"
):
    timer = Timer(timer_type=timer_type)
    try:
        timer.start()
        yield timer
    finally:
    timer.stop()
```

Then we can use it to time a block of code. 

```py
with timing() as timer:
    do_complex_stuff()
    print(timer.time_sec)
```


## References

[1] [5 real handy python decorators for analyzing/debugging your code](https://towardsdatascience.com/5-real-handy-python-decorators-for-analyzing-debugging-your-code-c22067318d47)

[2] [Execution Times in Python](https://towardsdatascience.com/execution-times-in-python-ed45ecc1bb4d)


