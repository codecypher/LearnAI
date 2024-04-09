# Concurrency and Parallelism

**Concurrency** is about dealing with lots of things at once.

**Parallelism** is about _doing_ lots of things at once.

Concurrency is about structure whereas parallelism is about execution.

Concurrency provides a way to structure a solution to solve a problem that may (but not necessarily) be parallelizable.


## Concurrency vs Parallelism

The goal of concurrency is to prevent tasks from blocking each other by switching among them whenever one is forced to wait on an external resource.

Example: handling multiple network requests.

The better way would be to launch every request simultaneously and switch among them as we receive the responses. Thus, we eliminate the time spent waiting for the responses.


Parallelism is maximizing the use of resources by launching processes or threads that make use of all the CPU cores of the computer.

For parallelization, we would split the work among all the workers, so that the work will be done faster and each worker will do less work.


- Concurrency is best for tasks that depend on external resources such as I/O (shared resources).

- Parallelism is best for CPU-intensive tasks.


There are two kinds of concurrency:

1. **Threading:** the CPU switches between different threads really fast, giving the appearance of concurrency.

Keypoint: only one thread is running at any given time. When one thread is running, others are blocked.

You can think of it as a priority queue. Threads can be scheduled. The CPU scheduler can give each thread a certain amount of time to run, pause them, pass data to other threads, then give them different priorities to run at a later time.

Threading is used extensively in servers: thousands of clients can request something at the same time, then getting what they requested at a later time.

Philosophy: Do different things together, but it does not reduce the total time.

2. Parallelism: threads are running parallel, usually in different CPU core, true concurrency.

Keypoint: mlutiple threads are running at any given time which is useful for heavy computations
and super long running processes. We can split data into sections for each machine to compute and pool them together at the end, but different machines/cores are hard to interact with each other.

Philosophy: do one thing in less time.



## Concurrency and Parallelism in Python

Python provides us with mechanisms to implement concurrency and parallelism

For concurrency, we have _multithreading_ and _async_.

For parallelism, we have _multiprocessing_.



## Types of Parallelization

Parallelization is possible in two ways:

- Multithreading: Using multiple threads of a process/worker

- Multiprocessing: Using multiple processors

Multithreading is useful for I/O bound applications such as when we have to download and upload multiple files.

Multiprocessing is useful for CPU-bound applications.


Here is anexample of one use case for multiprocessing.

Suppose we have 1000 images saved in a folder and for each image we need to perform the following operations:

- Convert image to grayscale
- Resize the grayscale image to a given size
- Save the modified image in a folder

Doing this process on each image is independent of each other -- processing one image would not affect any other image in the folder.

Therefore, multiprocessing can help us reduce the total time.

The total time will be reduced by a factor equal to the number of processors we use in parallel. This is one of many examples where you can use parallelization to save time.



----------


## Guide to Asyncio, Threading, and Multiprocessing

The article [3] discusses what it means for a task to run async vs in a separate thread vs in a separate process.

### Synchronous execution

There is nothing special in synchronous execution.We run the code as usual: one thing happens and then another thing happens.

Only one function can run at a time and only when it is finished, something else is allowed to happen.

```py
def do_first():
    print("Running do_first")
    ...

def do_second():
    print("Running do_second")
    ...

def main():
    do_first()
    do_second()

if __name__ == "__main__":
    main()
```

### Asynchronous execution (async)

In async we run one block of code at a time but we cycle which block of code is running.

The program needs to be built around async but we can call normal (synchronous) functions from an async program.

Here is a list of what you need in order to make your program async:

- Add async keyword in front of your function declarations to make them awaitable.

- Add await keyword when you call your async functions (without it they won’t run).

- Create tasks from the async functions you wish to start asynchronously. Also wait for their finish.

- Call asyncio.run to start the asynchronous section of your program.

Here is an example of a simple async program that runs two functions asynchronously:

```py
import asyncio

async def do_first():
    print("Running do_first block 1")
    ...

    # Release execution
    await asyncio.sleep(0)

    print("Running do_first block 2")
    ...

async def do_second():
    print("Running do_second block 1")
    ...

    # Release execution
    await asyncio.sleep(0)

    print("Running do_second block 2")
    ...

async def main():
    task_1 = asyncio.create_task(do_first())
    task_2 = asyncio.create_task(do_second())
    await asyncio.wait([task_1, task_2])

if __name__ == "__main__":
    asyncio.run(main())
```

Here is an example of the output:

```
Running do_first block 1
Running do_second block 1
Running do_first block 2
Running do_second block 2
```

We run the first block of code in the first function, then we give the execution back to the async engine which then ran the first block of code in the second function (while the first function was unfinished), then we again released the execution and then the final blocks of codes from each functions were run.

We ran the the blocks concurrently giving an impression of parallel execution.

### Concurrent execution (threading)

In threading, we execute one line of code at a time but we constantly change which line is run.

We use the threading library: we first create some threads, start them and then wait them to finish (such as using join).

Here is an example of a concurrent program:

```py
import threading

def do_first():
    print("Running do_first line 1")
    print("Running do_first line 2")
    print("Running do_first line 3")
    ...

def do_second():
    print("Running do_second line 1")
    print("Running do_second line 2")
    print("Running do_second line 3")
    ...

def main():
    t1 = threading.Thread(target=do_first)
    t2 = threading.Thread(target=do_second)

    # Start threads
    t1.start(), t2.start()

    # Wait threads to complete
    t1.join(), t2.join()

if __name__ == "__main__":
    main()
```

Here is an example of the output:

```
Running do_first line 1
Running do_second line 1
Running do_first line 2
Running do_second line 2
Running do_second line 3
Running do_first line 3
```

The code clearly rotated between the lines it executed from each of these functions. It ran lines of code from these functions a bit randomly.
However, the lines of code were not run at the same time which is not yet true parallelism.

### Parallel execution (multiprocessing)

In multiprocessing we actually run multiple lines of Python code at one time. We use multiple processes to achieve this.

To use multiprocessing, we need to: create processes, set them running and wait for them to finish (such aa using join).

```py
import multiprocessing

def do_first():
    print("Running do_first line 1")
    print("Running do_first line 2")
    print("Running do_first line 3")
    ...

def do_second():
    print("Running do_second line 1")
    print("Running do_second line 2")
    print("Running do_second line 3")
    ...

def main():
    t1 = multiprocessing.Process(target=do_first)
    t2 = multiprocessing.Process(target=do_second)

    # Start processes
    t1.start(), t2.start()

    # Wait processes to complete
    t1.join(), t2.join()

if __name__ == "__main__":
    main()
```

This may ran the first function and then the second function but this is due to the fact that launching a process is expensive and the first one finished when starting the second one.

If these were longer running functions we would see that the functions were executed in parallel and really at the same time if we had free cores.

### Major Differences

Now that we know the options, we discuss a bit about the major differences

- **Synchronous vs others:** In synchronous execution, we can decide which order everything is run. In async, threading and multi-processing we leave it to the underlying system to decide.

- **Multiprocessing vs others:** Multiprocessing is the only one that is really runs multiple lines of code at one time. Async and threading sort of fakes it. However, async and threading can run multiple IO operations truly at the same time.

- **Asyncio vs threading:** Async runs one block of code at a time while threading just one line of code at a time. With async, we have better control of when the execution is given to other block of code but we have to release the execution ourselves.

Note that due to GIL (Global Interpreter Lock), only multiprocessing is truly parallelized.

### Comparison of concurrency approaches

Perhaps the simplest way to compare these approaches are IO bound vs CPU bound problems:

- IO bound problems: use async if your libraries support it and if not, use threading.

- CPU bound problems: use multi-processing.

- None above is a problem: you are probably just fine with synchronous code. You may still want to use async to have the feeling of responsiveness in case your code interacts with a user.

In addition, if the code has a chance to get stuck (infinite loop, for example), multiprocessing is the only one that can be reliably terminated. We cannot actually terminate threaded tasks and with async, if the code gets stuck and never calls await, async cannot terminate it either. However, if the stuck task does await, it can be terminated.

It is usually best to use async when possible as race conditions are harder to manage with multiprocessing and threading often causing hard-to-debug problems.

Race conditions are cases where the order of execution depends on timing of the operating system. For example, sometimes one of your threads might run faster than the other threads by pure chance which might change the behaviour of your program.

Race conditions can lead to bugs that are hard to reproduce and occur only rarely.

Here is a summary of the choices:

<div>
  <div>
        <img width="600" alt="Comparison of concurrency approaches" src="https://miro.medium.com/max/1400/1*uBLNq0ms_pDFDutRYdckAA.png" />
    </div>
  <div class="caption">Figure 1: Comparison of concurrency approaches</div>
</div>

### Pitfalls

There are some problems with each option that can cause unexpected difficulties or even make it impossible for you to use the option.

#### Async

Async has the problem that your code needs to be built with it and also that project libraries to support it.

For example, if you have a database query but your database client does not support async, it means you cannot run multiple queries at the same time losing the benefit over synchronous code.

Avoid combining async with threading since async is not thread safe.

#### Threading

Threading has the problem that if multiple threads operate on the same data, there is a risk of corruption or even crashing.

For example, if one thread closes the connection that another one is currently using, there will be a problem.

These sorts of problems are less common with async aince each block of code should return the execution only when it is safe.

You need to make sure that the functions are thread safe if they share data or objects which means using threading locks if needed.

#### Multiprocessing

It might sound that multiprocessing is the best for everything based on what we have discussed so far. However, there are many significant problems that come with multiprocessing:

- It is quite expensive to create processes

- We cannot share memory between processes (but we can pass data via pipes)

- It is not always trivial to serialize data for processes

The last is usually not discussed much but when it occurs it causes unusual errors that are often confusing to solve. For example, decorators may fail on Windows with PermissionError:

but on Linux, they often work fine. Also, passing decorated functions to Process as arguments fail in pickling. There are even some odd errors found  in newer Python versions but not in older.

Try to avoid using decorators with multiprocessing and try to remove the attributes that are not needed in the process for custom classes when using with multiprocessing:

```py
from multiprocessing import Process

class MyClass:

    def __init__(self):
        self.problematic_attr = open("file.txt")

    def __getstate__(self):

        # Copy object's data
        data = self.__dict__.copy()

        # Remove attributes that cannot
        # be pickled here
        data.pop('problematic_attr')

        return data

def do_things(arg):
    ...

if __name__ == "__main__":
    p = Process(target=do_things, args=(MyClass(),))
    p.start()
    p.join()
```

In the code block above, we override `__getstate__` that handles the pickling of the instance in order to remove an attribute that we cannot pickle (file buffers cannot be pickled).

Note that child processes run the imports again so we should always use the `if __name__ == “__main__”:` block to start the program.

Also avoid doing heavy processing when importing and declaring globals, functions, and classes.


## References

R. H. Arpaci-Dusseau and A. C. Arpaci-Dusseau, Operating Systems: Three Easy Pieces, 2018, v. 1.01, Available online: https://pages.cs.wisc.edu/~remzi/OSTEP/


[1]: [Concurrency and Parallelism: What is the difference?](https://towardsdatascience.com/concurrency-and-parallelism-what-is-the-difference-bdf01069b081)

[2]: [Parallelize your python code to save time on data processing](https://towardsdatascience.com/parallelize-your-python-code-to-save-time-on-data-processing-805934b826e2)

[3]: [Practical Guide to Asyncio, Threading, and Multiprocessing in Python](https://itnext.io/practical-guide-to-async-threading-multiprocessing-958e57d7bbb8)

[4]: [Applying Python multiprocessing in 2 lines of code](https://medium.com/geekculture/applying-python-multiprocessing-in-2-lines-of-code-3ced521bac8f)
