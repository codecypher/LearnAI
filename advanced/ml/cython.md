# Cython

## Getting started with Cython

Creating a Cython package has some enormous benefits but it also takes a bit more effort than your regular Python programming. Think about the following before Cythonizing every last line of your code.

- Make sure your code is slow for the right reason, but we cannot write code that waits faster.

- If concurrency the problem, it may help using threads (such as waiting for an API) or running code in parallel over multiple CPUs (multiprocessing)?

- Make sure to use a virtual environment.


----------



## References

[1] [Why Python is so slow and how to speed it up](https://towardsdatascience.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)

[2] [Getting started with Cython](https://towardsdatascience.com/getting-started-with-cython-how-to-perform-1-7-billion-calculations-per-second-in-python-b83374cfcf77)

[3] [Cython for Absolute Beginners using CythonBuilder](https://towardsdatascience.com/cython-for-absolute-beginners-30x-faster-code-in-two-simple-steps-bbb6c10d06ad)

