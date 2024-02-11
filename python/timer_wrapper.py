#!/usr/bin/env python3
"""
  timer.py
  Timer wrapped as a decorator
"""
import time
import requests

# Code timer as a decorator 
def timerWrapper(func):
    """
    Code the timer
    """
    def timer(*args, **kwargs):
        """
        Start timer
        """
        start = time.perf_counter()
        
        output = func(*args, **kwargs)
        
        timeElapsed = time.perf_counter() - start
        print(f"Current function: {func.__name__}\n  Run Time: {timeElapsed}")
        return output

    return timer

# Func to make a request to given url
@timerWrapper
def getArticle(url):
    return requests.get(url, allow_redirects=True)


# Monitor the runtime
if __name__ == "__main__":
    getArticle('https://towardsdatascience.com/6-sql-tricks-every-data-scientist-should-know-f84be499aea5')

