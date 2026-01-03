# Python Code Snippets

Here are some useful python code snippets.

## Python one-liners

Here are some useful python one-liners that can save time [1]:

```py
    # Palindrome Python One-Liner
    phrase.find(phrase[::-1])

    # Swap Two Variables Python One-Liner
    a, b = b, a

    # Sum Over Every Other Value Python One-Liner
    sum(stock_prices[::2])

    # Read File Python One-Liner
    [line.strip() for line in open(filename)]

    # Factorial Python One-Liner
    reduce(lambda x, y: x * y, range(1, n+1))

    # Performance Profiling Python One-Liner
    python -m cProfile foo.py

    # Superset Python One-Liner
    lambda l: reduce(lambda z, x: z + [y + [x] for y in z], l, [[]])

    # Fibonacci Python One-Liner
    lambda x: x if x<=1 else fib(x-1) + fib(x-2)

    # Quicksort Python One-liner
    lambda L: [] if L==[] else qsort([x for x in L[1:] if x< L[0]]) + L[0:1] + qsort([x for x in L[1:] if x>=L[0]])

    # Sieve of Eratosthenes Python One-liner
    reduce( (lambda r,x: r-set(range(x**2,n,x)) if (x in r) else r), range(2,int(n**0.5)), set(range(2,n)))
```

## More Python one-liners

```py
    # swap two variables
    a,b = b,a

    # reverse list
    lst = [2,3,22,4,1]
    lst[::-1]

    # find square of even numbers with list comprehension
    result2 = [i**2 for i in range(10) if i%2==0]
    print(result2)

    # Dictionary comprehension
    myDict = {x: x**2 for x in [1,2,3,4,5]}
    print(myDict)

    # lambda function to square a number
    sqr = lambda x: x * x
    sqr(10)

    # Read file contents into a list: one-liner
    file_lines = [line.strip() for line in open(filename)]

    # convert binary number to int
    n = '100' ##binary 100
    dec_num = int(n,base = 2)
    print(dec_num)

    from itertools import combinations
    print(list(combinations([1, 2, 3, 4], 2)))

    from itertools import permutations
    print(list(permutations([1, 2, 3, 4], 2)))


    # Find longest string
    words = ['This', 'is', 'a', 'list', 'of', 'keyword']
    print(max(words, key=len))
```

## References

[1]: [Learn Python By Example: 10 Python One-Liners That Will Help You Save Time](https://medium.com/@alains/learn-python-by-example-10-python-one-liners-that-will-help-you-save-time-ccc4cabb9c68)
