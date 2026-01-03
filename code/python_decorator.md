# The Decorator Pattern

Here are some notes on Python decorators.

## Python decorator attributes

A _decorator_ is a function that takes another function as input, extends its behavior, and returns a new function as output [2].

A decorator wraps a function inside another function that adds extra functionality before or after the wrapped function is executed [5].

The decorator is useful for logging, access control, memoization, and function timing.

In a nutshell, decorators allow us to add behavior to functions without altering their core logic or structure [5].

This is possible because functions are first-class objects in Python which means they can be passed as arguments to functions and also be returned from functions just like other types of objects such as string, int, or float. Thus, a decorator can be used to decorate a function or a class.

Here we discuss three special decorators: @staticmethod, @classmethod, and @property which are “magical” decorators that can be very handy for our development work and make your code more clean [2].

### @staticmethod

A `static` method is a method that does not require the creation of an instance of a class.

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

## Using Decorators

Here are some useful topics on decorators discussed in [5]:

- Decorators with Arguments
- Chaining Multiple Decorators
- The Importance of functools.wraps

Practical Use Cases of Decorators

Some of the most common use cases for decorators:

- Logging: Track function calls, arguments, and results.

- Memoization: Cache function results to improve performance.

- Access Control: Restrict access to certain functions based on conditions (such as user permissions).

- Timing: Measure how long a function takes to execute.

## New Type Annotation Features

The improvement of type annotations in Python 3.11 can help to write bug-free code [3].

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

## Software Design

There are eight built-in Python decorators discussed in [4] that can help you write more elegant and maintainable code.

```pre
    @atexit.register
    @dataclasses
    @enum.unique
    @partial
    @singledispatch
    @classmethod
    @staticmethod
    @property
```

## References

[1]: [A Gentle Introduction to Decorators in Python](https://machinelearningmastery.com/a-gentle-introduction-to-decorators-in-python/)

[2]: [How to Use the Magical @staticmethod, @classmethod, and @property Decorators in Python](https://betterprogramming.pub/how-to-use-the-magical-staticmethod-classmethod-and-property-decorators-in-python-e42dd74e51e7?gi=8734ec8451fb)

[3]: [4 New Type Annotation Features in Python 3.11](https://betterprogramming.pub/4-new-type-annotation-features-in-python-3-11-84e7ec277c29)

[4]: [8 Built-in Python Decorators to Write Elegant Code](https://www.kdnuggets.com/8-built-in-python-decorators-to-write-elegant-code)

[5]: [Mastering Python Decorators: A Deep Dive Into Function Wrapping and Enhancements](https://medium.com/codex/mastering-python-decorators-a-deep-dive-into-function-wrapping-and-enhancements-2092d70ff26f)

----------

[Function Wrappers in Python: Model Runtime and Debugging](https://builtin.com/data-science/python-wrapper)

[The Python Decorator Handbook](https://www.freecodecamp.org/news/the-python-decorator-handbook/)
