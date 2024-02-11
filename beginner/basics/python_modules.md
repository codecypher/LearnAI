# Python Modules and Packages

## Modules

A _module_ is a file containing Python definitions and statements which is a way to organize independent code blocks

### Saving code in a separate file

Saving the functions in a script and run the script during the project has some limitations:

- It will  create all the variables and functions available inside the script, even if you needed only a select few.

- The objects created will occupy the namespace and may lead to name conflict or unintended consequences.

- We run rhe risk if running malicious code.  

Using a module helps to overcome these limitations. 

### Namespace

The _namespace_ is a memory space allocated on the computer where object names exist. Names are unique inside a namespace. 

Using modules is a great way to keep the namespace separate and clean.

When a module is used properly, Python makes all the names from the module available but does not make them available to the code’s namespace.


### How to Use a Module

We create a Python script called `messingAround.py` and save it in the same directory where we will run our Python program. 

```py
    # messingAround module
    pi = 6.38
    def makeHalf(val):
        return val/2
```


### Ways to Import

To make a module available, we must import it using the keyword import. There are different ways to import modules. 

```py
    # import the entire module
    # This make all the components of the module available to call
    # but will not load them as Python objects.
    import messingAround

    # load everything from a module (not recommended)
    # creates global variables
    from messingAround import *

    # import specific component from a module
    from messingAround import pi

    # give aliases to the module or components
    import messingAround as ma
```

### Demo


```py
    import messingAround as ma
    import math as m

    pi = m.pi * 3
    print("My pi value = {} \nmessingAround's pi value = {} \nMath module's pi value = {}".format(pi, ma.pi, m.pi))

    # My pi value = 9.42477796076938 
    # messingAround's pi value = 6.28 
    # Math module's pi value = 3.141592653589793
```

Thw Math module has a variable called pi that contains the value for π. We also have a variable with the same name but different value in our module. In addirion, we created a custom pi variable using the pi variable from math module.


### Peek inside a Module

We may need to check the module after importing it. 

The `dir()` function displays the components in alphabetical order.

```py
    dir(ma)
```


### How to Control Imported Packages

If `__all__` is specified in the module, only the names that are explicitly stated will be exported.

```py
    __all__ = ['numpy_prep', 'pd_prep']
```

```py
    from module import *
```


### How to Execute Package as main entry function

```py
    if __name__ == "__main__" :
```

----------


## Packages

Python packages are collections of modules. 

If Python modules are the home of functions and variables, packages are the homes of modules.

### Modules vs Packages

As the codebase grows,  we need to group functions into multiple modules based on the types of tasks they perform which usaually means we want to organize the modules in a directory form that can be turned into Python Package for easier use.

Once a package is installed, we can easily access to the modules stored in different directory levels using a _dot notation_.

### Package Example

Suppose we are working on a project where we get user names separated by a specific character stored in a long string, and need to break them down and assign them with unique numeric IDs so that they can be stored in a database in the future. 

We have teo tasks tasks:

- Break the input string into a set list of names. 
- Assign unique IDs to the names.


Ee will create two modules: stringProcess and idfier.

- stringProcess: A module to process strings which contains one function `stringSplit()` that splits a string based on the given separator.

- idfier: A module to create unique IDs whicj contains `randomIdfier()` that takes a list of name, assigns them randomly generated unique IDs, and saves them as a dictionary.


We create two separate .py files with the codes below and name the files by the module names. 

```py
    # stringProcess module
    print("stringProcess module is initialized")
    def stringSplit(string, separator):
        outputList = string.split(sep=separator)
        return outputList
```

```py
    # idfier module
    print("idfier module is initialized")
    import random as rn

    def randomIdfier(ids):
        outputDict = {}
        for id in ids:
            outputDict[id] = rn.randint(1000, 9999)

            return outputDict
```

### Useful Module Properties for Package Building

We have discussed general module properties in the last article. 


Here we discuss some additional properties that will help with package development.

#### Module Initialization

Once a module is imported, Python implicitly executes the module and initializes some of aspects of the module. 

> A module initialization takes place only once in a project.

If a module is imported multiple times, Python remembers the previous import and silently ignores the following initializations.


### Private Properties in Modules

We may need to have variables inside the Modules that are only intended for internal which are considered private properties. 

We can declare such properties (such as variables or functions), by one or two underscores. 

> Adding underscores is merely a convention and doesn't impose any protection per se.

#### __name__ Variable

Modules are essentially Python scripts. 

When Python scripts are imported as Modules, Python creates a variable called `__name__` tththat stores the name of the module.  

When a script is directly executed, the `__name__` variable contains `__main__ `. 


```py
    name = "stringProcess"
    if __name__ == "__main__":
        print(name, "is used as a script")
    else:
        print(name, "is used as a module")
```

In the code cell below we called `stringProcess` module both as a script and a module. 

```
%run -i stringProcess.py

import stringProcess

# stringProcess is used as a script
# stringProcess is used as a module
```

#### How can we use `__name__` variable?

The `__name__` variable can be a very useful feature to run some primary tests on code inside the module script.

Using this variable’s stored value, we can run some tests during the development mode when we are actively working with the script and ignore them when they are used as modules in a project. 


### Python Search for Modules

In real projects, we would like to keep our modules and packages in a separate location, so we copy our two modules to a different folder called `Silly_Anonymizer`.

> Python maintains a list of locations or folders in path variable from sys module where Python searches for modules.

Python searches the locations inside `sys.path` in the order they are stored in the list starting with the location where the scrip is executing. 


### Build a Package

First we place our modules inside `Silly_Anonymizer`. 

Next, we create two sub-directories inside `Silly_Anonymizer` named `NonStringOperation` and `StringOperation`. 

We move our two modules inside these two sub-directories: idfier.py inside NonStringOperation and stringProcess.py inside StringOperation. 


### Initialize a Package

Similat to Mmdules, Ppckages also need an initializer. 

We can include a file called `__init__.py` in the root directory of `Silly_Anonymizer`. H

Since a package is not a file  we cannot call of a function for initialization, so this file is used for initialization. 

The `__init__.py` can be left empty but it needs to be present at the root directory of a module directory to be considered a Package.

We can have `__init__.py` files in other sub-folders as well. 


### Import Module from Package

After we have `__init__.py` file in the module home directory, we are can use it as a Package using a fully qualified path from the root of the package. 

```py
    import Anonymizer.StringOperation.stringProcess as sp
    # stringProcess is used as a module

    sp.stringSplit(string="Arafath, Samuel, Tiara, Nathan, Moez", separator=",")
    # ['Arafath', ' Samuel', ' Tiara', ' Nathan', ' Moez']
```

> Python packages can be imported from zip folders. 

### Publish a Package

We have built our package and now it is ready to be used locally. 

We can also make it available for others using Python Packaging Index (PyPi) which is the most commonly used repository to host Python packages.




## References

[Use Modules to Better Organize your Python Code](https://levelup.gitconnected.com/use-modules-to-better-organize-your-python-code-75690ba6b6e?gi=2f486f522813)

[From Functions to Python Package](https://betterprogramming.pub/from-functions-to-python-package-f8a3bba8bb6b)

[Demystifying Modules and Packages in Python](https://betterprogramming.pub/demystifying-modules-and-packages-in-python-17ed869316c8?gi=71014fe601f0)

