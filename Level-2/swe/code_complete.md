# Code Complete

Steve McConnell, 2nd ed., 2004.


## Chapter 1 - Welcome to Software Construction

### 1.2 Why is Software Consruction Important?

Construction typically takes 30-80% of the total time spent on a project.

The productivity of individual programmers varied by a factor of 10 to 20 during construction.

Key Points


## Chapter 2 - Metaphors for a Richer Understanding of Software Development

List of Checklists

List of Tables

List of Figures

### 2.2 How to Use Software Metaphors

An _algorithm_ is a seft of well-defined instructions for carrying out a particular task. An algorithm is predictable, deterministic, and not subject to chance.

A _heuristic_ is a technique that helps you look for an answer. Its results are subject to chance because a heuristic tells you only how to look, not what to find.

An algorithm gives you the instructions directly. A heuristic tells you how to discover the instructions for yourself (or where to look for them).

### 2.3 Common Software Metaphors

As much as 90% of the development effort on a typical software system comes after its initial release, with two-thirds being typical.

The term _accretion_ means any growth or increase in size by a gradual external addition or inclusion (incremental, iterative, adaptive, or evolutionary).


## Chapter 3 - Measure Twice, Cut Once

### 3.1 Importance of Prerequisites

Paying attention to quality is also the best way to improve productivity.

The overarching goal of preparation is risk reduction.

Part of your job as a technical employee is to educate the nontechnical people around you about the development process.

The principle is to find an error as close as possible to the time at which it is introduced, Table 3-1.

Focusing on correcting defects earlier rather than later in a project can cut development costs and schedules by a factor of two or more.

### 3.2 Determine the Kind of Software You're Working On

Different kinds of software projects call for different balances between preparation and construction, Table 3-2.

One common rule of thumb is to plan to specify about 80% of the requirements up-front.

Another alternative is to specify only the most important 20% of the requirements up-front and plan to develop the rest of the software in small increments.

You might choose a more sequential (upfront) approach when...

You might choose a more iterative (as-you-go) approach when...

### 3.3 Problem-Definition Prerequisite

The first prerequisite you need to fulfill before beginning construction is a clear statement of the problem that the system is supposed to solve. This is sometimes called the vision statement or _mission statement_. Here it's called _problem definition_.

A problem definition defines what the problem is without any reference to possible solutions.

### 3.4 Requirements Prerequisite

The _requirements_ (requirements analysis, analysis, software requirements, or functional spec) describe in detail what a software system is supposed to do.

Paying attention to requirements helps to minimize changes to the system after development begins.

On large projects an error in requirements detected during the architecture stage is typically 3 times as expensive to correct as it would be if it were detected during the requirements stage. If detected during coding, it is 5-10 times as expensive, during system test 10 times, and during post-release 10-100 times as expensive.

The Myth of Stable Requirements: stable requirements are the holy grail of software development.

How much change is typical? The average project experiences about a 25% change in requirements during development.

Here are several things you can do to make the best of changing requirements during construction.

Checklist: Requirements

### 3.5 Architecture Prerequisite

The _software architecture_ (system architecture) is the high-level part of software design, the frame that holds the more detailed parts of the design.

Why have architecture as a prerequisite? Because the quality of the architecture determines the conceptual integrity of the system.

Good architecture makes construction easy.

**Aim for the 80/20 rule:** specify the 20% of the classes that make up 80% of the system's behavior.

#### Typical Architectural Components

- Define the major building blocks in a program.

- Specify the major classes to be used.

- Describe the major files and table designs to be used.

- The major elements of the user interface should have been specified at requirements time.

- Describe a plan for managing scarce resources.

- Describe the approach to design-level and code-level security.

- If performance is a concern, performance goals should be specified in the requirements.

- Describe how the system will address growth (scalability).

- Specify error handling - Error Processing.

- Indicate the kind of fault tolerance expected.

- Demonstrate that the system is technically feasible (proof-of-concept).

- Indicate whether programmers should err on the side of overengineering or on the side of doing the simplest thing that works.

- Describe a strategy for handling change.

- Be a polished conceptual whole with few adhoc additions. The essential problem with large systems is maintaining their conceptual integrity.

- Clearly state the objectives of the architecture.

- Identify risky areas.

- Contain multiple views.

### Checklist: Architecture

### Key Points

### 3.6 Amount of Time to Spend on Upstream Prerequisites

A well-run project devotes about 10-20% of its effort and about 20-30% of its schedule to requirements, architecture, and up-front planning.

If the requirements are unstable on any project treat the requirements work as its own project.

Checklist: Upstream Prerequisites


## Chapter 4 - Key Construction Decisions

### 4.1 Choice of Programming Language

The programming language chosen affects productivity and code quality.

Programmers are more productive using a familiar language.

Programmers who have extensive experience with a programming language are more than three times as productive as those with minimal experience. Table 4-1.

The Sapir-Whorf hypothesis: Your ability to think a thought depends on knowing the words capable of expressing the thought.

### 4.2 Programming Conventions

In a complex program, architectural guidelines give the program structural balance and construction guidelines provide low-level harmony, articulating each class as a faithful part of a comprehensive design.

Before construction begins, spell out the programming conventions to use.

### 4.3 Your Location on the Technology Wave

The programming tools do not have to determine how you think about programming.

Programming _in_ a language limits the thoughts to constructs that the language directly supports. Programming _into_ a language means that you first decide what thoughts you want to express and then determine _how_ to express those thoughts using the tools provided.

### 4.4 Selection of Major Construction Practices

### Checklist: Major Construction Practices

### Key Points


## Chapter 5 - Design in Construction

### 5.1 Design Challenges

#### Design is a wicked problem

A _wicked problem_ can be clearly defined only by solving it or a part of it.

#### Design is a Heuristic Process

Because design is nonderministic, design techniques tend to be heuristics (rules of thumb) rather than repeatable processes that are guaranteed to produce predictable results.

### 5.2 Key Design Concepts

Software development is made difficult because of two different classes of problems: the essential and the accidental.

_Managing complexity_ is the most important technical topic in software development.

When a project reaches the point at which no one completely understands the impact that code changes in one area will have on other areas, progress grinds to a halt.

At the software-architecture level, the complexity of a problem is reduced by dividing the system into subsystems.

The goal of all software-design techniques is to break a complicated problem into simple pieces.

#### How to Attack Complexity

#### Desirable Characteristics of a Design

#### Levels of Design

Figure 5-2

##### Level 1: Software system

The entire system.

##### Level 2: Division into Subsystems or Packages

Identification of all major subsystems

Deciding how to partition the program into major subsytems and defining how each subsystem is allowed to use each other subsystem.

Rules about how the various subsystems can communicate.

Allow communication between subsystems only on a _need to know_ basis.

A good general rule is that a system-level diagram like Figure 5-5 should be an _acyclic graph_ (a program shouldn't contain any circular relationships).

Common Subsystems

##### Level 3: Division into Classes

Identifying all classes in the system

Details of the ways in which each class interacts with the rest of the system are specified as the classes are specified. In particular, the class's interface is defined.

Making sure all the subsystems have been decomposed to a level of detail fine enough that you can implement their parts as individual classes.

##### Level 4: Division into Routines

Dividing each class into routines (i.e. detail the class's private routines).

##### Level 5: Internal Routine Design

Laying out the detailed functionality of the individual routines. Typically left to the individual programmer working on an individual routine.

### 5.3 Design Building Blocks: Heuristics

Because design is nondeterministic, skillful application of an effective set of heuristics is the core activity in good software design.

#### Software's Primary Technical Imperative is managing complexity.

The subsections describe a number of heuristics (ways to think about a design that sometimes produce good design insights) to help manage complexity.

The interface to a class should reveal as little as possible about its internal workings.

Designing the class interface is an iterative process.

Information hiding is useful at all levels of design - Two Categories of Secrets.

Get into the habit of asking "What should I hide?".

#### Identify Areas Likely to Change

Accomodating changes is one of the most challenging aspects of good program design.

#### Anticipating Different Degrees of Change

Design the system so that the effect or scope of the change is proportional to the chance that the change will occur.

Identify the minimal subset of the program that might be of use to the user.

#### Keep Coupling Loose

The term _coupling_ describes how tightly a class or routine is related to other classes or routines. The goal is to create small, direct, visible, and flexible relations to other classes and routines - loose coupling. Try to create modules that depend little on other modules (Coupling Criteria and Kinds of Coupling).

Classes and routines are first and foremost intellectual tools for reducing complexity.

If they're not making your job simpler, they're not doing their jobs.

#### Look for Common Design Patterns - Table 5-1

#### Other Heuristics

#### Guidelines for Using Heuristics

One of the most effective guidelines is not to get stuck on a single approach. You don't have to solve the whole design problem at once - Figure 5-10.

### 5.4 Design Practices

This section describes design practice heuristics, steps you can take that often produce good results.

Design is an iterative process.

When you come up with a first design attempt that seems good enough, don't stop!

In many cases, solving the problem with one approach wil produce insights that will enable you to solve the problem using another approach that's even better.

Divide and Conquer

#### Top-Down and Bottom-Up Design Approaches

The key difference between top-down and bottom-up strategies is that one is a decomposition strategy and the other is a composition strategy. One starts from the general problem and breaks it into manageable pieces while the other starts with manageable pieces and builds up a general solution. Both approaches have strengths and weaknesses that you must consider.

The word _prototyping_ means writing the absolute minimum amount of throwaway code that's needed to answer a specific design question.

Used with discipline, prototyping is the workhorse tool a designer has to combat design wickedness.

How Much Design is Enough? - Table 5-2

Capturing You Design Work

## 5.5 Comments on Popular Methodologies

Strive for simplicity and iterate!

### Checklist: Design in Construction

### Key Points


## Chapter 6 - Working Classes

### 6.1 Class Foundations: Abstract Data Types (ADTs)

Treat yourself to the highest possible level of abstraction - p. 130.

Try to make the names of classes and access routines independent of how the data is stored, and refer to the abstract data type - p. 131.

Abstract data types form the foundation for the concept of classes. Think of a class as an abstract data type plus inheritance and polymorphism - p. 133.

### 6.2 Good Class Interfaces

Each class should implement only one ADT - p. 135.

Make interfaces programmatic rather than semantic when possible - p. 137.

Consider abstraction and cohesion together - p. 138

To separate the class interface from the class implementation, include a pointer to the class implementation within the class declaration - p. 140.

Avoid friend classes

Favor read-time convenience to write-time convenience.

Ways that a user of a class can break encapsulation semantically - p. 142.

Be careful to program to the interface and not through

Watch for coupling that is too tight.

### 6.3 Design and Implementation Issues

Containment (_has a_ Relationships)

Be critical of classes that contain more than seven data members (7 +/- 2).

Inheritance (_is a_ Relationships)

_Liskov Substitution Principle (LSP)_: subclasses must be usable through the base class interface.

In other words, all the routines defined in the base class should mean the same thing when they are used in each of the derived classes.

Inherited routines come in three basic flavors: abstract overridable, overridable, and non-overridable - Table 6-1.

Avoid deep inheritance trees - p.147

Multiple inheritance is useful primarily for defining mixins.

Inheritance tends to work _against_ the primary technical imperative - managing complexity.

Summary of when to use inheritance and when to use containment - p. 149.

Member Funtions and Data

Constructors

### 6.4 Reasons to Create a Class

The most important reason to create a class is to reduce complexity.

Passing lots of data around suggests that a different class organization might work better.

Classes to Avoid

### 6.5 Language-Specific Issues

### Checklist: Class Quality

### Key Points


## Chapter 7 - High-Quality Routines

What is a high-quality routine? p. 161

Hide sequences

It's a good idea to hide the order in which events happen to be processed - p. 165.

### 7.2 Design at the Routine Level

The term _cohesion_ refers to how closely the operations in a routine are related. The goal is to have each routine do one thing well and nothing else.

### 7.3 Good Routine Names

Sometimes the only probelm with a routine is that its name is wishy-washy.

The optimum length for a variable name is 9-15 characters.

### 7.4 How Long Can a Routine Be?

One screen or two pages of program listing (50-150 lines).

### 7.5 How to Use Routine Parameters

Code that sets up for a call to a routine or takes down after a call is an indication that the routine is not well designed - p. 179.

If you find yourself frequently changing the parameters to a routine, you should probably be passing the whole object rather than specific elements.

### 7.6 Special Consideration in the Use of Functions

A _function_ is a routine that returns a value. A _subroutine_ does not.

Use a function if the primary purpose of the routine is to return the value indicated by the name. Otherwise, use a procedure.

Setting the Functions' Return Value

### Checklist: High-Quality Routines

### Key Points


## Chapter 8 - Defensive Programming

In _defensive programming_, the main idea is that if a routine is passed bad data, it will not be hurt.

### 8.1 Protecting Your Program from Invalid Inputs

Garbage in, garbage out (GIGO)

There are three general ways to handle garbage in:

- Check the values of all data from external sources.

- Check the values of all routine input parameters.

- Decide how to handle bad inputs.

### 8.2 Assertions

A _assertion_ is code that is used during development that allows a program to check itself as it runs.

Assertions are especially useful in large, complicated programs and high-reliability programs.

Use assertions to document assumptions made in the code and to flush out unexpected conditions - p. 190.

The C++ _assert_ macro does not provide for text messages.

An example of an improved ASSERT - p. 191.

Guidelines for Using Assertions

### 8.3 Error-Handling Techniques

How do you handle errors that you expect to occur? p. 194

_Correctness_ means never returning an inaccurate result (no result is better than a wrong one).

_Robustness_ means always trying to do something that will allow the software to keep operating (even if it leads to results that are wrong sometimes).

Safety-critical applications tend to favor correctness to robustness.

Consumer applications tend to favor robustness to correctness.

Be careful to handle invalid parameters in consistent ways throughout the program.

In C++ it is not required to do anything with a function's return value. **Test the return value.**

### 8.4 Exceptions

Like inheritance, exceptions used judiciously can reduce complexity.

Suggestions for realizing the benefits of exceptions and avoiding the difficulties often associated with them - p. 199 and Table 8-1.

Always consider the full set of error-handling alternatives:

Handling the error locally, propagating the error by using an error code, logging debug information to a file, shutting down the system, or some other approach.

Consider whether your program really needs to handle exceptions.

### 8.5 Barricade Your Program to Contain the Damage Caused by Errors

Barricades are a damage-containment strategy.

One way to barricade for defensive programming purposes is to designate certain interfaces as boundaries for safe areas.

Check data crossing the boundaries of a safe area for validity, and respond sensibly if the data is invalid - Figure 8-2. This same approach can be used at the class level.

The use of barricades makes the distinction between assertions and error-handling clean-cut.

Routines that are outside the barricade should use error-handling. Routines inside the barricade should use assertions. Deciding which code is inside and which is outside the barricade is an architecure-level decision.

### 8.6 Debugging Aids

Exceptional cases should be handled in a way that makes them obvious during development and recoverable in production - offensive programming.

Some ways to program offensively - p. 206.

Plan to avoid shuffling debugging code in and out of a program - p. 207.

### 8.7 Determining How Much Defensive Programming to Leave in Production Code

Some guidelines for deciding which defensive programming tools to leave in the production code - p. 209.

### Checklist: Defensive Programming

### Key Points


## Chapter 9 - The Pseudocode Programming Process

The Pseudocode Programming Process (PPP) reduces the work required during design and documentation and improves the quality of both.

### 9.1 Summary of Steps in Building Classes and Routines

Steps in Creating a Class

Steps in Building a Routine

### 9.2 Pseudocode for Pros

Guidelines for using pseudocode effectively

Benefits you can expect from using this style of pseudocode

### 9.3 Construction Routines by Using the PPP

### 9.4 Alternatives to the PPP

### Checklist: The Pseudocode Programming Process

### Key Points


## Chapter 10 - General Issues in Using Variables

### 10.3 Guidelines for Initializing Variables

Improper data initialization is one of the largest sources of error in computer programming.

_The Principle of Proximity_ — Keep related actions together.

Use `final` or `const` when possible.

If memory is allocated in the constructor, it should be freed in the destructor.

Check input parameters for validity.

Use a memory-access checker to check for bad pointers.

Initialize working memory at the beginning of the program.

### 10.4 Scope

The code between references to a variable is a window of vulnerability.

Keep a variable live for as short a time as possible.

General Guidelines for Minimizing Scope

### 10.5 Persistence

The main problem with persistence arises when you assume that a variable has a longer persistence than it really does.

Steps to avoid this kind of probelm

### 10.6 Binding Time

The _binding time_ is the time at which the variable and its value are bound together.

Use the latest binding time possible

### 10.7 Relationship Between Data Types and Control Structures

### Checklist: General Considerations in Using Data Initializing Variables

### Key Points


## Chapter 11 - The Power of Variable Names

### 11.1 Considerations in Choosing Good Names

Names should be as specific as possible

A good name tends to express the what more than the how.

The Effect of Scope on Variable Names

Computed-Value Qualifiers in Variable Names - put the modifier at the end of the name.

Common Opposites in Variable Names

### 11.2 Naming Specific Types of Data

Naming Loop Indexes

Naming Status Variables

Naming Temporary Variables - Temporary variables are a sign that the programmer does not yet fully understand the problem.

Naming Boolean Variables

### 11.4 Informal Naming Conventions

Guidelines for a Language-Independent Convention

Differentiate between variable and routine names

The book uses Option 5

Identify global (g_) and member (m_) variables.

Identify type definitions (UPPERCASE) - typedefs and structs

Identify elements of enumerated types

Guidelines for Language-Specific Conventions

C and C++ Conventions - Table 11-3.

### 11.5 Standardized Prefixes

Standardized prefixes are composed of two parts: the user-defined type (UDT) abbreviation and the semantic prefix.

### 11.6 Creating Short Names That Are Readable

General Abbreviation Guidelines

Comments on Abbreviations

Over the lifetime of a system, programmers spend far more time reading than writing code.

### 11.7 Kinds of Names to Avoid

Avoid names that sound similar (homonyms)

Avoid names that are commonly misspelled

Do not differentiate variable names solely by capitalization

Avoid names containing hard-to-read characters

### Checklist: Naming Variables

### Key Points


## Chapter 12 - Fundamental Data Types

### 12.1 Numbers in General

### 12.2 Integers

The easiest way to prevent integer overflow is to think through each of the terms in an arithmetic expression and try to imagine the largest value each can assume.

### 12.3 Floating-Point Numbers

### 12.4 Characters and Strings

Avoid magic characters and strings

Watch for off-by-one errors

Strings in C

### 12.5 Boolean Variables

### 12.6 Enumerated Types

Anytime you see a numeric literal, ask whether it makes sense to replace it with an enumerated type.

Enumerated types are especially useful for defining routine parameters.

Use enumerated types as an alternative to boolean variables.

Check for invalid values

Define the first and last entries of an enumeration for use as loop limits.

Reserve the first entry in the enumerated type as invalid

Beware of pitfalls of assigning explicit values to elements of an enumeration.

### 12.7 Named Constants

Use named constants in data declarations

Avoid literals even safe ones

### 12.8 Arrays

Make sure all array indexes are within the bounds of the array

Consider using container classes that you can access sequentially before automatically choosing an array.

Check the end points of an array

Watch out for index cross-talk - Switching loop indexes is called index cross-talk.

### 12.9 Creating Your Own Types (Type Aliasing)

Guidelines for Creating Your Own Types

Create types with functionally-oriented names

Avoid type names that refer to the kind of computer data underlying the type.

Avoid predefined types

If there is any possibility that a type might change, avoid using predefined types anywhere but in typedef or type definitions.

### Checklist: Fundamental Data

### Key Points


## Chapter 13 - Unusual Data Types

### 13.1 Structures

Use structures to clarify data relationships

Structures bundle groups of related items together.

Use structures to simplify operations on blocks of data

You can combine related elements into a structure and perform operations on the structure.

Use structures to simplify parameter lists

You can simplify routine parameter lists by using structered variables. Careful programmers avoid bundling data any more than is logically necessary, and they avoid passing a structure as a parameter when only one or two fields from the structure are needed - they pass the specific fields needed instead (information hiding). Information is passed around on a need-to-know basis.

### 13.2 Pointers

Many common security problems, especially buffer overruns, can be traced back to erroneous use of pointers.

General Tips on Pointers

Working with pointers successfully requires a two-pronged approach:

1. Avoid installing pointer errors in the first place.

2. Detect pointer errors as soon after they are coded as possible.

#### Pointer Notes

Isolate pointer operations in routines or classes

Declare and define pointers at the same time

Assign a variable its initial value close to where it is declared.

Delete pointers at the same scope as they were allocated

Keep allocation and deallocation of pointers symmetric.

Check pointers before using them

Before you use a pointer in a critical part of your program, make sure the memory location it points to is reasonable.

Check the variable referenced by the pointer before using it

Use dog-tag fields to check for corrupted memory

A _tag field_ or _dog-tag_ is a field you add to a structure for the purpose of error checking.

Simplify complicated pointer expressions

If your code contains a complicated expression, assign it to a well-named variable to clarify the intent of the operation.

Delete pointers in linked lists in the right order

Make sure that you have a pointer to the next element in the list before you free the current one.

Shred your garbage

In C you can force errors related to using deallocated pointers to be more consistent by overwriting memory blocks with junk data right before they are allocated.

Set pointers to null after deleting or freeing them

A common type of pointer error is to use a pointer that has been deleted or freed, _dangling pointer_. This does not change the fact that you can read data pointed to by a dangling pointer, but you do ensure that writing data to a dangling pointer produces an error.

Check for bad pointers before deleting a variable

Keep track of pointer allocations

Write cover routines to centralize your strategy to avoiding pointer problems.

In C++ you can use SAFE_NEW and SAFE_DELETE.

Use a nonpointer technique

If you can think of an alternative to using a pointer that works, save yourself a few headaches and use it instead.

#### C++ - Pointer Pointers

Understand the difference between pointers and references

Use pointers for pass by reference parameters and use const references for pass by value.

Use auto_ptrs.

Get smart about smart pointers

#### C - Pointer Pointers

Use explicit pointer types rather than the default type

Avoid type casting

Follow the asterik rule for parameter passing

### 13.3 Global Data

Common Problems with Global Data

Inadvertent changes to global data - side-effects

Reentrant code problems with global data - multithreaded code

Using Access Routines Instead of Global Data

Anything you can do with global data you can do better with access routines.

How to Use Access Routines

Hide data in a class using static. Write routines that let you read and change the data.

### Checklist: Considerations in Using Unusual Data Types

## Key Points


## Chapter 14 - Organizing Straight-Line Code

When statements have dependencies that require you to put them in a certain order, take steps to make the dependencies clear - p. 348.

Make the program read from top to bottom rather than jumping around.

### Checklist: Organizing Straight-Line Code

### Key Points


## Chapter 15 - Using Conditionals

### 15.1 if Statements

Put the normal case after the if rather than after the else

Putting code that results from a decision as close as possible to the decision.

Test the else clause for correctness

Check for reversal of the if and else clauses

Simplify complicated tests with boolean function calls

Put the most commmon cases first

### 15.2 case Statements

Choosing the Most Effective Ordering of Cases

#### Tips for Using case Statements

- Keep the code associated with each case short.

- Use the default clause only to detect legitimate defaults

- Use the default clause to detect errors

#### Checklist: Using Conditionals

#### Key Points


## Chapter 16 - Controlling Loops

### 16.1 Selecting the Kind of Loo

The Kinds of Loops - Table 16-1

When to Use a Loop-With-Exit Loop

Consider these finer points when you use this kind of loop

- Put all the exit conditions in one place.

- Use comments for clarification.

- The loop-with-exit loop is a one-entry, one-exit, construct, and it is the preferred kind of loop control.

  It is easier to understand and more closely models the way people think. It is a good technique to have inyour toolbox - as long as you use it carefully.

When to Use a for Loop

- Use for loops for simple activities that do not require internal loop controls.

- Use them when the loop control involves simple increments or decrements.

- If you have a condition under which execution has to jump out of a loop, use a while loop instead.

- Do not change the index value to force it to terminate - use a while loop instead.

When to Use a foreach Loop

The foreach loop is useful for performing an operation on each member of an array or other container.

### 16.2 Controlling the Loop

Minimize the number of factors that affect the loop - simplify. Treat the inside of a loop as if it were a routine - keep as much of the control as possible outside the loop. Think of a loop as a black box.
Checking Endpoints

A single loop usually has three cases of interest: the first case, an arbitrarily selected middle case, and the last case. When you create a loop mentally run through thaese cases to make sure the loop does not have any off-by-one errors.

Inefficient programmers tend to experiment randomly until they find a combination that works.

### Checklist: Loops

### Key Points


## Chapter 17 - Unusual Control Structures

### 17.1 Multiple Returns from a Routine

Use a return when it enhances readability

Use _guard clauses_ (early returns or exits) to simplify complex error processing

Code that has to check for numerous error conditions before performing its nominal actions can result in deeply indented code and obscure the nominal case. The flow of the code is sometimes clearer if the erroneuous cases are checked first.

Minimize the number of returns in each routine

### 17.2 Recursion

Use recursion selectively

For most problems it produces massively complicated solutions. Try iteration instead.

Consider alternatives to recursion before using it.

Tips for Using Recursion

Summary of Guidelines for Using goto

### Checklist: Unusual Control Structures

### Key Points


## Chapter 18 - Table-Driven Methods

A table-driven method is a scheme that allows you to look up information in a table rather than using logic statements (if and case) to figure it out. Virtually anything you can select with logic statements, you can select with tables instead. In simple cases, logic statements are easier and more direct. As the logic chain becomes more complex, tables become increasingly attractive.

### 18.1 General Considerations in Using Table-Driven Methods

When you use table-driven methods, you have to address two issues: How to look up entries in the table, and What you should store in the table.

There are three ways to look up an entry in a table: direct access. indexed access, and stair-step access.

Fudging Lookup Keys

Table-Driven Approach

Rather than hard-coding routines for each of the 20 kinds of messages, you can create a handful of routines that print each kind of primary data types. You can describe the contents of each kind of message in a table (including the name of each field) and then decode each message based on the description in the table. Data tends to be more flexible than logic.

### Checklist: Table-Driven Methods

### Key Points


## Chapter 19 - General Control Issues

### 19.1 Boolean Expressions

Compare boolean values to true and false implicitly.

Break complicated tests into partial tests with new boolean variables.

Move complicated expressions into boolean functions

If you use the test only once, it might not seem worthwihile to put it into a routine, but it improves readability and makes it easier to understand.

Use decision tables to replace complicated conditions

In if statements, convert negatives to positives and flip-flop the code in the if and else clauses.

Apply DeMorgan's Theorems to simplify boolean tests with negatives - Table 19-1

Use parentheses to clarify boolean expressions

Writing Numeric Expressions in Number-Line Order

Guidelines for Comparisons to 0

### 19.4 Taming Dangerously Deep Nesting

Summary of Techniques for Reducing Deep Nesting

### 19.5 A Programming Foundation: Structured Programming

The core of structured programming is the simple idea that a program should use single-entry, single-exit control constructs.

The Three Components of Structured Programming

Sequence (assignments and calls to routines), Selection (if-then-else and switch-case), and Iteration (for and while loops).
The core thesis of structured programming is that any control flow can be created from these three constructs.

The use of any control structure other than these three constructs - break, continue, return, throw-catch, etc. - should be viewed with a critical eye.
The complexity of a program is largely defined by its control flow.

### How to Measure Complexity

Complexity is measured by counting the number of decision points in a routine (McCabe's cyclomatic complexity metric).

### Checklist: Control-Structure Issues

### Key Points


## Chapter 20 - The Software-Quality Landscape

### 20.1 Characteristics of Software Quality

Software has both external and internal quality characteristics.

A user of the software is aware of the external. Programmers care about the internal characteristics of the software as well as the external.
The internal aspects of system quality are the main subject of this book.
The attempt to maximize certain characteristics inevitably conflicts with the attempt to maximize others.

Finding an optimal solution from a set of competing objectives is one activity that makes software develpoment a true engineering discipline. However, sometimes focusing on a specific characteristic does not always mean a tradeoff with another - Figure 20.1.

### 20.3 Relative Effectiveness of Quality Techniques

If project developers are striving for a higher defect-detection rate, they need to use a combination of techniques - Table 20-2.

Only about 20% of the errors found by inspections are found by more than one person.

Code reading detects more interface defects and function testing detects more control defects - Table 20-3.

The longer a defect remains in the system, the more expensive it becomes to remove.

A recommended combination of testing techniques for achieving higher-than-average quality:

Formal inspections of all requirements, architecture, and designs for critical parts of a system

Modeling or prototyping

Code reading or inspections

Execution testing

### 20.5 The General Principle of Software Quality

The General Principle of Software Quality is that improving quality reduces development costs.

Defects creep into software at all stages.

The single biggest activity on most projects is debugging and correcting code that does not work properly.

### Checklist: A Quality-Assurance Plan

### Key Points


## Chapter 21 - Collaborative Construction

### 21.1 Overview of Collaborative Development Practices

The term _collaborative construction_ refers to pair programming, formal inspections, informal technical reviews and document reading, and other techniques where developers share responsibility for creating code and other work products.

All collaborative construction techniques are based on the idea that developers are blind to some trouble spots in their work so it is beneficial to have someone else look at their work.

The primary purpose of collaborative construction is to improve software quality.

The pair programming technique can achieve a code-quality level similar to formal inspections

It costs more - 10-25% more, but it reduces development time by 45%.

Technical reviews have been studied much longer than pair programming and the results have been impressive - p. 480.

Reviews create a venue for more experienced and less experienced programmers to communicate about technical issues.

In addition to being more effective at catching errors than testing, collaborative practices find different kinds of errors than testing.

A concept that spans all collaborative construction techniques is the idea of collective ownership of code.

Checklist: Effective Pair Programming

### 21.3 Formal Inspections

An inspection differs from a regular review in several key ways - p. 485.

Individual inspections catch about 60% of defects

Higher than other techniques except prototyping and high-volume beta testing.

The combination of design and code inspections removes 70-85% or more of defects.

The author of the code should anticipate hearing criticisms of defects that are not really defects and some that seem debatable.

The author should acknowledge each alleged defect and move on. The author should not try to defend the work under review. After the review, the author can think about each point in private and decide whether it is valid.

The reviewers must remember that the author has the ultimate responsibility for deciding what to do about a defect.

Checklist: Effective Inspections

### 21.4 Other Kinds of Collaborative Development Practices

A walk-through is a popular kind of review but the term is loosely defined.

In some sense, where two or three are gathered together, there is a walk-through - see p. 492.

Used unintelligently, walk-throughs are more trouble than they are worth.

The low end of their effectiveness 20% is not worth much. At least one organization found peer reviews of code to be extremely expensive.

Inspections seem to be more effective than walk-throughs at removing errors.

Inspections are more focused than walk-throughs and generally pay off better.

If you are choosing a review standard for your organization, choose inspections first unless you have good reason not to.

An alternative to inspections and walk-throughs is code reading.

In code reading, you read source code and look for errors and also comment on qualitative aspects of the code (design, style, readability, maintainability, and efficiency) - see p.494.

At NASA, code reading found 20-60% more errors over the life of the project than the various kinds of testing did.

The difference between code reading and inspections and walk-throughs is that code reading focuses more on individual review of the code than on the meeting.

A dog-and-pony show is a review in which a software product is demonstrated to a customer.

It is a management review rather than a technical review.

Comparison of Collaborative Construction Techniques - Table 21-1.

### Key Points


## Chapter 22 - Developer Testing

Unit testing is the execution of a complete class, routine, or small program in isolation from the more complete system (single or team of programmers).

Component testing is the execution of a class, package, small program, or other program element in isolation from the more complete system (multple or team of programmers).

Integration testing is the combined execution of two or more classes, packages, components, or subsystems (multiple or team of prgrammers).

Regression testing is the repetition of previously executed test cases for the purpose of finding defects in software that previously passed the same set of tests.

System testing is the execution of software in it final configuration, including integration with other software and hardware systems.

It tests for security, performance, resource loss, timing problems, and other issues that cannot be tested at lower levels of integration.

Testing is usually broken into two broad categories: black-box and white-box testing. Black-box testing refers to tests in which the tester cannot see the inner workings of the item being tested. White-box testing refers to tests in which the tester is aware of the inner workings of the item being tested (by a developer).

Testing is a means of detecting errors. Debugging is a means of diagnosing and correcting the root cause of errors that have already been detecteed.

### 22.1 Role of the Developer Testing in Software Quality

Collaborative development practices have been shown to find a higher percentage of errors than testing and they cost less.

The combination of testing steps often finds less than 60% of errors.

Depending on the size and complexity of the project, developer testing should probably take 8-25% of the total project time - Figure 22-1.

### 22.2 Recommended Approach to Developer Testing

A systematic approach to developer testing maximizes your ability to detect errors of all kinds with a minimum of effort - p. 503.

Watch for the following limitations with developer testing - p. 504.

Developers tend to test for whether the code works (clean tests) rather than test for all the ways the code breaks (dirty tests).

### 22.3 Bag of Testing Tricks

Since exhaustive testing is impossible, the art of testing is choosing the test cases most likely to find errors.

You need to concentrate on picking a few that tell you different things rather than a set that tells you the same thing.

When planning tests, eliminate those that do not tell you anything new.

The idea of structured basis testing is to test each stement in a program at least once.

If the statement is a logical statement - e.g. an if or a while - you need to vary the testing according to how complicated the expression inside the logical statement is to make sure that the statement is fully tested.

The easiest way to make sure that you have gotten all the bases covered is to calculate the number of paths through the program and then develop the minimum number of test cases that will exercise every path through the program.

The term code coverage is testing all the paths through a program. In contrast, structured basis testing covers all the paths with a minimal set of test cases.

How to compute the minimum number of cases needed for basis testing - p. 506.

#### Data-Flow Testing

Both control flow and data flow are equally important in testing.

Data flow testing is based on the idea that data usage is at least as error-prone as control flow.

Data can exist in one of three states - defined, used, or killed.

It is convenient to have terms that describe entering or exiting a routine - entered or exited.

The normal combination of data states is that a variable is defined, used one or more times, and perhaps killed.

View the following patterns suspiciously - see. p. 509-510.

Check for these anomalous sequences of data states before testing begins.

After you have checked for these sequences, the key to writing data-flow test cases is to exercise all possible defined-used paths - p. 510.

#### Other Testing Tricks

A good test case covers a large part of the possible input data.

If two test cases flush out exactly the same errors, you only need one of them - equivalence partitioning.

In addition to the formal test techniques, good programmers use a variety of less formal heuristic techniques to expose errors in their code.

The term error guessing means creating test cases based upon guesses about where the program might have errors.

One of the most fruitful areas for testing is boundary conditions (boundary analysis) or off-by-one errors.

There are three boundary cases: just less than max, max itself, and just greater than max.

Boundary analysis also applies to minimum and maximum allowable values.

A more subtle kind of boundary condition is compound boundaries - when the boundary involves a combination of variables.

Typical bad-data test cases: too little or no data, too much data, invalid data, wrong size of data, uninitialized data.

### 22.4 Typical Errors

You may think that defects are distributed evenly throughout your source code, but that is wrong.

Most errors tend to be concentrated in a few highly defective routines.

80% of errors are found in 20% of a project's classes or routines (80/20 rule).

The corollary is that 20% of a project's routines contribute 80% of the cost of development.

50% of errors are found in 5% of a project's classes (50/5 rule).

Regardless of the exact proportion of the cost contributed by highly defective routines, highly defective routines are extremely expensive.

This means that you can cut close to 80% of the cost by avoiding troublesome routines.

36% of all construction errors were clerical mistakes.

Three of the most expensive software errors of all time - involved the change of a single character in a previously correct program.

Take the time you need to understand the design thoroughly.

About 85% of errors can be fixed in less than a few hours.

If the data that classifies errors is inconclusive, so is the data that attributes errors to the various development activities.

One certainty is that construction always results in a significant number of errors.

On small projects, construction defects make up the bulk of all errors.

Construction defects account for at least 35% of all defects regardless of project size.

Construction errors, although cheaper to fix than requirements and design errors, are still expensive - Figure 22-2.

The number of errors you should expect to find varies according to the quality of the development process you use - p. 521.
A common experience is to spend several hours trying to find an error only to find that the error is in the test data!

Test cases tend to be created on the fly - especially when a developer writes the test cases. You can do several things to reduce the number of errors in your test cases: check your work, plan test cases as you develop the software, keep your test cases, and plug unit tests into a test framework (JUnit).

### 22.5 Test-Support Tools

Scaffolding

A mock or stub object is a class that is dummied up so that it can be used by another class that is being tested.

A driver or test harness is a fake routine that calls the real routine being tested.

A dummy file which is a small version of the real thing that has the same types of components that a full-size has.

### Notes

Regression testing is a lot easier if you have automated tools to check the actual output against the expected output.

To check printed output is to redirect the output to a file and use a file-comparison tool (diff) to compare it to the expected output.

You can also write code to exercise selected pieces of a program systematically - see p. 525.

A coverage monitor is a tool that keeps track of the code that is exercised and the code that is not.

Testing done without measuring code coverage typically exercises only about 50-60% of the code.

Strong logging aids error diagnosis and supports effective service after the software has been released.

You can build your own data recorder by logging significant events to a file - record the system state prior to an error and details of the exact error conditions.

Logging can be compiled into the development version of the code and compiled out of the released version.

If you implement logging with self-pruning storage and thoughtful placement and content of error messages, you can include logging functions in release versions.

A symbolic debugger is a technological supplement to code walk-throughs and inspections.

Walking through code in a debugger is in many respects the same process as having other programmers step through your code in a review.

Another class of test-support tools are designed to perturb your system.

Test-support tools in this class have a variety of capabilities: memory filling, memory shaking, selective memory failing, and memory-access checking (bounds checking) - p. 527.

One powerful test tool is a database of errors that have been reported.

The only practical way to manage regression testing is to automate it.

In manually testing, only about half of all the tests are executed properly.

You need to measure the project so that you can tell for sure whether changes improve or degrade it - p. 529.

You might find it useful to keep track of your personal test records.

These records can include both a checklist of the errors you most commonly make as well as a record of the amount of time you spend writing code, testing code, and correcting errors.

### Checklist: Test Cases

### Key Points


## Chapter 23 - Debugging

### 23.1 Overview of Debugging Issues

There is a 20-to-1 difference in the time it takes experienced programmers to find the same set of defects found by inexperienced programmers.

Some programmers find more defects and make corrections more accurately. Recall the General Principle of Software Quality.

If you do not know exactly what your code is doing, you are programming by trial and error (hacking) so defects are guaranteed.

You do not need to learn how to fix defects, you need to learn how to avoid them in the first place.

An experienced programmer can make mistakes too.

If this is the case, the error in your program is a powerful opportunity for you to learn - p. 538.

Even if an error appears not to be your fault, assuming that it is your it is your fault improves your credibility.

It also helps avoid embarassment of having to recant publicly later when you find out that it was your defect after all.

#### The Devil's Guide to Debugging

Find the defect by guessing

Do not waste time trying to understand the problem

Fix the error with the most obvious fix

### 23.2 Finding a Defect

Debugging consists of finding the defect and fixing it.

Finding the defect and understanding it is usually 90% of the work.

#### The Scientific Method of Debugging - p. 541

An error that does not occur predictably is usually an initialization error, timing issue, or dangling-pointer.

Narrow the test case to the simplest one that still produces the error.

#### Tips for Finding Defects - p. 544

Sometimes trying cases that are similar to the error-producing case but not exactly the same can be helpful - triangulating the defect.

Use a binary search technique to divide and conquer.

Be suspicious of classes and routines that have had defects before - take at second look at the error-prone classes and routines.

Check code that has changed recently

Check for common defects

Use code-quality checklists to stimulate your thinking about possible defects - also review the checklists in the book.

Talk to someone else about the problem - confessional debugging

If you are debugging and making no progress, take a break.

The onset of anxiety is a clear sign that it is time to take a break.

Brute-Force Debugging - p. 548

Syntax errors - p. 549

Do not trust line numbers in compiler messages.

When a compiler reports a mysterious syntax error, look immediately before and after the error.

Find misplaced comments and quotation marks (syntax errors).

To find an extra comment or quotation mark insert the following into your code in C, C++, and Java `/*"/**/`.

### 23.3 Fixing a Defect

Understand the problem before you fix it!

If you understand the context in which a problem occurs, you are more likely to solve the problem completely rather than only one aspect of it.

Before you rush to fix a defect, make sure you have diagnosed the problem correctly.

Rushing to solve a problem is one of the most time-ineffective things you can do. Relax long enough to make sure your solution is correct. Avoid taking shortucts.

Fix the problem, not the symptom

Make one change at a time

Check your fix

Check the program yourself, have someone else check it for you, or walk through it with someone else. Run the same triangulation test cases you use to diagnose the problem that all aspects of the problem have been resolved.

Add a unit test that exposes the defect

Look for similar defects

When you find one defect, look for others that are similar. Defects tend to occur in groups. If you cannot figure out how to look for similar defects, you do not completely understand the problem.

### 23.4 Psychological Considerations in Debugging

Seeing what you expect to see is called psychological set.

The ease with which two items can be differentiated is called psychological distance.

As you debug, be ready for the problems caused by insufficient psychological distance between similar variable names and similar routine names.

As you code, choose names with large differences so that you avoid the problem.

### 23.5 Debugging Tools

A source-code comparator such as diff is useful when you;

Make several changes and need to remove so that you cannot quite remember or

Discover a defect in a new version that you do not remember in an older version.

### Checklist: Debugging Reminders

### Key Points


## Chapter 24 - Refactoring

Coding, debugging, and unit testing consume 30-65% of the effort on a typical project.

If coding and unit testing were straight-forwared processes, they would consume no more than 20-30% of the total effort on a project.

Modern development practices increase the potential for code changes during construction.

### 24.1 Kinds of Software Evolution

The key distinction between kinds of software evolutions is whether the program's quality improves or degrades under modification.

If you treat modifications as opportunities to tighten up the original \ design of the program, quality improves.

A second distinction in the kinds of software evolution is the one between changes made during construction and those made during maintenance.

When you have to make a change, strive to improve the code so that future changes are easier.

Cardinal Rule of Software Evolution: Evolution should improve the internal quality of a program.

The key strategy in achieving it is refactoring.

### 24.2 Introduction to Refactoring

Here are some warning signs (smells) that indicate where refactoring are needed - p. 565.

Code is duplicated

DRY (Don't Repeat Yourself) principle, and Copy and paste is a design error.

A routine is too long

Routines longer than a screen are rarely needed. One way to improve a system is to increase its modularity (break it up into more routines that do one thing and do it well).

A loop is too long or too deeply nested

A class has poor cohesion

A parameter list has too many parameters

A long parameter list is a warning that the abstraction of the routine interface has not been well thought out.

Changes require parallel modifications to multiple classes

case statements have to be modified in parallel

A class does not do very much

A middleman object is not doing anything

A subclass uses only a small percentage of its parents' routines.

Comments are used to explain difficult code

A routine uses setup code before a routine call or takedown code after a routine call.

Checklist: Reasons to Refactor

### 24.3 Specific Refactorings

Data-level, statement-level, routine-level, class implementation, class interface, and system-level refactorings.

Checklist: Summary of Refactorings

### 24.4 Refactoring Safely

Refactoring is a powerful technique for improving code quality. Like all powerful tools, refactoring can do more harm than good if misused.

A few simple guidelines can prevent refactoring missteps:

Do not use refactoring as a cover for code and fix

Refactoring refers to changes in working code that do not affect the program's behavior. Proframmers who are tweaking broken code are not refactoring, thy are hacking.

Avoid refactoring instead of rewriting

Sometimes code does not need small changes (it needs to be tossed out so that you can start over). If you find yourself in a major refactoring session, ask yourself whether it should be redesigned and reimplented.

### 24.5 Refactoring Strategies

Refactoring is subject to the same law of diminishing returns as other programming activities and the 80/20 rule. Spend your time on the 20% of the refactorings that provide 80% of the benefit.

Consider the following guidelines when deciding which refactorings are most important:

Refactor when you add a routine

Refactor when you add a class

Refactor when you fix a defect

Target error-prone modules

Target modules with the highest complexity

In a maintenance environment improve the parts you touch

Define an interface between clean and ugly code, and move code across the interface.

### Checklist: Refactoring Safely

### Key Points


## Chapter 25 - Code-Tuning Strategies

### 25.1 Performance Overview

You can often find other ways to improve performance than by code tuning (in less time and with less harm to the code).

Users are more interested in tangible program characteristics than they are in code quality.

Users tend to be more interested in program throughput than raw performance.

Delivering software on time, providing a clean user interface, and avoiding downtime are often more significant.

Performance is only loosely related to code speed.

Be wary of sacrificing other quality characteristics to make your code faster. Your work on speed might hurt overall performance rather than help it.

If you have chosen efficiency (speed or size) as a priority, think about efficiency from each of these viewpoints: program requirements, program design, class and routine design, operating-system interactions, code compilation, hardware, and code tuning.

Performance is stated as a requirement far more often than it actually is a requirement.

If you know that a program's size and speed are important, design the program's architecture so that you can reasonably meet your size and speed goals. Then set resource goals for individual subsystems, features, and classes.

### 25.2 Introduction to Code Tuning

The term code tuning is the practice of modifying correct code in ways that make it run more efficiently.

It refers to small-scale changes that affect a single class, routine, or a few lines of code.
Notes

Code tuning is not the most effective or easiest way to improve performance.

The Pareto Principal (80/20 rule) states that you can get 80% of the result with 20% of the effort.

20% of program's routines consume 80% of its execution time.

Less than 4% of a program usually accounts for more than 50% of its run time.

You should measure the code to find the hotspots and then put your resources into optimizing the few percent that are used the most.

Knuth profiled his line-count program and found that it was spending half its execution time in two loops.

The best is the enemy of the good

Working toward perfection might prevent completion. Complete it first, then perfect it!

Common Misconceptions on Code Tuning (Old Wives' Tales) - p. 593

Do not optimize until you know you need to.

Optimizing compilers are better at optimizing straightforward code than they are at optimizing tricky (clever) code.

### 25.3 Kinds of Fat and Molasses

Input/Output operations

Use an in-memory data structure unless space is critical.

Paging

An operation that causes the operating system to swap pages of memory is much slower than an operations that works only on one page of memory.

System calls

Calls to system routines are often expensive and invlove a context switch - saving the program's state, recovering the kernal's state, and the reverse.

Errors

A final source of performance problems is errors in the code.

### 25.4 Measurement

Measure your code to find the hotspots.

Once you have found the hotspots and optimized them, measure your code again to asses how much you have improved it.

Many aspects of performance are counterintuitive.

Experience does not help much with optimization.

You can never be sure about the effect of an optimization until you measure the effect.

Measurements Need to Be Precise

### 25.5 Iteration

You will rarely get a 10-fold improvement from one technique, but you can effectively combine techniques. Keep trying even after you find one that works.

### 25.6 Summary of the Approach to Code Tuning

### Checklist: Code-Tuning Strategies

### Key Points



## Chapter 26 - Code-Tuning Techniques

This chapter focuses on improving speed and includes a few tips for making code smaller. Performance usually refers to both speed and size, but size reductions tend to come from redesinging classes and data.

The term _code tuning_ refers to small-scall changes.

_Refactoring_ is code changes that improve a program's internal structure. These changes degrade the internal structure in exchange for gains in performance.

The only reliable rule of thumb is to measure the effect of each tuning in your environment.

This chapter presents a catalog of _Things to Try_.

### 26.1 Logic

Arrange tests so that the one that's fastest and most likely to be true is performed first.

In some circumstances, a lookup table may be quicker than traversing a complicated change of logic.

A program can use _lazy evaluation_ to avoid doing any work until the work is needed.

### 26.2 Loops

The hotspots in a program are often inside loops.

_Switching_ refers to making a decision inside a loop every time it is executed.

You can unswitch the loop by making the decision outside the loop.

_Jamming_ or fusion is the result of combining two loops that operate on the same set of elements.

The goal of loop unrolling is to reduce the amount of loop housekeeping.

The main hazard of loop unrolling is an off-by-one error in the code.

Try to minimize the work done inside a loop.

Try to evaluate a statement or part of a statment outside of a loop so that only the result is used inside the loop.

When you have a loop with a compound test, you can often save time by using a sentinel value.

This technique can be applied to virtually any situation where you use a linear search.

When you have nested loops, think about which loop to place on the outside and inside.

Replacing an expensive operation such as multiplication with a cheaper operation such as addition is called _strength reduction_.

### 26.3 Data Transformations

Changes in data types can be a powerful aid in reducing program size and improving execution speed.

Try to minimize array references.

Using a _supplementary index_ means adding related data that makes accessing a data type more efficient.

You can add the related data to the main data type or store it in a parallel structure.

A _string-length index_ can be used to keep track of the length of the structure.

This is often more efficient than computing the length each time you need it.

An independent, _parallel index structure_ can be used to manipulate an index to a data type rather than the data type itself.
This is sometimes more efficient.

If you know that the same values tend to be requested repeatedly, you can cache the values.

### 26.4 Expressions

Complicated expressions (mathematical or logical) tend to be expensive so this section looks at ways to make them cheaper.

#### Notes on Expressions

- You can use algebraic identities to replace costly operations with cheaper ones.

- You can use strength reduction to replace expensive operations.

- If you are using a named constant or a magic number in a routine call and that is the only argument, you can precompute the number, put it into a constant, and avoid the routine call.

- System routines are expensive and provide accuracy that is often wasted

  Math routines using floating-point values, a right-shift operations is the same as dividing by two.

- If computed results are used many times, it is often cheaper to compute them once and save them rather than computing the results on the fly.

- Optimizing a program by precomputation can take several forms - p. 638.

- If you find an expression that is repeated several times, assign it to a variable and refer to the variable rather than recomputing the expression in several places.

### 26.5 Routines

One of the most powerful tools in code tuning is a good routine decomposition. Small, well-defined routines save space and make a program easy to optimize and they are relatively easy to rewrite in a low-level langauage.

### Checklist: Code-Tuning Techniques

### Key Points


## Chapter 27 - How Program Size Affects Construction

### 27.1 Communication and Size

As the number of people on a project increases, the number of communication paths increases.

It increases proportionally to the square of the number of people - Figure 27.1.

The more communication paths you have, the more time you spend communicating and the more opportunities are created for mistakes in communication.

### 27.3 Effect of Project Size on Errors

Both quantity and type of errors are affected by project size.

As project size increases, a larger percentage of errors can usually be attributed to mistakes in requirements and design - Figure 27.2.
On small projects, construction errors make up about 75% of all the errors found.

Methodology has less influence on code quality, and the biggest influence on program quality is often the skill of the individual writing the program.

As the kinds of defects change with size, so do the number of defects.

The _density of defects_ increases - the number of defects per 1000 lines of code - Table 27-1.

A large project will need to work harder than a small project to achieve the same error rate.

### 27.4 Effect of Size on Productivity

Productivity has a lot in common with software quality when it comes to project size.

At small sizes (2K LOC or less), the single biggest influence on productivity is the skill of the individual programmer. As project size increases, team size and organization become greater influences on productivity.

How big does a project need to be before team size begins to affect productivity? Table 27-2

Productivity on small projects can be 2-3 times as high as on large projects, and productivity can vary by a factor of 5-10 from the smallest projects to the largest.

27.5 Effect of Project Size on Development Activities

As project size increases, construction becomes a smaller part of the total effort - Figure 27-3 and 27-4.

List of activities that grow at a more than linear rate - p. 655.

Regardless of the size of a project, a few techniques are always valuable: disciplined coding practices, design and code inspections by other developers, good tool support, and use of high-level languages.

A more subtle influence on project size is the quality and the complexity of the final software - p. 656.

Kinds of software: program, product, system, and system product.

Failure to appreciate the differences in polish and complexity among different types of software is a common cause of estimation errors.

### Key Points


## Chapter 28 - Managing Construction

### 28.1 Encouraging Good Coding

Assign two people to every part of the project

Review every line of code

Typically involves the programmer and at least two reviewers.

Require code sign-offs

Route good code examples for review

Emphasize that code listings are public assets

Reward good code

The reward should be something the programmer wants and the should be exceptionally good or you lose credibility.

One easy standard

The manager should be able to read an understand any code written for the project.

28.2 Configuration Management

You have to plan software configuration management (SCM) carefully so that it is an asset rather than an albatross around your neck.

Requirements and Design Changes - p. 666.

A disk image of a standard developer workstation should be created and used.

Backup Plan

Try doing a restore at some point to make sure that the backup contains everything you need and that the recovery works.

### Checklist: Configuration Management

### 28.3 Estimating a Construction Schedule

The average large software project is one year late and 100% over budget. Developer estimates tend to be 20-30% too low.

#### Estimation Approaches

Establish objectives

Allow time for the estimate and plan it

Rushed estimates are inaccurate estimates.

Spell out software requirements

It is unreasonable to expect you to be able to estimate the amount of work required to build something that has not yet been defined.

Estimate at a low level of detail

Use several different estimation techniques and compare the results

Reestimate periodically - Figure 28-2

#### Influences on Schedule

The largest influence on a software project's schedule is the size of the program to be produced - Table 28-1.

See list of less easily qualified factors that can influence the schedule - bottom of p. 674.

#### What to Do If You are Behind

Hope they you will catch up

Delays and overruns generally increase toward the end of a project. Projects do not make up lost time later, they fall further behind.

Expand the time

Adding people to a late software project makes it later.

Reduce the scope of the project

### 28.4 Measurement

For any project attribute, it is possible to measure that attribute in a way that is superior to not measuring it at all.

Be aware of measurement side effects

People tend to focus on work that is measured and ignore work that is not.

To argue against measurement is to argue that it is better not to know what is really happening on your project - Table 28-2.

### 28.5 Treating Programmers as People

Highly technical companies offer parklike corporate campuses, organic organizational structures, confortable offices, and other _high-touch_ environmental features.

How Do Programmers Spend Their Time? Table 28-3

Variation in performance and quality

Talent and effort among individual programmers vary tremendously. Many studies have shown order-of-magnitude differences in the quality of the programs written, the sizes of the programs written, and the productivity of programmers.

Individual Variation

The ratio of initial coding time between the best and worst programmers was 20:1, debugging 25:1, program size 5:1, and program execution speed 10:1. They found no relationship between a programmer's amount of experience and code quality or productivity.

Team variation

Good programmers tend to cluster. Also, 80% of the contribution comes from 20% of the contributors.

Religious issues - p. 683

Physical environment makes a big difference in productivity

### 28.6 Managing Your Manager

The trick is to do it in a way that allows your manager to continue believing that you are the one being managed. Educate your manager about the right way to do things.

### Key Points


## Chapter 29 - Integration

The term _integration_ refers to the software development activity in which you combine separate software components into a single system. If you construct and integrate software in the wrong order, it is harder to code, harder to test, and harder to debug.

The benefits of careful integration - p. 690.

### 29.2 Integration Frequency - Phased or Incremental?

#### Phased Integration

Until a few years ago, phased integration was the norm - also called big-bang integration.

1. Unit development - Design, code, test, and debug each class
2. System integration - Combine the classes into one large system
3. System disintegration - Test and debug the whole system

#### Incremental Integration

In _incremental integration_ you write and test a program in small pieces and then combine the pieces one at a time.

1. Develop a small, functional part of the system

   Thoroughly test and debug it. This will serve as a skelton on which to build the remaining parts of the system.

2. Design, code, test, and debug a class.

3. Integrate the new class with the skeleton.

### 29.3 Incremental Integration Strategies

With incremental integration, you have to plan more carefully.

The order in which components are constructed has to support the order in which they will be integrated.

In _top-down integration_, the class at the top of the hierarchy is written and integrated first.

The interfaces between classes must be carefully specified.

A good alternative to pure top-down integration is the vertical-slice approach - Figure 29-6.

In _bottom-up integration_, you write and integrate the classes at the bottom of the hierarchy first.

You write test drivers to exercise the low-level classes initially and then add higher-level classes to replace driver classes as they are developed. The disadvantages are that it leaves integration of the major, high-level system interfaces until last and requires you to complete the design of the whole system before you start integration.

A good alternative to pure bottom-up integration is to use a hybrid approach - Figure 29-8.

Other approaches are feature-oriented integration and T-shaped itegration.

None of these approaches are robust procedures that you should follow methodically.

Like software design approaches, they are heuristics rather than algorithms. Instead, develop your own unique strategy tailored to your specific project.

### 29.4 Daily Build and Smoke Test

A good approach to integrating the software is the _daily build and smoke test_. Every file is compiled, linked, and combined into an executable program every day, and the program is then put through a smoke test - a simple check to see if the product smokes when it is run.

Here are some of the ins and outs of using daily builds:

Build daily

Check for broken builds

Each project sets its own standard for what constitutes breaking the build. At a minimum a good build should:

  1. Compile all files, libraries, and other components successfully.
  2. Link all files, libraries, and other components successfully.
  3. Pass the smoke test.

Smoke test daily

The smoke test should exercise the entire system from end to end.

Keep the smoke test current

Automate the daily build and smoke test

More topics - p. 704-706

### Checklist: Integration

### Key Points


## Chapter 30 - Programming Tools

### 30.2 Source-Code Tools

List of features that a good IDE supports - p. 710.

Multiple-File String Search and Replace

Source-Code Beautifiers

Templates are an easy way to encourage consistent coding and documentation styles.

Picky Syntax and Semantics Checkers

Metrics Reporters

Refactorers

A _data dictionary_ is a database that describes all the significant data in a project.

### 30.3 Executable-Code Tools

Code Creation - make, code libraries, debugging, and testing

Code Tuning - execution profiler

The Art of Unix Programming

The Pragmatic Programmer

### Checklist: Programming Tools

### Key Points



## References

[1]: [Code Complete](http://www.cc2e.com)