# Facts and Fallacies of Software Engineering

Robert L. Glass, 1st ed., 2002. 



# Chapter 1 - About Management

## Fact 1

> The most important factor in software work is the quality of the programmers. 

## Fact 2

> The best programmers are up to 28 times better than the worst programmers. 

## Fact 3

> Brooke's Law: Adding people to a late project makes it later.

- Not all people contribute the same amount to a projet, so not all man-months are equal.

## Fact 4

> The working environment has a profound impact on productivity and product quality. 

## Fact 6

> Learning a new tool or technique initially lowers programmer productivity and product quality.

## Fact 7

> Software developers talk a lot about tools. They evaluate quite a few, buy a fair number, and use practically none.

shelfware = tools that were purchased, put on the shelf, and never used.

The problem is a culture that puts schedule performance (using impossible schedules)
above all else.

## Fact 8

> One cause of runaway projects is poor estimation.

The two causes of runaway projects are: poor estimation and unstable requirements.

The problem with _function points_:

1. Experts disagre on what should be counted and how counting should happen.

2. For some projects, FPs make sense, but for others they make no sense at all (where the number of inputs and outputs is far less significant than the complexity of the logic inside the program).

## Fact 9

> Estimation usually occurs at the wrong time (before the requirements are defined).

## Fact 10

> Software estimates are done by the wrong people (usually upper management).

## Fact 11

> Software estimates are rarely adjusted as the project proceeds.

## Fact 12

> Since sofware estimates are so faulty, there is little reason to be concerned when projects do not meet estimated targets. But everyone is concerned anyway.

Management _by schedule_ should be replaced with: by product, by issue, by risk, by business objectives, by quality.

_Extreme Programming_ suggests that after the customer or user choosers three of the four factors (cost, schedule, features, and quality) the software developers get to choose the fourth.

- In our culture today, people are trying so hard to achieve impossible schedules that they are willing to sacrifice completeness and quality in getting there.

- Projects where no estimates were prepared at all were best on productivity.

- There is a very strong correlation between level of productivity and feeling of control.

## Fact 14

> The answer to a feasibility study is almost always "yes".

## Fact 15

> Reuse-in-the-small (libraries of subroutines) began nearly 50 years ago and is a well-solved problem.

## Fact 16

> Reuse-in-the-large (components) remains a mostly unsolved problem, even though everyone agrees that it is important and desirable.

The key word in understanding this problem is _useful_.

## Fact 17

> Reuse-in-the-large works best in families of related systems, so it is domain-dependent.

## Fact 18

There are two _rules of three_ in reuse:

1. It is three times as difficult to build reusable components as single-use components.

2. A reusable component should be tried out in three different applications before it will be sufficiently general to accept in a reuse library.

## Fact 19

> Modification of reused code is particularly error-prone.

- The problem underlying the difficulties of modifying existing software: _comprehending the existing solution_.

- It is almost always a mistake to modify packaged, vendor-produced softare.

- This same problem has interesting ramifications for open-source software (where it is easy to access the code to modify it, but the wisdom of this is clearly questionable.

- It is necessary to accept the fact that software products are difficult to build and maintain.

- If a software system is to be modified at or above 20-25%, then it is cheaper and easier to start over and build a new project.

- Software work is the most complex that humanity has ever undertaken.

## Fact 20

> Design pattern reuse is one solution to the problems inherent in code reuse.

## Fact 21

> For every 25% increase in problem complexity, there is 100% increase in complexity of the software solution. 

Software solutions are complex.

## Fact 22

> 80% of software work is intellectual. A fair amount is creative. Little of it is clerical.



# Chapter 2 - Requirements

## Fact 23

> The other cause of runaway projects is unstable requirements (Fact 8).

## Fact 24

> Requirements errors are the most expensive to fix when found during production but the cheapest to fix early in development.

## Fact 25

> Missing requirements are the hardest requirements errors to correct.

The most persistent software errors (that escape testing) are errors of omitted logic (caused by missing requirements).

## Fact 26

> When moving from requirments to design, there is an explosion of _derived requirements_ (for a particular design solution) caused by the complexity of the solution process. The list of these design requirments is often 50 times longer than the list of original requirements.

This is because it is difficult to implement requirements traceability even though everyone agrees it is desirable to do so.

## Fact 27

> There is seldom one best design solution to a software problem.

## Fact 28

> Design is a complex, iterative process. The initial design solution will likely be wrong and certainly not optimal.

## Fact 29

> The transition from design to coding can be smooth as long as the person who did the design is the same person who does the coding.

The designer and coder need to agree on the _primitives_ -- the fundamental software units that are well known and easily coded.

## Fact 31

> Error removal is the most time-consuming phase of the life cycle.

Time spent on project: 20/20/20/40 - 20% for requirements, 20% for design, 20% for coding, and 40% for error reomval.

## Fact 35

> Test automation rarely is.

## Fact 37

> Rigorous inspections can remove up to 90% of errors from a software product before the first test case is run.

## Fact 38

> Rigorous inspections should not replace testing.

## Fact 41

> Maintenance typically consumes 40-80% of software costs. It is probably the most important life cycle phase of software.

Many companies do not collect enough data to know how much maintenance they are doing.

## Fact 42

> Enhancement is responsible for 60% of software maintenance and error correction for 17%.

> The 60/60 rule: 60% of software cost is spent on maintenance and 60% of that maintenance is enhancement.

## Fact 43

> Maintenance is a solution, not a problem.

## Fact 44

> The task of understanding the existing product  consumes about 30% of the total maintenance time and is the dominant maintenance activity. Therefore, it could be said that **maintenance is more difficult than development**.

The numbers for maintenance life cycle on bottom of p. 121.

## Fact 45

> Better software engineering leads to more maintenance, not less.



# Chapter 3 - Quality

> Management's job is to facilitate and enable technical people and then get out of the way.

## Fact 46

> Quality is a collection of attributes.

There are seven attributes: portability, reliability, efficiency, usability, testability, understandability, and modifiability.

What do those attributes mean? p. 129

## Fact 48

> There are errors that most programmers tend to make.

## Fact 49

> Errors tend to cluster.

- Some parts of a program are more complex than others.

- Some programmers tend to make more errors.

## Fact 50

> There is no single best approach to error removal.

## Fact 52

> Efficiency stems more from good design than from good coding.

## Fact 53

> High-order language (HOL) code (with appropriate compiler optimizations) can be about 90% as efficient as the comparable assembler code.

## Fact 54

> There are tradeoffs between size and time optimization.



# Chapter 4 - About Research

## Fact 55

Many software researchers advocate rather than investigate. 

As a result, (a) some advocated concepts are worth far less than their advocates believe, and (b) there is a shortage of evaluative research to help determine what the value of such concepts really is.



# Chapter 5 - About Management

## Fallacy 1

> You cannot manage what you cannot measure.

- GQM approach is to establish Goals to be satisfied by metrics, determine what questions should be asked to meet those goals, and only then collect the metrics needed to answer only those questions.

- Good knowledge worker managers tend to measure qualitatively not quantitatively.

## Fallacy 4

> Tools and techniques: one size fits all.

- If _shelfware_ is the current status of many software tools, then _dull thud_ is the status of most methodologies.

## Fallacy 6

> To estimate cost and schedule, first estimate lines of code.



# Chapter 6 - About the Life Cycle

## Fallacy 7

> Random test input is a good way to optimize testing.

Remember that: 1) there are common errors that most programmers tend to make and 2) errors tend to cluster.

## Fallacy 8

> Given enough eyeballs, all bugs are shallow.

There is a maximum number of useful inspection participants (somewhere between two to four) beyond which the success of inspection falls off rapidly.

## Fallacy 9

> The way to predict future maintenance costs and to make product replacement decisions is to look at past cost data.

There is a _bathtub_ shape to maintenance costs. 

There is a lot of maintenance when a product is first put into production. Then time passes, and we descend into stable, low-maintenance middle of the maintenace process. As enhancement interest drops off, the bugs are pretty well under control. Then, after the product has been out a while, it has been pushed to the ragged edge of its design envelope to the point where simple changes or fixes tend to get more expensive, and we reach the upward slope at the other end of the bathtub. 

Changes that used to be cheap are now getting expensive.

## Fallacy 10

> You teach people how to program by showing them how to write programs.

In learning a language, the first thing to do is to learn to _read_ it.



# References

[1] R. L. Glass, _Facts and Fallacies of Software Engineering_, 1st ed., Addison-Wesley Professional, ISBN: 0321117425, 2002. 

[2] R. L. Glass, _Software Creativity_, 1995. 

[3] G. Weinberg, _The Psychology of Computer Programming_, 1971

[4] A. Cockburn, _Agile Software Development_, 2002

