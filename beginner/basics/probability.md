# Probability Review

## Probability Spaces

- A _sample space_ is a collection of basic outcomes or sample points.

- An _event_ is a subset of a sample space which is a basic outcome of a sample space.

- Ω or S represents the space of all possible experimental outcomes.

- ∅ represents an impossible event


## Random variables

- A _random variable (rv)_ is a function from sample points to some range.

- A random variable can be discrete or continuous.

- A discrete random variable is a function X: Ω → S where S is a countable subset of ℝ

- If X : Ω → {0, 1} then X is called an _indicator random variable_ or a _Bernoulli trial_

- A _probability mass function (pmf)_ defines the probabilities for a discrete random variable.

- A _probability density function (pdf)_ specifies the probability for a continuous random variable — the likelihood the value falls into a range of values.

Since a random variable has a numeric range, we can often do mathematics more easily by working with the values of a random variable


Introduction to Mathematical Statistics Section 2.1. 


## Axioms of Probability Theory

Probability Theory provides us with the formal mechanisms and rules for manipulating propositions represented by probabilistically.

The following are the three axioms of probability theory:

  1. 0 ≤ P(A) ≤ 1 for all A ∈ S (for all A in the sample space of S)

  2. P(S) = 1

  3. P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

If A and B are disjoint (A ∩ B = ∅) then P(A ∪ B) = P(A) + P(B)

Two events A and B are _independent_ iff P(A ∩ B) = P(A) P(B)


## Joint probability

- A _joint probability_ is the probability of two events happening together.

The two events are usually designated A and B.

It can be written as:  p(A and B) or p(A ∪ B) or p(A, B)

- Joint probability is also called the _intersection_ of two (or more) events which can be represented by a _Venn diagram_.


## Conditional Probability

- A _conditional probability_ is a measure of the probability of an event given that another event has occurred.

P(A|B) = P(B|A) * P(A) / P(B)

We can simplify this calculation by removing the normalizing value of P(B). 

P(A|B) = P(B|A) * P(A)

posterior = likelihood * prior

The probability of an event before we consider this additional knowledge P(A) is called the _prior_ probability of the event.

The probability that results from using our additional knowledge P(A|B) is the _posterior_ probability.

The reverse conditional probability P(B|A) is called the _likelihood_. 

- The conditional probability of an event A given that an event B has occurred is the _chain rule_ or _multiplication rule_:

P(B|A) = P(A ∩ B) / P(A)  or  
P(A|B) = P(A,B) / P(B)  if P(B) ≠ 0

P(A,B) = P(A) P(B|A)

- The generalization of the chain rule to multiple events gives us:

P(A ∩ B ∩ C) = P(A|B ∩ C) P(B|C) P(C)


## Probability Rules

Sum Rule:  P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

Conditional Probability:  P(A|B) = P(A,B) / P(B) if P(B) ≠ 0

Product Rule:  P(A,B) = P(A|B) P(B) = P(B|A) P(A)

Chain rule (multiple application of product rule)

  P(A,B) = P(A|B) P(B) = P(B|A) P(A)

  P(A|B) = P(A,B) / P(B)

We can generalize the product rule for a joint probability of an arbitrary number of variables.

  P(A,B,C) = P(A|B,C) P(B|C) P(C)

The chain rule is useful in the study of Bayesian networks which describe a probability distribution in terms of conditional probabilities.


Conditionalized version of the Chain Rule:

  P(A, B|C) = P(A|B, C) P(B|C)  iff  P(A|B,C) = P(A, B|C) / P(B|C)


## Bayes’ Rule (Theorem)

P(A|B) = P(A,B) / P(B) = P(B|A) P(A) / P(B)

P(B|A) = P(A,B) / P(A) = P(A|B) P(B) / P(A)

We can also think of this as:

  P(hypothesis|evidence) = P(evidence|hypothesis) P(hypothesis) / P(evidence)

If we are simply interested in which event out of some set is the most likely given A , we can ignore it so that P(B|A) = P(A|B) P(B)


## General form of Bayes' Rule

The general form of Bayes’ rule with normalization is

  P(A|B) = P(B|A) P(A) / P(B) = α P(B|A) P(A)

where α is the normalization constant needed to make the entries in P(Y|X) sum to 1.


Conditionalized version of Bayes's Rule:

  P(A|B, C) = P(B|A, C) P(A|C) / P(B|C)


## Independence

A and B are _independent_ iff

  P(A|B) = P(A)  or  P(B|A) = P(B)  or  P(A,B) = P(A) P(B)

A and B are _conditionally independent_ given C iff

  P(A|B, C) = P(A|C)
  P(B|A, C) = P(B|C)
  P(A, B|C) = P(A|C) P(B|C)



----------


# Reasoning under Uncertainty

## Why Reason Probabilistically?

- In many problem domains it is not possible to create complete, consistent models of the world. Therefore agents (and people) must act in uncertain worlds.

- We want an agent to make rational decisions even when there is not enough information to prove that an action will work.

- Some of the reasons for reasoning under uncertainty:

  - True uncertainty. Example: flipping a coin.

  - Theoretical ignorance.

    There is no complete theory which is known about the problem domain. Example: medical diagnosis.

  - Laziness.

    The space of relevant factors is very large, and would require too much work to list the complete set of antecedents and consequents.
    Furthermore, it would be too hard to use the enormous rules that resulted.

  - Practical ignorance.

    Uncertain about a particular individual in the domain because all of the information necessary for that individual has not been collected.

  - Probability theory will serve as the formal language for representing and reasoning with uncertain knowledge.


## Axioms of Probability Theory

Probability Theory provides us with the formal mechanisms and rules for manipulating propositions represented probabilistically.

The following are the three axioms of probability theory:

1. 0 ≤ P(A=a) ≤ 1 for all a ∈ A (for all a in the sample space of A)

2. P(True)=1, P(False)=0

3. P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

From these axioms we can show the following properties also hold:

  P(~A) = 1 - P(A)

  P(A) = P(A ∩ B) + P(A ∩ ~B)

  Sum{P(A=a)} = 1 where the sum is over all possible values a ∈ A


## Joint Probability Distribution

Given an application domain in which we have determined a sufficient set of random variables to encode all of the relevant information about that domain, we can completely specify all of the possible probabilistic information by constructing the full joint probability distribution, P(V1=v1, V2=v2, ..., Vn=vn) which assigns probabilities to all possible combinations of values to all random variables.

### Example

Consider a domain described by three Boolean random variables, Bird, Flier, and Young.

We can enumerate a table showing all possible interpretations and associated probabilities:

```
  Bird    Flier	Young	Probability
  T	    T	    T	    0.0
  T	    T	    F	    0.2
  T	    F	    T	    0.04
  T	    F	    F	    0.01
  F	    T	    T	    0.01
  F	    T	    F	    0.01
  F	    F	    T	    0.23
  F	    F	    F	    0.5
                          ------
                          1.0
```

Notice that there are 8 rows in the above table representing the fact that there are 2^3 ways to assign values to the three Boolean variables.

In general, for n Boolean variables the table will be of size 2^n. And if n variables each had k possible values, then the table would be size k^n.

Also notice that the sum of the probabilities in the right column must equal 1 since we know that the set of all possible values for each variable are known.

This means that for n Boolean random variables, the table has 2^n - 1 values that must be determined to completely fill in the table.

**If all of the probabilities are known for a full joint probability distribution table, we can compute any probabilistic statement about the domain.**

Using the table above, we can compute

  P(Bird=T) = P(B) = 0.0 + 0.2 + 0.04 + 0.01 = 0.25

  P(Bird=T, Flier=F) = P(B,~F) = P(B,~F,Y) + P(B,~F,~Y) = 0.04 + 0.01 = 0.05


## Conditional Probabilities

Conditional probabilities are key for reasoning because they formalize the process of accumulating evidence and updating probabilities based on new evidence.

For example, if we know there is a 4% chance of a person having a cavity, we can represent this as the _prior_ (unconditional) probability P(Cavity)=0.04. Say that person now has a symptom of a toothache, we would like to know what is the _posterior_  probability of a Cavity given this new evidence, i.e. compute P(Cavity | Toothache).

If P(A|B) = 1, this is equivalent to the sentence in Propositional Logic B => A. Similarly, if P(A|B) = 0.9, then this is like saying B => A with 90% certainty. In other words, we have made implication fuzzy because it is not absolutely certain.

Given several measurements and other "evidence", E1, ..., Ek, we will formulate queries as P(Q | E1, E2, ..., Ek) meaning "what is the degree of belief that Q is true given that we know E1, ..., Ek and nothing else."

Conditional probability is defined as:

  P(A|B) = P(A ^ B) / P(B) = P(A,B) / P(B)

One way of looking at this definition is as a normalized (using P(B)) joint probability (P(A,B)).


## Example: Computing Conditional Probability from the Joint Probability Distribution

Say we want to compute `P(~Bird | Flier)` and we know the full joint probability distribution function given above. We can do this as follows:

```
   P(~B|F) = P(~B,F) / P(F)
     = (P(~B,F,Y) + P(~B,F,~Y)) / P(F)
     = (.01 + .01) / P(F)
```

Next, we could either compute the marginal probability P(F) from the full joint probability distribution, or we could do it by using a process called _normalization_ which first requires computing:

```
   P(B|F) = P(B,F) / P(F)
  	  = (P(B,F,Y) + P(B,F,~Y)) / P(F)
  	  = (0.0 + 0.2) / P(F)
```

Now we also know that P(~B|F) + P(B|F) = 1, so substituting from above and solving for P(F) we get P(F) = 0.22. Hence, P(~B|F) = 0.02 / 0.22 = 0.091.

While this is an effective procedure for computing conditional probabilities, it is intractable in general because it means that we must compute and store the full joint probability distribution table, which is exponential in size.


## Some important rules related to conditional probability

Rewriting the definition of conditional probability, we get the Product Rule:

  P(A,B) = P(A|B) P(B) = P(B|A) P(A)

  P(A|B) = P(A,B) / P(B)

Chain Rule:

  P(A,B, C) = P(A|B, C) P(B|C) P(C)

which generalizes the product rule for a joint probability of an arbitrary number of variables. Note that ordering the variables results in a different expression but all have the same resulting value.

Conditionalized version of the Chain Rule:

  P(A,B|C) = P(A|B,C) P(B|C)  <=> P(A|B,C) = P(A,B|C) / P(B|C)

Bayes's Rule:

  P(A|B) = P(B|A) P(A) / P(B) = α P(B|A) P(A)

which can be written to more clearly emphasize the _updating_ aspect of the rule:

  P(A|B) = P(A) * [P(B|A) / P(B)]

Note: The terms P(A) and P(B) are called the _prior_ (or marginal) probabilities. The term P(A|B) is called the _posterior_ probability because it is derived from or depends on the value of B.

Conditionalized version of Bayes's Rule:

  P(A|B, C) = P(B|A, C) P(A|C) / P(B|C)

Conditioning (Addition) Rule:

  P(A) = Sum{P(A|B=b) P(B=b)}

where the sum is over all possible values b in the sample space of B.

```
  P(~B|A) = 1 - P(B|A)
```

## General form of Bayes' Rule

The general form of Bayes’ rule with normalization is

  P(Y|X) = α P(X|Y) P(Y)

where α is the normalization constant needed to make the entries in P(Y|X) sum to 1.


----------


# Conditional Independence in General

Conditional independence of two random variables A and B given C holds just in case

(1)  p(A, B|C) = p(A|C) p(B|C)                                 (13.13)

Equivalently, we could have said that A and B are conditionally independent given C just in case B does not tell us anything about A if we already know C:

(2)  p(A|C) = p(A|B, C)

It is easy to see that (1) and (2) are equivalent by observing that in general

(3)  p(A, B|C) = p(A,B,C) / p(C)
              = p(A|B,C) p(B,C) / p(C)
              = p(A|B,C) p(B|C).

In general,

  P(A|B) = P(B|A) P(A) / P(B) = α P(B|A) P(A)

where α is the normalization constant needed to make the entries in P(A|B) sum to 1.

## Conditional independence in Bayesian Networks

It is crucial to keep in mind that the discussion of conditional independence in Bayesian Networks is always about nodes/variables that are _necessarily_ independent, given the structure of the underlying DAG.

One might also call this "topological independence", since one only takes into account properties of this DAG.  Necessary/topological independence holds regardless of probability assignments.

Even if two nodes are not necessarily/topologically (conditionally) independent, they might still turn out to be independent if one takes into account specific probability assignments.

### Example

Consider a simple DAG with vertices A and B and one directed edge A->B.

Then, by the definition of Bayesian Networks, A and B are not topologically independent.

However, they might still be independent, if we happen to assign probabilities in such a way that

  p(B|A) = p(B)

This is the case whenever we have a conditional probability table for p(B|A) of the following form:

  A   p(B)
  t   x
  f   x

for any real number x in [0;1].  It just means that A tells us nothingabout B, i.e., B happening is independent of A happening.

Textbooks typically deal with the first notion of topological independence, but it is not always clear what some exercises ask about.

Simply asking "are A and B independent" in the above example is ambiguous, since topologically they are not _necessarily_ independent, though for some probability tables they actually are.

R&N's first definition should probably go like this:

(4)  A node is necessarily conditionally independent of its non-descendants given its parents.


## The specific case of Figure 14.2

Abbreviate the nodes as B, E, A, J, and M.

The simplest illustration of necessary/topological independence involves B and E.

By (4), B is independent of E, since E does not descend from B, and unconditionally so, since B has no parents.

Thus, P(B|E) = P(B), P(E|B) = P(E), and P(B,E) = P(B) P(E).

This is a trivial result if we only consider the DAG over {B, E}, whose edge set is empty.

By the semantics of BNs, the overall joint distribution p(B, E) is the product of all conditional distributions, here p(B) p(E).

```
    p(B) = 0.001 and p(E) = 0.002

    B E   p(A) = p(a|b,e)
    T T   0.95
    T F   0.94
    F T   0.29
    F F   0.001
```

However, B and E are not independent given A.

To see this, consider the graph restricted to the nodes {B, E, A}.

Now the joint distribution is:

```
  p(B,E,A) = p(A|B,E) p(B) p(E)
           = p(B|A,E) p(A) p(E)

  P(A|B) = P(A,B) / P(B)              [Definition]
  P(A,B) = P(A|B) P(B) = P(B|A) P(A)  [Product rule]

  P(A,B,C) = P(A|B,C) P(B|C) P(C)     [Chain rule]

  p(B,E,A) = p(B|A,E) p(E|A) p(A)
           = p(B|A,E) p(E) p(A)
```

We can obtain `p(B,E|A)` as follows:

```
  P(A|B) = P(A,B) / P(B)  [Definition]

  p(B,E|A) = p(B,E,A) / p(A)
           = p(B,E,A) / sum_b,e p(B=b,E=e,A)
           = p(B,E,A) / [ p(B=0,E=0,A) + p(B=0,E=1,A) + p(B=1,E=0,A) + p(B=1,E=1,A) ]
```

Conditionalized version of the Chain Rule:

  P(A,B|C) = P(A|B,C) P(B|C)  iff  P(A|B,C) = P(A,B|C) / P(B|C)

Conditionalized version of Bayes's Rule:

  P(A|B,C) = P(B|A,C) P(A|C) / P(B|C)

Conditioning (Addition) Rule:

  P(A) = Sum{P(A|B=b) P(B=b)}

where the sum is over all possible values b in the sample space of B.

```
  P(B=b,E=e,A=a) = p(a|b,e) p(b) p(e)   P(B) = 0.001, P(E) = 0.002

  p(a|b,e) = [a = (b,e)]

  b e a  p(a|b,e)
  0 0 0  0.999 = 1 - 0.001
  0 0 1  0.001
  0 1 0  0.71  = 1 - 0.29
  0 1 1  0.29
  1 0 0  0.06  = 1 - 0.94
  1 0 1  0.94
  1 1 0  0.05  = 1 - 0.95
  1 1 1  0.95

  P(A|B) = P(A,B) / P(B)  [Definition]

  P(B|A) = P(A,B) / P(A) = P(A|B) P(B) / P(A)  [Bayes Rule]

  P(A|B,C) = P(B|A,C) P(A|C) / P(B|C)  [Bayes Rule]

  p(B,E|A) = p(B,E,A) / p(A) => p(B,E,A) = P(B,E|A) P(A)

  P(A,B|C) = P(A|B,C) P(B|C)  iff  P(A|B,C) = P(A,B|C) / P(B|C)  [Chain Rule]

  p(B,E,A) = p(A|B,E) p(B) p(E)

  p(b=0,e=0,a=0) = 0.999 * 0.999 * 0.998 = 0.996004998
  p(b=0,e=1,a=0) = 0.710 * 0.999 * 0.002 = 0.00141858
  p(b=1,e=0,a=0) = 0.060 * 0.001 * 0.998 = 0.00005988
  p(b=1,e=1,a=0) = 0.050 * 0.001 * 0.002 = 0.0000001

  0.996004998 + 0.00141858 + 0.00005988 + 0.0000001 = 0.997483558

  p(b=0,e=0,a=1) = 0.001 * 0.999 * 0.998 = 0.000997002
  p(b=0,e=1,a=1) = 0.290 * 0.999 * 0.002 = 0.00057942
  p(b=1,e=0,a=1) = 0.940 * 0.001 * 0.998 = 0.00093812
  p(b=1,e=1,a=1) = 0.950 * 0.001 * 0.002 = 0.0000019

  0.000997002 + 0.00057942 + 0.00093812 + 0.0000019 = 0.002516442

  p(B,E|A) = p(B,E,A) / p(A)
           = p(B,E,A) / sum_b,e p(B=b,E=e,A)

  p(b=0,e=0|a=0) = (0.999 * 0.999 * 0.998) / 0.997483558
                 = 0.9985177099

  p(b=0,e=0|a=1) = (0.001 * 0.999 * 0.998) / 0.002516442
                 = 0.396195104

  p(b=0,e=1|a=0) = (0.710 * 0.999 * 0.002) / 0.997483558
                 = 0.00142215878

  p(b=0,e=1|a=1) = (0.290 * 0.999 * 0.002) / 0.002516442
                 = 0.2302536677
```

This gives us the following table:

```
  B E A  p(B,E|A)
  0 0 0  0.998517709903
  0 0 1  0.39619510404      (trigger-happy alarm!)
  0 1 0  0.00142215878008
  0 1 1  0.230253667678
  1 0 0  6.00310646925e-05
  1 0 1  0.372796193991
  1 1 0  1.00252279046e-07
  1 1 1  0.000755034290478
```

Next we compute `p(B|A)` as

```
  p(B|A) = p(B,A) / p(A)
         = sum_e p(A|B,E=e) * p(B) * p(E=e) /
           sum_b,e p(A|B=b,E=e) * p(B=b) * p(E=e)

  B E A  p(B,A)
  0 0 0  0.998517709903
  0 1 0  0.00142215878008

  0.998517709903 + 0.00142215878008 = 0.9999398687

  0 0 1  0.39619510404      (uh-oh, trigger-happy alarm!)
  0 1 1  0.230253667678

  0.39619510404 + 0.230253667678 = 0.6264487717

  1 0 0  6.00310646925e-05
  1 1 0  1.00252279046e-07

  6.00310646925e-05 + 1.00252279046e-07 = 6.013131697e-05

  1 0 1  0.372796193991
  1 1 1  0.000755034290478

  0.372796193991 + 0.000755034290478 = 0.3735512283
```

which gives us this table:

```
  B A  p(B|A)
  0 0  0.999939868683
  0 1  0.626448771718
  1 0  6.01313169715e-05
  1 1  0.373551228282
```

Proceed analogously for `p(E|A)`:

```
  E A  p(E|A)
  0 0  0.998577740968
  0 1  0.768991298031
  1 0  0.00142225903236
  1 1  0.231008701969
```

Now it is easy to see that p(B,E|A) != p(B|A) * p(E|A).

For example:

```
  p(B=0,E=1|A=1) ~= 0.23
```

but

```
  p(B=0|A=1) ~= 0.626
  p(E=1|A=1) ~= 0.231
```

whose product is clearly not equal to 0.23.


## Example on p. 514

To illustrate this, we can calculate the probability that the alarm has sounded (but neither a burglary nor an earthquake has occurred) and both John and Mary call.

We multiply entries from the joint distribution (using single-letter names for the variables):

```
  P(j,m,a,¬b,¬e) = P(j|a) P(m|a) P(a|¬b∧¬e) P(¬b) P(¬e)
                 = 0.90 × 0.70 × 0.001 × 0.999 × 0.998
                 = 0.000628.
```

Section 13.3 explained that the full joint distribution can be used to answer any query about the domain.

If a Bayesian network is a representation of the joint distribution, it can also be used to answer any query, by summing all the relevant joint entries.

Section 14.4 explains how to do this but also describes methods that are much more efficient.



## Terminology

Some of the basic terminology and concepts of probability distribution.

**Random Experiment:** A random experiment is an experiment for which we cannot predict the output with certainty. For example, tossing a coin. We can have two possible outcomes, heads or tails.

**Sample Space:** It is the set of all possible outcomes of an experiment. It is usually represented by ‘S’.

For example, sample space for tossing a coin will be, S = { Heads, Tails }; for tossing two coins, S = {HT, TH, HH, TT}

**Event:** An event is the subset of sample space (S), and is usually useful in calculating probability.

For example, tossing two coins simultaneously and the outcomes which have at least one head, then the set of all such possibilities can be given as: E = { HT, TH }

**Random Variable:** It is a function that maps every outcome in the sample space to real number. Depending upon the values it take, a random variable can be classified into two types, i.e. discrete and continuous.

**Discrete Random Variable:** A discrete random variable can have only finite or countably infinite set of values. Example : For a coin toss either it’ll be heads or tails(binary), number of orders received in a shop(countably infinite).

These are described using probability mass function (PMF) and cumulative distribution function (CDF). PMF is the probability that a random variable X takes a specific value (eg. number of returns in an e-commerce site is 30, P(X = 30).

CDF is the probability that the random variable X will take a values less than or equal to 30, P(X≤ 30).

**Continuous Random Variable:** A continuous random variable can have infinite set of values. Example : Attrition percentage in a company.

These are described using probability density function (PDF) and cumulative distribution function (CDF). 

PDF is the probability that the continuous random variable will take in the neighborhood of x.

Figure: Probability Density Function

CDF is the probability that random variable will take the value less than or equal to value ‘z’.

Figure: Cumulative Distribution Function


### Binomial Distribution

A binomial distribution is a discrete probability distribution in which the random variable can have only two outcomes, pass or fail. 

The probability of pass is p and for fail is (1-p). 

Examples are customer churn, loan default ( default / no default), and coin toss (heads / tails).

### Code Samples

On any particular day in a bank about 20% of their loan repayment is defaulted. 

On one specific day, 25 customers are to repay their loan. 

We can calculate the following:

1. Probability that exactly 10 customers will default.

2. Probability that a maximum of 10 customers will default.

3. Probability that more than 10 customers will default.



## References

[Probability Distribution : A Brief Introduction](https://medium.com/@the.erised/probability-distribution-a-brief-introduction-55cc11148f35)

[Reasoning under Uncertainty (Chapters 13 and 14.1 - 14.4)](http://pages.cs.wisc.edu/~dyer/cs540/notes/uncertainty.html)
l
[Conditional Independence in General](http://www.cs.columbia.edu/~kathy/cs4701/documents/conditional-independence-bn.txt)


