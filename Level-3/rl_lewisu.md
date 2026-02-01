## [Reinforcement Learning](https://medium.com/machine-learning-for-humans/reinforcement-learning-6eacf258b265)

In supervised learning, training data comes with an answer key from some godlike "supervisor" (gold standard)

In **reinforcement learning (RL)** there is no answer key, but your reinforcement learning agent still has to decide how to act to perform its task.

In the absence of existing training data, the agent learns from experience. It collects the training examples ("this action was good, that action was bad") through **trial-and-error** as it attempts its task with the goal of maximizing long-term **reward**.

In this final section of Machine Learning for Humans, we will explore:

- The exploration/exploitation tradeoff

- Markov Decision Processes (MDPs), the classic setting for RL tasks

- Q-learning, policy learning, and deep reinforcement learning

- The value learning problem


### A robot mouse in a maze

Suppose we are playing a game where our mouse is seeking the ultimate reward of cheese at the end of the maze (+1000 points) or the lesser reward of water along the way (+10 points). Meanwhile, robo-mouse wants to avoid locations that deliver an electric shock (-100 points).

After a bit of **exploration**, the mouse might find the mini-paradise of three water sources clustered near the entrance and spend all its time **exploiting** that discovery by continually racking up the small rewards of these water sources and never going further into the maze to pursue the larger prize.

This brings up the **exploration/exploitation tradeoff**. 

One simple strategy for exploration would be for the mouse to take the best known action most of the time (say, 80% of the time) but occasionally explore a new, randomly selected direction even though it might be walking away from known reward.

This strategy is called the **epsilon-greedy strategy** where **epsilon** is the percent of the time that the agent takes a randomly selected action rather than taking the action that is most likely to maximize reward given what it knows so far (in this case, 20%). 

We usually start with a lot of exploration (a higher value for epsilon). Over time, as the mouse learns more about the maze and which actions yield the most long-term reward, it would make sense to steadily reduce epsilon to 10% or even lower as it settles into exploiting what it knows.

It is important to keep in mind that the reward is not always immediate: in the robot-mouse example, there might be a long stretch of the maze you have to walk through and several decision points before you reach the cheese.


### Markov Decision Processes (MDPs)

The mouse’s wandering through the maze can be formalized as a **Markov Decision Process** which is a process that has specified transition probabilities from state to state.

MDPs include:

- A finite set of states which are the possible positions of our mouse within the maze.

- A set of actions available in each state: {forward, back} in a corridor and {forward, back, left, right} at a crossroads.

- Transitions between states. 

  For example, if you go left at a crossroads you end up in a new position. 
  
  These can be a set of probabilities that link to more than one possible state (such as when you use an attack in a game of Pokémon you can either miss, inflict some damage, or inflict enough damage to knock out your opponent).
  
- Rewards associated with each transition.

  In the robot-mouse example, most of the rewards are 0, but they are positive if you reach a point that has water or cheese and negative if you reach a point that has an electric shock.

- A discount factor γ between 0 and 1. 

  This quantifies the difference in importance between immediate rewards and future rewards. 
  
  For example, if γ is .9 and there is a reward of 5 after 3 steps, the present value of that reward is (.9^3 * 5)
  
- Memorylessness. 

  Once the current state is known, the history of the mouse’s travels through the maze can be erased because the current Markov state contains all useful information from the history. 
  
  In other words, “the future is independent of the past given the present”.

Now that we know what an MDP is, we can formalize the mouse’s objective. 

We are trying to maximize the sum of rewards in the long term:

Now that we set up our reinforcement learning problem and formalized the goal, we can explore some possible solutions.


### Q-learning: learning the action-value function

**Q-learning** is a technique that evaluates which action to take based on an **action-value function** that determines the value of being in a certain state and taking a certain action at that state.

We have a function Q that takes as an input one state and one action and returns the **expected reward** of that action (and all subsequent actions) at that state. 

Before we explore the environment, Q gives the same (arbitrary) fixed value. But as we explore the environment, Q gives us a better and better approximation of the value of an action a at state s, so we update our function Q as we go.

This equation from the Wikipedia page on Q-learning explains it all very nicely. 

It shows how we update the value of Q based on the reward we get from our environment:

Let us ignore the discount factor γ by setting it to 1 again. 

First, keep in mind that Q is supposed to show you the full sum of rewards from choosing action Q and all the optimal actions afterward.

Now let us go through the equation from left to right. 

When we take action at in state st, we update our value of Q(st, at) by adding a term to it which contains:

- Learning rate **alpha**: this is how aggressive we want to be when updating our value. 

  When alpha is close to 0, we are not updating very aggressively. 
  
  When alpha is close to 1, we are simply replacing the old value with the updated value.

- The **reward** is the reward we got by taking action at at state st, so we are adding this reward to our old estimate.

- We are also adding the **estimated future reward** which is the maximum achievable reward Q for all available actions at st + 1.

- Finally, we subtract the old value of Q to make sure that we are only incrementing or decrementing by the difference in the estimate (multiplied by alpha of course).

Now that we have a value estimate for each state-action pair, we can select which action to take according to our **action-selection strategy** 

We do not necessarily just choose the action that leads to the most expected reward every time -- with an epsilon-greedy exploration strategy we would take a random action some percentage of the time. 

In the robot mouse example: 

  1. We can use Q-learning to figure out the value of each position in the maze and the value of the actions {forward, backward, left, right} at each position. 

  2. We can use our action-selection strategy to choose what the mouse actually does at each time step.


### Policy learning: a map from state to action

In the Q-learning approach, we learned a **value function** that estimated the value of each state-action pair.

**Policy learning** is a more straightforward alternative in which we learn a **policy function** π which is a direct map from each state to the best corresponding action at that state. 

Think of π as a behavioral policy: “when I observe state s, the best thing to do is take action a”. 

For example, an autonomous vehicle’s policy might effectively include something like: "if I see a yellow light and I am more than 100 feet from the intersection, I should brake. Otherwise, keep moving forward".

A policy is a map from state to action: a = π(s)

Thus, we are learning a function that will maximize expected reward. 

What do we know that is really good at learning complex functions? Deep neural networks!

Andrej Karpathy’s Pong from Pixels provides an excellent walkthrough on using **deep reinforcement learning** to learn a policy for the Atari game Pong that takes raw pixels from the game as the input (state) and outputs a probability of moving the paddle up or down (action).

In a policy gradient network, the agent learns the optimal policy by adjusting its weights through gradient descent based on reward signals from the environment.

If you want to get your hands dirty with deep RL, you can work through Andrej’s post which implements a 2-layer policy network in 130 lines of code, and you will also learn how to plug into OpenAI’s Gym which allows you to quickly get up and running with your first reinforcement learning algorithm, test it on a variety of games, and see how its performance compares to other submissions.


-----------


# [Three Baseline Policies Your Reinforcement Learning Algorithm Absolutely Should Outperform](https://towardsdatascience.com/three-baseline-policies-your-reinforcement-learning-algorithm-absolutely-should-outperform-d2ff4d1175b8)

The great appeal of RL is that it allows solving complicated _sequential_ decision problems. 

Determine the best action right now might be a straightforward problem but anticipating how that action keeps affecting our rewards and environment long afterwards is another matter.

Similar to the random policy, the myopic policy could be somewhat of a litmus test. In highly stochastic environments, the action you take today may have very limited effect on the world tomorrow, and even less the day after that. 

Transitions from one state to another typically contain a **stochastic and a deterministic component** -- if the stochastic component is large and mostly noise, there is little to be gained from predicting downstream effects. 

Contrasting policies with and without lookahead quantify the degree to which anticipating the future actually helps.

Radical developments of completely novel algorithms are rare. 

The solution for your problem is likely a tweaked version of some RL algorithm that already exists rather than something built from scratch. 

It is good to emphasize there is a stark difference between a baseline and a competitive benchmark. 

The baselines mentioned in this article should be considerably outperformed to demonstrate that your RL algorithm learns _something_ useful which is not enough to show it is a good solution though. 

If you want to publish an academic paper or want to upgrade your companies’ planning system, you had better contrast your algorithm to some serious competitors.

