# AI Project Planning

## Background

- 95% or more of AI projects fail (Pilot paradox)
- 90% of DevOps projects fail
- 80% or more of software projects fail
- 268% Higher Failure Rates for Agile Software Projects

- Why AI Projects Fail
- What are the risks with AI
- When you should not use AI

- Ethics Crisis in AI Research
- Ethics Issues from AI (bias, safety, etc.)

The student of AI needs the following:

- A healthy dose of skepticism [4].

- Knowledge of the history and reasons that led to the _AI Winter_.

- Be aware of the hype, misinformation, and  alchemy that is common in the field of AI.

The two guiding principles of AI engineering are:

1. Occam's Razor
2. No Free Lunch Theorem

## The AI Collaboration Matrix

The AI Collaboration Matrix is a straightforward way to visualize and guide your AI journey [5].

The AI Collaboration Matrix examines two simple questions: who is making your stuff (humans or AI?) and who ia running the show (humans or AI?).

By mapping where we are today, we can avoid random AI adoption, spot opportunities to level up, and explain our AI strategy to stakeholders without their eyes glazing over.

As AI evolves from a helpful assistant to a teammate who gets things done, this framework helps us navigate the transition without getting lost.

Most productivity frameworks we use today were built for a human-centric world. They measure human effort, human decision-making, and human-created output.

As AI plays an increasingly central role in creating work products and managing workflows across all knowledge domains, we need a new way to understand and track its impact.

The AI Collaboration Matrix provides a snapshot of AI usage and a flexible, forward-looking framework to help organizations measure their progress toward an AI-driven future where human and AI collaboration is the norm.

By placing AI’s role on a two-dimensional spectrum that compares human vs. AI-produced resources and human vs. AI-led processes, we can track an organization’s journey toward fully agentic, AI-powered work and identify where to focus next.

## Tips for Successful AI Projects

Using machine learning to help your business achieve edge on competition requires a plan.

### Focus on the Business Problem

- Identify Business Problem
- Identify hidden data resources that you can take advantage of

### The Machine Learning Cycle

The Machine Learning Lifecycle is a structured process used to develop, train, deploy and maintain machine learning models efficiently [3].

- Provides a systematic workflow for building scalable and reliable ML models
- Helps continuously improve model performance through monitoring and retraining

### Pilot Project

- Define an opportunity for growth
- Conduct pilot project with your concrete idea from Step 1
- Evaluation
- Next actions

### Determining the Best Learning Model

Here are some  tips for choosing machine learning models [8]:

- Understand the data
- Define the Problem Clearly
- Start Simple
- Evaluate Multiple Models
- Consider Computational Resources

### Choosing the Right Algorithm for Your Problem

Here is a Question-Based Template designed to guide AI, ML, and data analysis project leaders to the right choice of ML algorithm to use for addressing their specific problem [9].

There is also a visual, decision tree-based guide to navigate you towards the most suitable machine learning algorithm for your task, depending on the nature and complexity of your available data [10].

#### Key Question 1: What type of problem do you need to solve?

- 1.A. Do you need to predict something?
- 1.B. If so, is it a numerical value, or classification into categories?
- 1.C. If you want to predict a numerical value, is it based on other variables or features? Or are you predicting future values based on past historical ones?

#### Key Question 2: What type of data do you have?

2.A. Structured and simpler data arranged in tables with few attributes can be leveraged with simple ML algorithms like linear regression, decision tree classifiers, k-means clustering, etc.

2.B. Data with intermediate complexity (structured but having dozens of attributes, or low-resolution images) can be addressed with ensemble methods for classification and regression, which combine multiple ML model instances into one to attain better predictive results.

Examples of ensemble methods are random forests, gradient boosting, and XGBoost. For other tasks like clustering, try algorithms like DBSCAN or spectral clustering.

2.C. Last, highly complex data such as images, text, and audio usually require more advanced architectures such as deep neural networks or even transformer-based architectures like large language models (LLMs

#### Key Question 3: What level of interpretability do you need?

In certain contexts where it is important to understand how an ML algorithm makes decisions like predictions, which input factors influence the decision, and how, interpretability is another important aspect that may influence your algorithm choice.

Rule of thumb: the simpler the algorithm, the more interpretable.

#### Key Question 4: What volume of data do you handle?

This one is closely related to key question 2. Some ML algorithms are more efficient than others, depending on the volume of data used for training them.

In contast, complex options such as neural networks normally require larger amounts of data to learn to perform the task they are built for, even at the cost of sacrificing efficient training.

A good rule here is that data volume is in most cases tightly related to data complexity when it comes to choosing the right type of algorithm.

#### Application Examples

Here is a table with some real-world use cases where the decision factors considered in this article are outlined [10]:

### Tools to perform algorithm selection

TODO: Add notes on AutoML tools for model selection.

## AI Use Cases

Here are 5 AI features you can build that add immediate value to an application [4]:

1. Vector search

   Vector search is a technique used to search for similar items (text, images, audio, etc) in a database.

2. Filter with AI

   An "Ask AI" input search allows your users to filter the dashboard with a free-text filter.

   Product teams have usually solved this by adding a "Filter by" section to the dashboard to allow users to choose a dimension and a value or set of values. But when you have dozens of dimensions, this can become impractical.

3. Visualize with AI

   Once you start using LLM queries for structured outputs, you see a lot of opportunities to use them for other features.

   Instead of providing a predefined dashboard, for example, you can let your users ask AI to visualize the data the way they want ot.

4. Auto-fix with AI

   This is a feature that you can find in many developer tools (such as Tinybird or Cursor).

   When building with a devtool, we often make mistakes all the time: syntax errors, missing imports, etc.

   But LLM models are not deterministic, and they can give you the wrong answer. Therefore, make the LLM evaluate if the answer is correct.

5. Explain with AI

   State-of-the-art LLMs are sometimes terrible at trivial tasks but great at explaining complex concepts, or describing your own data, or gathering multiple sources of information.

Most technical companies have the following:

- Public product documentation
- API, CLI, SDKs references
- Internal knowledge base
- Internal documentation and wikis
- Issue trackers, PRs, and PRDs
- Slack channels, internal discussions, etc.

There is a single source of truth for technical questions: _the code_.

When integrated with an LLM, this info can be used to answer most of the support requests from users.

## When not to use ML

The article [1] discusses four reasons when you should not use machine learning.

### Data-related issues

In the [AI hierarchy of needs][^ai_hierarchy], it is important that you have a robust process for collecting, storing, moving, and transforming data. Otherwise, GIGO.

Not only do you need your data to be **reliable** but you need **enough** data to leverage the power of machine learning.

### Interpretability

There are two main categories of ML models:

- Predictive models focus on the model’s ability to produce accurate predictions.

- Explanatory models focus on understanding the relationships between the variables in the data.

ML models (especially ensemble models and neural networks) are predictive models that are much better at predictions than traditional models such as linear/logistic regression.

However, when it comes to understanding the relationships between the predictive variables and the target variable, these models are a _black box_.

You may understand the underlying mechanics behind these models, but it is still not clear how they get to their final results.

In general, ML and deep learning models are great for prediction but lack explainability.

### Technical Debt

Maintaining ML models over time can be challenging and expensive.

There are several types of debt to consider when maintaining ML models:

- **Dependency debt:** The cost of maintaining multiple versions of the same model, legacy features, and underutilized packages.

- **Analysis debt:** This refers to the idea that ML systems often end up influencing their own behavior if they update over time, resulting in direct and hidden feedback loops.

- **Configuration debt:** The configuration of ML systems incur a debt similar to any software system.

### Better Alternatives

ML should not be used when simpler alternatives exist that are equally as effective.

You should start with the simplest solution that you can implement and iteratively determine if the marginal benefits from the next best alternative outweighs the marginal costs.

> Simpler = Better (Occam's Razor)

## Lessons Learned

The following are some lessons learned in [6].

### The Pilot Paradox

Projects stayed siloed as pilot or POC and were never deployed to prodyction.

LESSON: Teams never defined success or ROI.

What Actually Worked: Solving specific, unglamorous problems with clear success metrics.

The biggest returns came from the least revolutionary applications: automating repetitive internal tasks, extracting insight from messy datasets, accelerating early-phase ideation.

MIT’s Work of the Future initiative:

- Organizations using AI to replace workers saw productivity gains of 8–12%.

- Organisations using AI to augment workers (enhance their decision-making, test their assumptions, challenge their cognitive biases) saw gains of 35–40%.

### The Counterintuitive Move: Slowing Down to Win

The most strategically sound organisations are deliberately slowing down.

They are asking different questions:

- What are we actually trying to solve with this technology?
- Who is accountable when it produces unexpected outcomes?
- How does this tool align with our values, not just our KPIs?
- What human capabilities need strengthening before we add machine capabilities?

### The Governance Gap: The Phase 2 Problem

Ethics, compliance, explainability? They were “Phase 2” problems. We will sort them out later, once we’d proven value.

Harvard Business School’s research on AI governance found that organisations with clear AI principles and decision-making frameworks were three times more likely to successfully scale AI implementations beyond pilot phase.

The organizations winning with AI in 2026 are not those moving fastest: they are companies who built the governance infrastructure first, then accelerated within it.

### Need for Strategic Thinking

> The $847 Million Question: According to Gartner’s post-mortem analysis, organisations collectively wasted $847M in 2025 on AI implementations that never delivered measurable value.

Strategic thinking is the new scarce resource. AI just makes its absence more expensive.

The businesses that will do best in 2026 and beyond will be the ones that build systems where human insight and machine capability work in genuine tandem, responsibly, repeatably, and with clear strategic purpose.

## Mistakes to Avoid in AI

Here are eight mistakes to avoid when using machine learning [2]:

1. Not understanding the user

   You must understand from the beginning what the user or business really wants.

2. Not performing failure analysis

   If you do not perform a failure analysis (an analysis of the frequency of different categories of failure of your system) you may be expending a lot of effort for little result.

3. Not looking at the model

   Clearly look for the weights and splits which may end up causing you to choose the wrong model

4. Not using existing solutions

   Explore the existing solutions from the major technology companies. It is not always a good idea to create unique solutions.

5. Not comparing to a simple baseline model

   It is natural to want to start with a complex model. But sometimes a single neutron(logistic regression) performs as well as a deep neural network with six hidden layers

6. Not looking for data leakage

   In case of data leakage, the proper information or clues wont be available at the time of prediction, as a result wrong solution would come

7. Not looking at the data

   When you do not look at the data carefully, you can miss useful insights which will lead to a data error and missing data

8. Not qualifying the use case

   Before starting a machine learning project, it is important to determine whether the project is worth doing and to consider its ramifications.

## Using AI Tools for Development

Faster code does not mean better engineering [7].

AI does not fix poor engineering habits [7].

If a team already skips design discussions, ignores observability, avoids refactoring, merges weak pull requests, and treats production issues as surprises, AI will not solve the root problem. AI will simply create bad code faster.

AI is a multiplier and not a foundation.

AI tools work best when they operate inside strong engineering boundaries.

AI improves disciplined teams more than undisciplined ones.

AI does not replace discipline [7]:

- It does not define the problem.
- It does not protect your architecture.
- It does not guarantee secure code.
- It does not create meaningful tests by itself.
- It does not understand your users unless you bring that context.
- It does not own the production system after launch.

## References

[1]: https://towardsdatascience.com/4-reasons-why-you-shouldnt-use-machine-learning-639d1d99fe11 "4 Reasons Why You Shouldn't Use Machine Learning"

[2]: https://medium.com/@monodeepets77/8-mistakes-to-avoid-while-using-machine-learning-d61af954b9c9 "8 Mistakes to avoid while using Machine Learning"

[3]: https://www.geeksforgeeks.org/machine-learning/machine-learning-lifecycle/ "Machine Learning Lifecycle"

[4]: https://www.tinybird.co/blog-posts/ai-features-that-work/ "Hype v. Reality: 5 AI features that actually work in production"

[5]: https://devinterrupted.substack.com/p/the-matrix-that-makes-your-ai-strategy "The Matrix That Makes Your AI Strategy Make Sense"

[6]: https://medium.com/codex/the-multi-million-question-nobodys-asking-about-ai-b3501979c5b9?source=rss----29038077e4c6---4 "The Multi Million Question Nobody’s Asking About AI"

[7]: https://pub.towardsai.net/ai-will-not-fix-a-team-that-lacks-engineering-discipline-db1697eca7c5?source=rss----98111c9905da---4 "AI Will Not Fix a Team That Lacks Engineering Discipline"

[8]: https://machinelearningmastery.com/tips-for-choosing-the-right-machine-learning-model-for-your-data/ "Tips for Choosing the Right Machine Learning Model for Your Data"

[9]: https://machinelearningmastery.com/practical-guide-choosing-right-algorithm-your-problem/ "A Practical Guide to Choosing the Right Algorithm for Your Problem"

[10]: https://www.kdnuggets.com/choosing-the-right-machine-learning-algorithm-a-decision-tree-approach "Choosing the Right Machine Learning Algorithm: A Decision Tree Approach"

-----

[Strategic ROI Assessment for AI Projects: A Pre-Implementation Framework](https://pub.towardsai.net/strategic-roi-assessment-for-ai-projects-a-pre-implementation-framework-24dd9827d140)

[How Strategic AI Consulting Converts Potential into Performance](https://pub.towardsai.net/how-strategic-ai-consulting-converts-potential-into-performance-aeb0c2019a44?source=rss----98111c9905da---4)

[Governing the Unseen Risks of GenAI: Why Bias Mitigation and Human Oversight Matter Most](https://securityboulevard.com/2025/11/governing-the-unseen-risks-of-genai-why-bias-mitigation-and-human-oversight-matter-most/)

-----

[Generative AI System Design Interview: A Step-by-Step Guide](https://www.systemdesignhandbook.com/guides/generative-ai-system-design-interview/)

[^ai_hierarchy]: https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007 "The AI Hierarchy of Needs"
