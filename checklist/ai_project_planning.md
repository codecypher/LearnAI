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

The AI Collaboration Matrix is a straightforward way to visualize and guide your AI journey [23].

The AI Collaboration Matrix examines two simple questions: who is making your stuff (humans or AI?) and who ia running the show (humans or AI?).

By mapping where we are today, we can avoid random AI adoption, spot opportunities to level up, and explain our AI strategy to stakeholders without their eyes glazing over.

As AI evolves from a helpful assistant to a teammate who gets things done, this framework helps us navigate the transition without getting lost.

Most productivity frameworks we use today were built for a human-centric world. They measure human effort, human decision-making, and human-created output.

As AI plays an increasingly central role in creating work products and managing workflows across all knowledge domains, we need a new way to understand and track its impact.

The AI Collaboration Matrix provides a snapshot of AI usage and a flexible, forward-looking framework to help organizations measure their progress toward an AI-driven future where human and AI collaboration is the norm.

By placing AI’s role on a two-dimensional spectrum that compares human vs. AI-produced resources and human vs. AI-led processes, we can track an organization’s journey toward fully agentic, AI-powered work and identify where to focus next.

## 5 Steps for Successful AI Projects

Using machine learning to help your business achieve edge on competition requires a plan and roadmap [13].

You cannot simply hire a group of data scientists and hope that they will be able to produce results for the business.

1. Focus on the Business Problem

   Identify Business Problem

   Where are the hidden data resources that you can take advantage of?

2. The Machine Learning Cycle

3. Pilot Project

   Step 1: Define an opportunity for growth
   Step 2: Conduct pilot project with your concrete idea from Step 1
   Step 3: Evaluation
   Step 4: Next actions

4. Determining the Best Learning Model

5. Tools to determine algorithm selection

## AI Use Cases

Here are 5 AI features you can build that add immediate value to an application [21]:

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

## Integrating LLMs into Software System

Despite the many advantages of integrating LLMs into existing software, there are many risks that need to be considered [10].

> The tendency to make things up is holding chatbots back. But that’s just what they do. (MIT Technology Review)

- Computational costs of training models and during model inferences due to heavy reliance on high-end GPUs and TPUs

- Making frequent API calls can be expensive, especially for high-traffic applications

- If sensitive data is sent to an LLM, it might be processed, stored, and exposed to another user, especially when the LLM being utilized is accessible to the public

- Aside from properly fine-tuned custom models, most LLMs can only provide open-ended and generic responses and cannot provide domain-based knowledge

- Training LLMs require high energy usage, which can lead to high carbon emissions and environmental pollution

### Tips for LLM Integration

Here is a guide for the process of integrating LLMs into your systems [22]:

1. Identify suitable use cases for LLMs within your existing systems.
2. Select the right LLM for your specific needs.
3. Integrate LLMs with your current architecture seamlessly.
4. Leverage the unique capabilities of LLMs to enhance your system’s functionality.

- Integrating LLMs into System Architecture
- Data Cleanup and Enrichment
- Prompt Templating
- Generalized Problem Solving
- Systems that can Adapt and Grow

#### Working with Non-Determinism

Beyond testing and iterating on your prompts, you can design systems that anticipate and manage non-deterministic outputs from LLMs. We already have experience building fault-tolerant systems, so extending this mindset to handle the variability of LLM responses is a natural next step.

By incorporating strategies such as:

- Output validation
- Fallback mechanisms
- Confidence thresholding

We can create robust systems that leverage the power of LLMs while accounting for their inherent non-determinism.

#### Stigmergy: Inspiring Adaptive Systems

In nature, stigmergy is a concept where the trace left in the environment by an individual action stimulates subsequent actions by the same or different agents. This principle can be effectively applied to building adaptive systems using LLMs and AI models.

Rather than defining explicit behaviors, we can harness LLMs to create systems that evolve through data-driven traces left in their environment. This approach brings us closer to developing more generalized, adaptable systems that grow organically.

We can achieve this using the following:

1. Establish basic constraints and rules to enable the subsystem’s initial operation.

2. Leave sufficient environmental traces for the system to infer its next steps autonomously.

3. Leverage LLMs to fill gaps or resolve ambiguities, guiding the system through small, incremental progressions.

These incremental steps can number in the thousands within a given system. The potential of a stigmergic system is both fascinating and promising.

The stigmergic system represents a new paradigm in system building – one that embraces the generative nature of LLMs to create reliable, organic digital systems.

By adopting this approach, we shift from prescribing every detail to fostering adaptive growth, allowing our systems to evolve and improve continually.


## Lessons Learned

Here are some lessons learned in [24].

### The Pilot Paradox

Projects stayed siloed as pilot or POC and were never deployed to prodyction [24].

LESSON: Teams never defined success or ROI [24].

What Actually Worked: Solving specific, unglamorous problems with clear success metrics [24].

The biggest returns came from the least revolutionary applications: automating repetitive internal tasks, extracting insight from messy datasets, accelerating early-phase ideation [24].

MIT’s Work of the Future initiative [24]:

- Organizations using AI to replace workers saw productivity gains of 8–12%.

- Organisations using AI to augment workers (enhance their decision-making, test their assumptions, challenge their cognitive biases) saw gains of 35–40%.

### The Counterintuitive Move: Slowing Down to Win

The most strategically sound organisations are deliberately slowing down.

They are asking different questions [24]:

- What are we actually trying to solve with this technology?

- Who is accountable when it produces unexpected outcomes?

- How does this tool align with our values, not just our KPIs?

- What human capabilities need strengthening before we add machine capabilities?

### The Governance Gap: The Phase 2 Problem

Ethics, compliance, explainability? They were “Phase 2” problems. We will sort them out later, once we’d proven value.

Harvard Business School’s research on AI governance found that organisations with clear AI principles and decision-making frameworks were three times more likely to successfully scale AI implementations beyond pilot phase [24].

The organizations winning with AI in 2026 are not those moving fastest: they are companies who built the governance infrastructure first, then accelerated within it [24].

### Need for Strategic Thinking

> The $847 Million Question: According to Gartner’s post-mortem analysis, organisations collectively wasted $847M in 2025 on AI implementations that never delivered measurable value.

Strategic thinking is the new scarce resource. AI just makes its absence more expensive [24].

The businesses that will do best in 2026 and beyond will be the ones that build systems where human insight and machine capability work in genuine tandem, responsibly, repeatably, and with clear strategic purpose [24].

## Mistakes to Avoid in AI

Here are eight mistakes to avoid when using machine learning [12]:

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

## References

[1]: T. Shin, [4 Reasons Why You Shouldn't Use Machine Learning](https://towardsdatascience.com/4-reasons-why-you-shouldnt-use-machine-learning-639d1d99fe11), Towards Data Science, Oct 5, 2021.

----------

[10]: [Integrating Language Models into Existing Software Systems](https://www.kdnuggets.com/integrating-language-models-into-existing-software-systems)

[12]: [8 Mistakes to avoid while using Machine Learning](https://medium.com/@monodeepets77/8-mistakes-to-avoid-while-using-machine-learning-d61af954b9c9)

[13]: [5 Steps to follow for Successful Machine Learning Project](https://addiai.com/successful-machine-learning-project/)

[21]: A. Romeu, [Hype v. Reality: 5 AI features that actually work in production](https://www.tinybird.co/blog-posts/ai-features-that-work), tinybird, April 2, 2025.

[22]: [LLMs - A Ghost in the Machine](https://zacksiri.dev/posts/llms-a-ghost-in-the-machine/)

[23]: B. Pearson and O. Affias, “The Matrix That Makes Your AI Strategy Make Sense,” Dev Interrupted, April 3, 2025.

[24]: C. Cooper, [The Multi Million Question Nobody’s Asking About AI](https://medium.com/codex/the-multi-million-question-nobodys-asking-about-ai-b3501979c5b9?source=rss----29038077e4c6---4), CodeX, Dec. 4, 2025.

[Thirty Years, Five Technologies, One Failure Pattern: From Lean to AI](https://itnext.io/thirty-years-five-technologies-one-failure-pattern-from-lean-to-ai-628b8d7195a1)

[One real reason AI isn't delivering: Meatbags in manglement](https://www.theregister.com/2025/12/24/reason_ai_isnt_delivering/)

----------

[How Strategic AI Consulting Converts Potential into Performance](https://pub.towardsai.net/how-strategic-ai-consulting-converts-potential-into-performance-aeb0c2019a44?source=rss----98111c9905da---4)

[Why AI projects fail, and how developers can help them succeed](https://www.infoworld.com/article/4010313/why-ai-projects-fail-and-how-developers-can-help-them-succeed.html)

[Governing the Unseen Risks of GenAI: Why Bias Mitigation and Human Oversight Matter Most](https://securityboulevard.com/2025/11/governing-the-unseen-risks-of-genai-why-bias-mitigation-and-human-oversight-matter-most/)

----------

[^ai_hierarchy]: <https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007>
