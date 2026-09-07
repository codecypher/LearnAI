# Introduction to AI

## Key Facts

- 95% or more of AI projects fail [5], [6], [9]
- 85% or more of software projects fail [11], [12], [18] to [20]
- 90% of DevOps projects fail [13]
- 268% Higher Failure Rates for Agile Software Projects [14]

- Less than 1% of AI practitioners and researchers have an advanced degree in AI [TODO].

- 34% or more of of journal article authors admit to manipulating results [7].

- In general, the results of journal articles are irreproducible [7], [8].

## Background

As an AI/ML engineer, you should be willing to settle for “good enough” which is called _satisﬁcing_ rather than trying to find the “best” model or approach.

Scrum is a popular project management approach rather than a software development methodology [1]. 

A better approach is an iterative, agile feature-driven development (FDD) methodology where team members are able to work independently without the rigid constraints of Scrum [2].

Here are some lists of articles that cover some of the problems with **AI Engineering** which is required knowledge for an AI project to be successful.

## Key Concepts

The student and practioner of AI needs a healthy dose of skepticism [4].

> Any competent software engineer become proficient with any software development tool. The key to a successful AI project is knowlege of the theory and best practices for AI, especially the limitations.

The practioner of AI needs an expert knowledge of the following:

- The history and issues that led to the "AI Winter".
- The hype, misinformation, and  alchemy that is common with AI.
- The capabilities, risks, and limitations of AI.
- AI Research: Reproducibility, Fabrication, and Falsification
- The ethical implications, risks, and responsibilities of AI.

- Why AI Projects Fail
- How AI Projects are Different
- Risks of agile approach for AI projects
- AI alchemy = hype and false claims.
- Problems with Cloud AI

There are two guiding principles for AI engineering:

1. Occam’s Razor: The simplest algorithm that fits the data is usually the best.

2. No Free Lunch Theorem: There is no such thing as best, only good enough.

## Getting Started

J. Holmes, "[How to Learn AI](https://pub.towardsai.net/how-to-learn-ai-1b9814ed3681)," Towards AI, Aug 24, 2023.

J. Holmes, "[Getting Started with AI](https://pub.towardsai.net/getting-started-with-ai-f565c7877bee)," Towards AI, Aug 25, 2023.

A. Pillai, [Begin with problems, sandbox, identify trustworth vendors — a quick guide to getting started with AI](https://venturebeat.com/ai/begin-with-problems-sandbox-identify-trustworth-vendors-a-quick-guide-to-getting-started-with-ai/), VentureBeat, Feb. 8, 2025.

A. Romeu, [Hype v. Reality: 5 AI features that actually work in production](https://www.tinybird.co/blog-posts/ai-features-that-work), tinybird, April 2, 2025.

M. Jayasinghe, [Top Strategies for Building Scalable and Secure AI Applications](https://thenewstack.io/top-strategies-for-building-scalable-and-secure-ai-applications/), The New Stack, Feb. 5, 2025.

[Pragmatic AI Automation — Balancing Efficiency & Risk](https://blog.gopenai.com/pragmatic-ai-automation-balancing-efficiency-risk-d39c85333704), GoPenAI, Feb. 10, 2025.

T. Shin, [4 Reasons Why You Shouldn’t Use Machine Learning](https://towardsdatascience.com/4-reasons-why-you-shouldnt-use-machine-learning-639d1d99fe11/), Towards Data Science, Oct. 5, 2021.

[Recommended Resources](./Level-1/tips/ai_books.md)

## What is AI Engineering?

[What is Artificial Intelligence Engineering?](https://www.sei.cmu.edu/our-work/artificial-intelligence-engineering/)

A. Sandman, "[Is Agile dead in the age of AI?](https://sdtimes.com/agile/is-agile-dead-in-the-age-of-ai/)," SD Times, Aug. 1, 2025.

R. Ramesh, "[AI’s Blind Spot: When Models Ignore Causal Relationships and Settle for Correlations]," Medium, Aug. 6, 2025.

J. B. Michael and M. Orescanin, "[Developing and Deploying Artificial Intelligence Systems](https://ieeexplore.ieee.org/document/9789299)," IEEE Computer, vol. 55, no. 6, pp. 15-17, June 2022, doi: 10.1109/MC.2022.3166488.

P. Ferguson, [AI Development vs Software Engineering: Key Differences Explained](https://medium.com/towards-data-science/ai-development-vs-software-engineering-key-differences-explained-0709633e81d2), Towards Data Science, Jan. 31, 2025.

L. Ellen, [When OpenAI Isn’t Always the Answer: Enterprise Risks Behind Wrapper-Based AI Agents](https://towardsdatascience.com/when-openai-isnt-always-the-answer-enterprise-risks-behind-wrapper-based-ai-agents/), Towards Data Science, April 28, 2025.


## AI Best Practices

Here are some best practiced for AI agents [17]:

- Treat every tool an agent has access to as a scoped, individually justified permission rather than a broad capability granted once and never revisited. 

- Build the escalation path (the circuit breaker) with an explicit threshold where autonomous action stops and human review begins before you scale usage.

- Design for the silent-failure pattern the research keeps surfacing, an agent’s own logs knowing something went wrong is worthless if that knowledge never reaches a human in a form they can actually act on in time.

Building the resilient boundary around autonomous systems rather than trusting prompt-level instructions to hold under real, adversarial pressure: "AI for Production Incidents: The Senior Engineer’s Safe Workflow".


## Why AI Projects Fail?

First, we need to understand why AI projects fail [5], [6], [9], [10], [15].

- Gartner finds that 85% of AI projects generally fail to reach production [17]. 

- McKinsey finds that fewer than 20% of AI pilots scale to production within eighteen months [17].

Projects building security architecture concurrently with agent development, rather than retrofitting it after the fact, are roughly four times more likely to pass enterprise security review without timeline-destroying delays with an upfront cost of 15 to 20% of total development effort, against retrofitting costs that frequently exceed 60% of the original budget once security gets bolted on after the fact [17].

J. Dabass, [The $1 Trillion Mistake: Why 90% of AI Projects Will Fail](https://medium.com/tech-ai-made-easy/the-1-trillion-mistake-why-90-of-ai-projects-will-fail-d405dec77970), AI in Plain English, Aug. 21, 2025.

K. Ahuja, [Why a $1.2M AI Project Failed (And How to Avoid the Same Mistake)](https://pub.towardsai.net/why-a-1-2m-ai-project-failed-and-how-to-avoid-the-same-mistake-c873235b5d1d), Towards AI, Aug. 24, 2025.

I. Bernardo, [Why AI Projects Fail](https://towardsdatascience.com/why-ai-projects-fail/), Towards Data Science, June 6, 2025.

Y. Kosarenko, [The majority of business analytics and AI projects are still failing](https://www.datadriveninvestor.com/2020/04/30/the-majority-of-business-analytics-and-ai-projects-are-still-failing/), Data Driven Investor, April 30, 2020.

[Failed Machine Learning (FML)](https://github.com/kennethleungty/Failed-ML), GitHub, kennethleungty/Failed-ML.

S. Mulligan, [AI trained on AI garbage spits out AI garbage](https://www.technologyreview.com/2024/07/24/1095263/ai-that-feeds-on-a-diet-of-ai-garbage-ends-up-spitting-out-nonsense/),
MIT Technology Review, July 24, 2024.


## What are the Risks with AI?

O. Enuku, [The Dark Side of Model Evaluation That Nobody Talks About](https://blog.gopenai.com/the-dark-side-of-model-evaluation-that-nobody-talks-about-b2050ccf0814), GoPenAI, Dec. 22, 2024.

S. De Simone, [GenAI Increases Workloads and Decreases Productivity, Upwork Study Finds](https://www.infoq.com/news/2024/07/genai-hampers-productivity-study/), InfoQ, July 29, 2024.

D. Ferraro, "[Uncontrolled Artificial Intelligence: Big Tech Companies Fail on Safety (Part One)](https://www.codemotion.com/magazine/cybersecurity/uncontrolled-artificial-intelligence-big-tech-companies-fail-on-safety-part-one/)," codemotion, Nov 25, 2024.

M. Kumaran, "[AI Models Are Blackmailing Their Own Companies (And It’s Getting Worse)](https://pub.towardsai.net/ai-models-are-blackmailing-their-own-companies-and-its-getting-worse-c38cfb37d842?source=rss----98111c9905da---4)," Towards AI, July 11, 2025.

D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D.  Ebner, V. Chaudhary, and M. Young
[Machine Learning: The High Interest Credit Card of Technical Debt](https://research.google.com/pubs/pub43146.html?authuser=2), SE4ML: Software Engineering for Machine Learning (NIPS 2014 Workshop), 2014.

M. Troller, [Beware AI’s hidden costs before they bankrupt innovation](https://techcrunch.com/2023/12/27/beware-ais-hidden-costs-before-they-bankrupt-innovation/), techcrunch, Dec. 27, 2023.

B. Cheatham, K. Javanmardian, and H. Samandari, [Confronting the risks of artificial intelligence](https://www.mckinsey.com/capabilities/quantumblack/our-insights/confronting-the-risks-of-artificial-intelligence), McKinsey Quarterly, April 26, 2019.

## Hidden Problems with Sofware Projects

E. Gent, [Public AI Training Datasets Are Rife With Licensing Errors](https://spectrum.ieee.org/data-ai), IEEE Spectrum, Nov. 8, 2023.

C. Y. Laporte, G. Verret, and M. Muñoz, "[A Software Project That Partially Failed: A Small Organization That Ignored the Management and Technical Practices of Software Standards](https://ieeexplore.ieee.org/document/10109288)," Computer, vol. 56, no. 5, pp. 138-144, May 2023, doi: 10.1109/MC.2023.3253979.

B. Hubert, "[Why Bloat is Still Software's Biggest Vulnerability](https://spectrum.ieee.org/lean-software-development)," IEEE Spectrum, vol. 61, no. 4, pp. 22-50, April 2024, doi: 10.1109/MSPEC.2024.10491389.


## Limitations of GenAI

AI can generate ideas, write code, compose images, summarize research, prototype interfaces, and produce entire design systems in minutes.

The most important thing to understand about AI right now are:

- Speed is not the same as quality.
- Scale is not the same as judgment.
- AI does not remove the need for expertise; It increases the need for expertise.
- The constraint is no longer production; The constraint is discernment.
- The bottleneck is no longer making things; It is deciding what should exist.
- AI is not intelligence in the way humans experience intelligence; It is a pattern engine.
- AI inherits the biases, noise, and limitations of its training data
- AI amplifies the capability of the person using it.
- AI lowers the barrier to producing convincing work.
- AI can create an illusion of expertise.
- The difference between good and great work is rarely visible at the surface level.
- When a signal is amplified without control, it gets louder and distorted (signal integrity)

For most of the past century, the bottleneck in creative and knowledge work was production.

Designers explored less because time was limited. Writers produced fewer drafts because writing was laborious. Engineers prototyped less because building systems took effort. The constraint is no longer production; The constraint is discernment.

As AI takes over scale and speed, the real limitation becomes the human ability to interpret, evaluate, and refine what the machine produces. The bottleneck is no longer making things; It is deciding what should exist.

AI is not intelligence in the way humans experience intelligence; It is a pattern engine. AI predicts what comes next based on what it has seen before.

That makes AI incredibly powerful, but it also means it inherits the biases, noise, and limitations of its training data.

This is why generative systems can produce outputs that appear coherent but are ultimately shallow, inaccurate, or misaligned.

Which leads to a fundamental truth:

AI amplifies the capability of the person using it.

- If the user has strong expertise, taste, and judgment, AI accelerates excellence.
- If the user lacks those things, AI accelerates mediocrity.

One of the more subtle dangers of AI is that it lowers the barrier to producing convincing work which can hide weak thinking underneath.

Some analysts have warned that overreliance on AI may even create an illusion of expertise where individuals appear capable without actually developing the underlying skills required for deep problem solving.

This is particularly risky in fields where craft and judgment matter -- design, writing, product strategy, research, engineering.

Because the difference between good and great work is rarely visible at the surface level.

In audio engineering, there is a concept called signal integrity.

- When a signal is amplified cleanly, it becomes clearer, stronger, more precise.
- When a signal is amplified without control, it gets louder and distorted.
- Noise increases. Clarity breaks down. The output degrades.

AI does not judge the quality of what it is amplifying; It only increases it.

This means the outcome is entirely dependent on the quality of the input and the discernment of the person shaping it.

The difference is not the tool; It is the signal being amplified.

As AI tools become more powerful, the role of the human expert is changing.
We are moving from being the primary producers of work to becoming curators of possibility.

The job is no longer simply to make things:

- Ask the right questions
- Frame the problem correctly
- Evaluate quality with discernment
- Shape raw output into meaningful outcomes
- Ensure coherence, integrity, and craft

The role of the expert is evolving into something more specific:

- Not just creator.
- Not just operator.
- Signal guardian.
- The responsibility is to protect signal integrity.
- To ensure that what gets amplified is actually worth amplifying.

- AI will scale whatever it is given.
- The system does not decide; The human does.

The deeper question for creative industries is not whether AI will change how we work; It already has.
The real question is whether we will raise our standards along with it.

When everyone can generate something instantly, the value shifts but it will never replace human discernment.

In the end, without judgment, taste, and experience, AI simply scales mediocrity.

## References

[1]: I. Sommerville, Software Engineering 10th ed., Pearson, ISBN: 978-0133943030, 2015.

[2]: P. Bourque and R. E. Fairley, [Guide to the Software Engineering Body of Knowledge (SWEBOK)](https://www.computer.org/education/bodies-of-knowledge/software-engineering), v. 3, IEEE, 2014.

[3]: E. Alpaydin, Introduction to Machine Learning, 4th ed., MIT Press, ISBN: 9780262358064, 2020.

[4]: S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Upper Saddle River, NJ: Prentice Hall, ISBN: 0-13-461099-7, 2021.

[5]: B. M. Nedgu, “Why 85% of AI projects fail,” Towards Data Science Nov. 11, 2020.

[6]: S. Reisner, “Why most AI implementations fail and what enterprises can do to beat the odds,” Venture Beat, June 28, 2021.

[7]: J. F. DeFranco and J. Voas, “Reproducibility, Fabrication, and Falsification,” Computer, vol. 54 no. 12, 2021.

[8]: M. Parashar, M. A. Heroux,and V. Stodden, "Research Reproducibility," Computer, vol. 55, no. 8, pp. 16-18, Aug. 2022, doi: 10.1109/MC.2022.3176988.

[9]: S. Estrada, "[MIT report: 95% of generative AI pilots at companies are failing](https://fortune.com/2025/08/18/mit-report-95-percent-generative-ai-pilots-at-companies-failing-cfo/)," Fortune, Aug 18, 2025.

[10]: A. DeNisco Rayome, [Why 85% of AI projects fail](https://www.techrepublic.com/article/why-85-of-ai-projects-fail/), TechRepublic, June 20, 2019.

[11]: https://www.linkedin.com/pulse/why-80-software-projects-fail-how-avoid-bitsolutionss-y8m5f "Why 80% of Software Projects Fail and How to Avoid It"

[12]: https://dev.to/ekele/why-90-of-software-development-projects-fail-and-how-you-can-avoid-it-3dnb "Why 90% of Software Development Projects Fail"

[13]: B. Gain, [Most DevOps Plans Fail, but Things Are Getting Better](https://thenewstack.io/most-devops-plans-fail-but-things-are-getting-better/), The New Stack, Nov. 30, 2021.

[14]: J. Ali, [268% Higher Failure Rates for Agile Software Projects, Study Finds](https://www.engprax.com/post/268-higher-failure-rates-for-agile-software-projects-study-finds/), Impact Engineering, Engprax Ltd, ISBN-10: 106860574X, July 14, 2024.

[15]: https://www.infoworld.com/article/4010313/why-ai-projects-fail-and-how-developers-can-help-them-succeed.html "Why AI projects fail, and how developers can help them succeed"

[16]: https://medium.com/@aesthetikal/without-judgment-taste-and-experience-ai-simply-scales-mediocrity-91ef6da48a68 "Without judgment, taste, and experience, AI simply scales mediocrity"

[17]: https://medium.com/codex/65-of-companies-had-an-ai-agent-security-incident-this-year-6bc883b50f25 "65% of Companies Had an AI Agent Security Incident This Year"

[18]: https://opencommons.org/CHAOS_Report_on_IT_Project_Outcomes "CHAOS Report on IT Project Outcomes"

[18]: https://rockstardeveloperuniversity.com/software-project-failure-statistics/ "Software Project Failure Statistics 2026"

[20]: https://digitaloctopusgroup.com/breaking-down-the-70-failure-rate-in-software-development/ "Breaking Down the 70% Failure Rate in Software Development"

-----

[Chaos Report — why this study about IT project management is so unique](https://thestory.is/en/journal/chaos-report/)

[Project Failure Statistics](https://gitnux.org/project-failure-statistics/)

[Software Project Failure Statistics](https://gitnux.org/software-project-failure-statistics/)

[70% of Software Projects Fail: 2026 Reality Check](https://codeandcoffe.com/70-of-software-projects-fail-2026-reality-check/)

[Why software projects fail? The real causes of failure that leaders often ignore](https://ardura.consulting/blog/why-software-projects-fail-real-causes-leaders-ignore/)

[Thirty Years, Five Technologies, One Failure Pattern: From Lean to AI](https://itnext.io/thirty-years-five-technologies-one-failure-pattern-from-lean-to-ai-628b8d7195a1)

[One real reason AI is not delivering: Meatbags in manglement](https://www.theregister.com/2025/12/24/reason_ai_isnt_delivering/)
