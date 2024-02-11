# Project Management

Here are some notes on project management.

Every software engineer and manager needs to read the following:

R. L. Glass, _Facts and Fallacies of Software Engineering_, Addison-Wesley Professional, 1st ed., ISBN: 0321117425, 2002.

F. Brooks, _The Mythical Man-Month_, Anniversary Edition, Addison-Wesley Professional, 1995. 

T. Demarco, _Peopleware: Productive Projects and Teams_, 2nd ed., Dorset House, 1999. 



## Project Quickstart

1. Who is the user?

Once you identify the core user, you can identify their core needs.

If the software does not solve a problem, it does not merit being adopted.

2. How many people will use the application?

Most projects must reach a critical number of users in order to be successful, pay the bills, and get off the ground.

Choosing the feature that will carry the most impact is essential to making sure the time of the entire team goes into the work with the biggest pay-off.

3. Is it the easiest possible solution?



## Avoid project drift

Project drift is when the goals and scope of a project change over time. Sometimes project drift starts immediately at the launch of a project. Or the problem emerges slowly and builds over time.

### Define the deliverable

The best way to avoid project drift is to clearly define the deliverable. 

- Spend as much time as you need to figure out what success looks like to your client. If your client cannot define success, you may not be ready to launch the project. 

- Make a clear connection between the problem they are trying to solve and how your analyses will lead to actionable insights or solutions.

### Be specific about the deliverable in the contract

Be as specific as possible and write everything down. Thus, when you receive additional requests for work that is out of scope, you can bring the conversations back to the written contract rather than trying to recollect the initial agreement.

One way to handle requests is to create a list of the different ideas to ensure your client feels heard. 

You can discuss how these ideas could be part of a supplemental contract while you redirect the conversation to the actual deliverables.

### Create a project timeline

Having a project timeline can help further define the boundaries of your work and help manage client expectations. 

Using a project management tool with the client is an excellent way to ensure transparency in all the work. 

### Structured project meetings

Here is a sample agenda for consulting project meetings:

- Review accomplishments since the what progress the last meeting

- Discuss deliverables for the next meeting

- Define who is doing what

- Identify roadblocks that may interfere with the completion of deliverables

- Review other issues as needed

The key is to ensure that everybody remains aligned with the project goals and deliverables. 

Be cautious about allowing too many discussions about exploring interesting avenues. 

Many things may seem “interesting” and “useful” but will not move you closer to the deliverable.



## Managing Design Feedback in Meetings

### Good feedback starts with good preparation

When feedback is discussed in meetings, it is best to share the work the day before or earlier on the same day if the meeting is in the afternoon.

Asking people for feedback during a meeting puts them on the spot and the best you can hope for is an off-the-cuff comment, which might not even reflect what they think. 

### Don’t take design solutions at face value

When feedback is discussed, the goal is to understand the shortcomings of a design proposal, not to brainstorm solutions. 

It is best to maintain a healthy dose of skepticism, even if a solution sounds reasonable, and take time to analyze it later.

### Resist the urge to respond immediately

Resist the urge to go down the rabbit hole of debating specific solutions during the meeting. It might work fine for very simple issues, but for complex ones, you will need time to think through the proposed solution and make sure it does not conflict with other use cases. 

Take notes, express appreciation for their insight, and explain you need time to consider their idea and work it into the design. Also make sure they are not left with the impression that their suggested solution will be implemented verbatim. 

### Repeat back and align

When people share feedback they want to feel heard more than they want to see their idea being implemented. 

The best way to do this is to rephrase and repeat back what someone said until they confirm that is what they meant. 

### Express gratitude and define actions in public

When it is time to wrap it up, be sure to thank everyone for their input. You want people to keep coming to the meetings and be eager to help. End  on a positive note by acknowledging everyone’s effort and expressing gratitude.

Recap the actions you are going to take. You can even create the actions in a project management system that is visible to everyone (Notion, Asana, etc.).



## Changing Software Requirements

Change is the devastating disruptor of software development. It ruins code, designs to plans. 

Once you change code or design in software, you do not know what might break and you might need to change everything built on top of it.

When the requirements change, here are a few of the software changes required:

- Updating requirements and sign off.
- Code needs to be updated (potentially breaking dependent code). 
- Integrations updated
- Data updated
- Deployment/DevOps updated
- Data migration updated
- Testing updated
- Documentation updated
- Training updated
- A new release through the software lifecycle

The most significant change is extending deadlines and the plan being updated to incorporate all the work needed to make the changes above. When plans change, emotions rise and pressure increases.

A fundamental assumption being clarified can trigger an incredible amount of change and cost.

### Assumptions

Software development processes generate feedback and squash assumptions. 

We assume assumptions are just related to requirements, but there are lots of assumptions in software development [4]:

- We assume the software should work in all scenarios. 

- We have all the requirements. 

- We can build the current code. 

- The current code base does exactly what the requirements specify. 

- We will deliver to the current plan. 

- The software will do what the business needs it to do. 

- Everything will be in place when we deploy to live

- We have the skills to create the software. 

- The software will work with predicated capacity/load in production. 

- The current software meets regulatory standards. 

- If the server crashes, we have a backup which will restore everything (anyone practiced disaster recovery on the project?). 

- If the development server explodes, we have all the code in source control and a playbook to recreate everything. 

- If the development team died on a boat party, another team could pick up based on source control and documentation. 

### Feedback

The best tool to combat assumptions is teamwork, questions, and feedback [4]:

- Nightly builds will build and deploy the code every day

- Unit tests will test the code does what we expect after changing it

- Developers, testers and business users check requirements for missing requirements

- Performance testing shows your software works at expected loads

Developers and testers should ask business users lots of questions to understand the business, find missing requirements, and clarify existing requirements meets their business needs.

We need to clarify every assumption because an assumption with lots of dependencies on it is like pulling a Jenga block from the bottom, it can send the whole tower crashing down.

Feedback is a fundamental tool in software development, agile develoment is based on early feedback. 

Finding problems, assumption and other potential change as soon as possible where its easier and quicker to fix.


----------



## Project Startup Red Flags

There are several red flags with tech startups that you need to be aware of before applying to these types of companies. 

While it could be argued that these red flags apply to all companies, they tend to be unique features of startups that you will noy be aware of until you have worked for one.

1. A lack of credible monetization led by people without a solid track record in the industry

You need to determine how much money the company has, how long it is expected to last, who is running the show, and how the money is expected to be spent. 

2. No obvious market, direction, or customer base

You need to determine whether or not you buy into what the company is selling when you are doing your pre-interview research. 

3. The key product is still in the initial phases of development

The startup may be a sinking ship.

4. A lack of staff

It is vital to broach the topic of staffing early on in the conversation. Having an idea of team member workloads and responsibilities can help you decide whether the team is running short-staffed or conservatively.

5. We are a family

Workplace family culture [6] is a toxic effect that creates an exaggerated sense of loyalty that can become harmful, can create a power dynamic where employees can get taken advantage of, and causes personal and professional lines to become blurred.

What you need to look for in a startup is a team who can work together professionally, values new ideas of how to handle menial data problems, and prioritizes healthy work-life balances.



## Stuart Woodley

[Bad Faith In Software Engineering](https://medium.com/codex/bad-faith-in-software-engineering-d8f413dee61f)

[Cognitive Dissonance Has Become The Default](https://medium.com/codex/cognitive-dissonance-has-become-the-default-115c833c9a69)

[The Constant Grind of The Grand Game](https://medium.com/codex/the-constant-grind-of-the-grand-game-7b66fd04e324)

[How To Motivate A Software Engineer](https://medium.com/codex/how-to-motivate-a-software-engineer-6b9888f19da0)


[Variety Is The Spice Of Development](https://medium.com/codex/variety-is-the-spice-of-development-3aa19b9af0d3)

[The Longer The Interview Process, The Worse The Job](https://medium.com/codex/the-longer-the-interview-process-the-worse-the-job-e4a47ceb0410)


----------


[Constructive Criticism Is Impossible](https://medium.com/codex/constructive-criticism-is-impossible-7ed346a9d7f3)

[Why You Should Never Consent to a Coding Test in an Interview](https://medium.com/swlh/why-you-should-never-consent-to-a-coding-test-in-an-interview-8e22f5078c7f)

[Just Why Is It Called The Agile ‘Manifesto’?](https://medium.com/codex/just-why-is-it-called-the-agile-manifesto-9bd7e349b838)

[How Best To Deal With Irritating “Senior” Managers](https://medium.com/codex/how-best-to-deal-with-irritating-senior-managers-282f6810870e)

[The Futility Of Small Scale Appraisals](https://medium.com/codex/the-futility-of-small-scale-appraisals-390782455907)



## References

[1] [The 3 Questions You Must Ask Before Building a Product](https://medium.com/geekculture/the-3-questions-you-must-ask-before-building-a-product-b2ba104e52f9)

[2] [The Myth of Small Incremental Improvements](https://betterprogramming.pub/the-myth-of-small-incremental-improvements-fd0bfd5e1977)

[3] [5 pro tips on managing design feedback in meetings](https://uxplanet.org/5-pro-tips-on-managing-design-feedback-in-meetings-3a56176da569)

[4] [Assumptions Are Deadly to Development Teams and Software Projects](https://itnext.io/assumptions-are-deadly-to-development-teams-and-software-projects-12b9bc740673)

[5] [5 Data Science Startup Red Flags You Need to Be Aware of Before Applying](https://towardsdatascience.com/5-data-science-startup-red-flags-you-need-to-be-aware-of-before-applying-a06a039aa87d)

[6] [The Toxic Effects of Branding Your Workplace a “Family”](https://hbr.org/2021/10/the-toxic-effects-of-branding-your-workplace-a-family#:~:text=According%20to%20research%2C%20when%20an,attach%20themselves%20to%20the%20organization.)

[7] [3 Questions Experienced Developers Ask to Reveal a Bad Workplace](https://medium.com/codex/3-questions-experienced-developers-ask-to-reveal-a-bad-workplace-34a8d344615a)


