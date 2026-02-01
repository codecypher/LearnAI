# Natural Language Processing (NLP)

## NLP Checklist

There is an NLP checklist given in [1] and project guide in [2].

### NLP Python Libraries

- spacy
- NLTK
- genism
- lexnlp
- Holmes
- Pytorch-Transformers

### Text Preprocessing

Text Preprocessing is the data cleaning process for an NLP application.

When we are dealing with text removing null values and imputing them with mean and median isn’t enough.

- Removing punctuations like . , ! $( ) * % @
- Removing URLs
- Conver to Lowercase
- Converting numbers into words / removing numbers

Some code examples using NLTK are given for the following:

- Removing Stop words
- Tokenization
- Stemming
- Lemmatization
- Part-of-Speech Tagging

### Word Embeddings

Word Embeddings are used to convert the text into a vector of numerical values. The dimensions differ for each model.

Word Embeddings help in faster calculations and reduce storage space.

There are two types of word embeddings: Context-less and Context-driven.

#### Context-Less Word Embeddings

These models are used to convert each word to vector without taking into what situation the text was written.

They focus more on the statistical part of the sentence structure rather than the context.

TF-IDF uses how many times (frequency) a word is used in a sentence and gives priority accordingly.

- Term Frequency-Inverse Document Frequency (TF-IDF)
- Word2Vec
- GloVe

#### Context-Driven Word Embeddings

Context-Driven Word Embeddings holds the context of the sentence together.

These models will be able to differentiate between the nuances of how a single word can be used in different contexts.

This drastically increases the model’s understanding of the data compared to Context-less Word Embeddings.

- Elmo
- OpenAI GPT
- BERT
- RoBERTa
- ALBERT
- ELECTRA
- Distil BERT
- XLNet

Pre-trained word embeddings:

- Word2Vec (Google, 2013), uses Skip Gram and CBOW
- Vectors trained on Google News (1.5GB)
- Stanford Named Entity Recognizer (NER)
- LexPredict: pre-trained word embedding models for legal or regulatory text

### Text Similarity

Text Similarity means to find out how similar sentences or words are (King and Male), (Queen, Female, Mother. These groups are similar to each other this can be found out using text similarity.

Note: The words have to be converted to vectors using any of the models above to find the similarity.

1. Euclidean distance
2. Cosine Similarity

### Named Entity Recognition

NER is the process to find the important labels that are present in the text.

### Deep Learning Models for NLP

- Seq2Seq Models
- RNN
- N-gram Language Models
- LSTM

## NLP Projects

[5 Amazing Ideas For Your Next NLP Project](https://medium.com/pythoneers/5-amazing-ideas-for-your-next-nlp-project-97eb14ebb38)

[Python Chatbot Project](https://data-flair.training/blogs/python-chatbot-project/)

[10 Exciting Project Ideas Using Large Language Models (LLMs) for Your Portfolio](https://towardsdatascience.com/10-exciting-project-ideas-using-large-language-models-llms-for-your-portfolio-970b7ab4cf9e)

## LLM Tools

LangChain is a framework to help developers build LLM applications that combine LLMs with other sources of computation or knowledge.

LlamaIndex is a framework to help developers connect custom data with LLMs by providing the framework for ingesting, structuring, and accessing private or domain-specific data in LLM applications.

ChromaDB is an open-source embeddings database for AI applications that provides efficient storage and retrieval of vector embeddings which is ideal for semantic search and information retrieval systems.

Weaviate is a vector search engine that enables semantic search across multiple data types that can handle large-scale vector operations with rich querying capabilities via GraphQL.

Weights and Biases is an experiment tracking and model monitoring platform.

LangSmith is a model monitoring and evaluation platform for LLM applications.

## Integrating LLMs into Software System

Despite the many advantages of integrating LLMs into existing software, there are many risks that need to be considered [3].

> The tendency to make things up is holding chatbots back. But that’s just what they do. (MIT Technology Review)

- Computational costs of training models and during model inferences due to heavy reliance on high-end GPUs and TPUs

- Making frequent API calls can be expensive, especially for high-traffic applications

- If sensitive data is sent to an LLM, it might be processed, stored, and exposed to another user, especially when the LLM being utilized is accessible to the public

- Aside from properly fine-tuned custom models, most LLMs can only provide open-ended and generic responses and cannot provide domain-based knowledge

- Training LLMs require high energy usage, which can lead to high carbon emissions and environmental pollution

### Tips for LLM Integration

Here are some tips for integrating LLMs into existing systems [4]:

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

## LLM Benchmarks

[MCP-Universe benchmark shows GPT-5 fails more than half of real-world orchestration tasks](https://venturebeat.com/ai/mcp-universe-benchmark-shows-gpt-5-fails-more-than-half-of-real-world-orchestration-tasks)

[Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents](https://arxiv.org/abs/2502.15840)

[CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions](https://arxiv.org/abs/2505.18878)

[Vector Institute aims to clear up confusion about AI model performance](https://www.infoworld.com/article/3959786/vector-institute-aims-to-clear-up-confusion-about-model-ai-performance.html)

## LLM Limitations

[Beyond Context: Unveiling the Limits of Large Language Models’ Performance](https://medium.com/@fabio.matricardi/beyond-context-unveiling-the-limits-of-large-language-models-performance-8b804e03c7e8)

[Machine Bullshit: Why AI Systems Care More About Sounding Good Than Being Right](https://pub.towardsai.net/machine-bullshit-why-ai-systems-care-more-about-sounding-good-than-being-right-052f88dd5d6d)

## References

[1]: [NLP Cheatsheet](https://medium.com/javarevisited/nlp-cheatsheet-2b19ebcc5d2e)

[2]: [LLMs Project Guide: Key Considerations](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/getting-started/llmops-checklist)

[3]: [Integrating Language Models into Existing Software Systems](https://www.kdnuggets.com/integrating-language-models-into-existing-software-systems)

[4]: [LLMs - A Ghost in the Machine](https://zacksiri.dev/posts/llms-a-ghost-in-the-machine/)

----------

[Monitoring unstructured data for LLM and NLP](https://towardsdatascience.com/monitoring-unstructured-data-for-llm-and-nlp-efff42704e5b?source=rss----7f60cf5620c9---4)
