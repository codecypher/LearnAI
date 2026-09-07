# Context Engineering

## What is Context Engineering?

Context engineering is the discipline of building dynamic systems that supply an LLM with everything it needs to accomplish a task [1].

It is essential to realize that an LLM prompt is not just a single string of text, but an entire context window of input the model sees before producing output which includes multiple components beyond the immediate user question.

![The many components of context engineering](https://cdn.thenewstack.io/media/2025/07/0ba15209-context-engineering-1024x586.png)

Context engineering involves assembling a variety of components including a basic prompt, memory, output from RAG pipelines, output from tool invocation, well-defined and structured output format, and guardrails.

### Context Engineering vs Prompt Engineering

How is context engineering different from the prompt engineering? [1]

- Context engineering is a superset of prompt engineering.

- The key difference is scope.

- A prompt engineering mindset often leads to trial-and-error tinkering.

- Context engineering emphasizes systematic, repeatable frameworks.

### Context Engineering vs RAG

Retrieval-augmented generation (RAG) combines a search step with LLM to allow authoritative passages from an external knowledge store to be fetched and injected into the prompt just before generation [1].

RAG becomes one of the components that context engineering relies on for grounding the responses in factual data to reduce hallicinations.

A pragmatic guideline is to choose RAG whenever the knowledge base is larger or more dynamic than the context window, and use long context only when a self-contained document fits comfortably inside the limit.

Even while designing a simple RAG stack, we must decide how to embed the retrieved passages: positioning, formatting, and compression are context engineering choices that influence final accuracy.


## Three Levels of Context Engineering

Long-running LLM applications degrade when context is unmanaged. Context engineering turns the context window into a deliberate, optimized resource [2].

Context engineering treats the context window as a managed resource with explicit allocation policies and memory systems.

This article explains context engineering at three levels [2]:

### Level 1: Understanding The Context Bottleneck

LLMs have fixed context windows. Everything the model knows at inference time must fit in those tokens.

![Context Engineering Level 1](https://www.kdnuggets.com/wp-content/uploads/bala-context-engg-level1.png "Context Engineering Level 1")

Context engineering is about designing for continuous curation of the information environment around an LLM throughout its execution.

### Level 2: Optimizing Context In Practice

Effective context engineering requires explicit strategies across several dimensions.

#### Budgeting Tokens

Allocate your context window deliberately.

Conversation history, tool schemas, retrieved documents, and real-time data can all add up quickly.

With a very large context window, there is plenty of headroom.
With a much smaller window, you are forced to make hard tradeoffs about what to keep and what to drop.

#### Truncating Conversations

Some systems implement _semantic compression_ which means extracting key facts rather than preserving verbatim text.

Test where your agent breaks as conversations extend.

#### Managing Tool Outputs

Large API responses consume tokens fast.

Request specific fields instead of full payloads, truncate results, summarize before returning to the model, or use multi-pass strategies where the agent first gets metadata then requests details for relevant items only.

#### Using The Model Context Protocol And On-demand Retrieval

Instead of loading everything upfront, connect the model to external data sources it queries when needed using the model context protocol (MCP).

The agent decides what to fetch based on task requirements.

#### Separating Structured States

Put stable instructions in system messages. Put variable data in user messages where it can be updated or removed without touching core directives. Treat conversation history, tool outputs, and retrieved documents as separate streams with independent management policies.

![Context Engineering Level 2](https://www.kdnuggets.com/wp-content/uploads/bala-context-engg-level2.png)

The shift is to treat context as a dynamic resource that needs active management across an agent's runtime rather than a static thing you configure once.

### Level 3: Implementing Context Engineering In Production

Context engineering at scale requires sophisticated memory architectures, compression strategies, and retrieval systems working in concert.

Here is how to build production-grade implementations [2]:

#### Designing Memory Architecture Patterns

Separate memory in agentic AI systems into tiers:

- Working memory (active context window)
- Episodic memory (compressed conversation history and task state)
- Semantic memory (facts, documents, knowledge base)
- Procedural memory (instructions)

Working memory is what the model sees now, which is to be optimized for immediate task needs.

Episodic memory stores what happened.

You can compress aggressively but preserve temporal relationships and causal chains.

For semantic memory, store indexes by topic, entity, and relevance for fast retrieval.

#### Applying Compression Techniques

Naive summarization loses critical details.

A better approach is extractive compression where you identify and preserve high-information-density sentences while discarding filler.

- For tool outputs, extract structured data (entities, metrics, relationships) rather than prose summaries.
- For conversations, preserve user intents and agent commitments exactly while compressing reasoning chains.

#### Designing Retrieval Systems

When the model needs information not in context, retrieval quality determines success.

Implement hybrid search: dense embeddings for semantic similarity, BM25 for keyword matching, and metadata filters for precision.

- Rank results by recency, relevance, and information density.
- Return top K but also surface near-misses; the model should know what almost matched.
- Retrieval happens in-context, so the model sees query formulation and results.
- Bad queries produce bad results; expose this to enable self-correction.

#### Optimizing At The Token Level

Profile your token usage continuously.

- System instructions consuming 5K tokens that could be 1K? Rewrite them.
- Tool schemas verbose? Use compact JSON schemas instead of full OpenAPI specs.
- Conversation turns repeating similar content? Deduplicate.
- Retrieved documents overlapping? Merge before adding to context.

Every token saved is a token available for task-critical information.

#### Triggering Memory Retrieval

The model should not retrieve constantly; this is expensive and adds latency.

Implement smart triggers: retrieve when the model explicitly requests information, when detecting knowledge gaps, when task switches occur, or when user references past context.

![Context Engineering Level 3](https://www.kdnuggets.com/wp-content/uploads/bala-context-engg-level3.png)

When retrieval returns nothing useful, the model should know this explicitly rather than hallucinating.

Return empty results with metadata: "No documents found matching query X in knowledge base Y."

This lets the model adjust strategy by reformulating the query, searching a different source, or informing the user the information is not available.

#### Synthesizing Multi-document Information

When reasoning requires multiple sources, process hierarchically.

- First pass: extract key facts from each document independently (parallelizable).
- Second pass: load extracted facts into context and synthesize.

This avoids context exhaustion from loading 10 full documents while preserving multi-source reasoning capability.

For contradictory sources, preserve the contradiction.

Let the model see conflicting information and resolve it or flag it for user attention.

#### Persisting Conversation State

For agents that pause and resume, serialize context state to external storage.

- Save compressed conversation history, current task graph, tool outputs, and retrieval cache.
- On resume, reconstruct minimal necessary context; do not reload everything.

#### Evaluating And Measuring Performance

Track key metrics to understand how your context engineering strategy is performing.

Monitor context utilization to see the average percentage of the window being used, and eviction frequency to understand how often you are hitting context limits.

Measure retrieval precision by checking what fraction of retrieved documents are actually relevant and used.

Finally, track information persistence to see how many turns important facts survive before being lost.


## Enhancing RAG Systems with Context Engineering

This article covers a production-ready, next-generation RAG pipeline that demonstrates how modern systems go beyond simple "retrieve and generate" approaches by combining [3]:

- Document ingestion (PDF, DOCX, TXT, CSV)
- Vector-based retrieval with FAISS
- Contextual compression to filter noise and focus on relevant information
- Dual LLM query analysis to adapt responses based on style and complexity
- Vectorstore persistence for saving and loading knowledge bases
- Streamlit-powered chat UI with session state, history, and metadata tracking
- Local experimentation with Ollama, enabling easy swaps between LLMs

![Enhancing RAG Systems with Context Engineering](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*T_XoY9JyHMZDl7vp2-JQwg.png)

The complete code can be found on GitHub: [AgenticRAG-Context-Engineering](https://github.com/vikrambhat2/AgenticRAG-Context-Engineering)

Here is how to build this project step-by-step from environment setup to multi-agent orchestration [3]:

### Step-by-Step Code Walkthrough

Some key libraries include:

langgraph — to build the context engineering workflow as a graph.
langchain-community — to integrate Ollama and external tools.
python-dotenv — for managing environment variables securely.
streamlit — for the interactive interface (if included in your demo).

The project uses Ollama as the local LLM runtime for context-aware tasks (Ollama + LLaMA 3.x Model).

### Key Design Principles Demonstrated

The architecture creates a production-ready RAG chatbot with advanced features such contextual compression, multi-format document support, and a user-friendly interface [3].

- Separation of Concerns: UI logic separate from RAG logic
- State Management: Immutable state transitions in workflow
- Error Handling: Graceful degradation at every level
- User Experience: Clear feedback, loading states, and help text
- Performance: GPU optimization, efficient retrieval, compression
- Extensibility: Modular design allows easy feature additions
- Observability: Comprehensive metadata tracking for debugging

### Summary

By implementing advanced techniques like contextual compression, dual LLM architectures, and state-driven workflows, we have created a system that does not just answer questions — it intelligently processes information.

Here are the key features of the implementation:

- Intelligent Information Processing: The dual-retrieval system first casts a wide net, then uses LLM-powered compression to extract only the most relevant information. This approach dramatically reduces noise while maintaining context, leading to more accurate and focused responses.

- Production-Ready Architecture: From GPU optimization and error handling to session state management and vectorstore persistence, every component is designed for real-world deployment. The modular architecture makes it easy to swap components, add new features, or scale individual parts of the system.

- Developer Experience: The extensive use of type hints, comprehensive error handling, and detailed metadata tracking makes this system maintainable and debuggable. The LangGraph workflow provides clear visibility into each processing stage.

Here are the key concepts for RAG projects:

- Context Quality Over Quantity: More retrieved documents doesn’t always mean better answers. Intelligent compression and filtering can dramatically improve response quality.

- State Management is Critical: Using proper state machines (like LangGraph) makes complex workflows manageable and debuggable.

- User Experience Matters: A powerful backend is only as good as its interface. Streamlit’s reactive patterns, combined with proper session management, create smooth user interactions.

- Plan for Production: Error handling, performance optimization, and observability aren’t afterthoughts — they should be built into the architecture from day one.


## References

[1]: https://joaolealdasilva.medium.com/stop-wrestling-with-your-docker-containers-a-complete-dockhand-mini-course-05df4de8e329 "Context Engineering: Going Beyond Prompt Engineering and RAG"

[2]: https://joaolealdasilva.medium.com/stop-wrestling-with-your-docker-containers-a-complete-dockhand-mini-course-05df4de8e329 "Context Engineering Explained in 3 Levels of Difficulty"

[3]: https://pub.towardsai.net/beyond-rag-context-engineering-for-smarter-ai-systems-96caf8d562f7 "Beyond RAG: Context Engineering for Smarter AI Systems"

[12 Ways to Reduce LLM Latency and Inference Costs in Production](https://www.kdnuggets.com/12-ways-to-reduce-llm-latency-and-inference-costs-in-production)

