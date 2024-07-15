# Natural Language Processing (NLP)

## Basics

### Terminology

Starting with the smallest unit of data, a **character** is a single letter, number, or punctuation [6]. 

A **word** is a list of characters and a **sentence** is a list of words. 

A **document** is a list of sentences and a **corpus** is a list of documents.

Figure: Venn diagram for NLP [6]. 


### Common Tasks

Some NLP use cases and tasks are [3]:

- **Sentiment Analysis:** to understand the sentiment (negative, positive, neutral) a certain document/text holds.

Example: social media posts about Climate Change.


- **Topic Modeling:** to draw clusters or organize the data based on the topics it contains (the goal is to learn the topics).

Example: an insurance company wants to identify fraudulent claims by classifying causes into main labels and then further analyze the ones containing suspicious content/topics.


- **Text Generation:** to create new data (textual mainly) based on previous examples from the same domain.

Example: chatbots, quotes, email replies, etc.


- **Machine Translation:** to automatically convert from one language to another.

Example: English to German.


### Python Libraries

Natural language toolkit (NLTK) is the defacto standard for building NLP  projects.

There are three leading Python libraries for NLP. These tools will handle most of the heavy lifting during and especially after pre-processing [6]:

- NLTK

The Natural Language Tool Kit (NLTK) is the most widely-used NLP library for Python. Developed at UPenn for academic purposes, NLTK has a plethora of features and corpora. NLTK is great for playing with data and running pre-processing.

- SpaCy

SpaCy is a modern and opinionated package. While NLTK has multiple implementations of each feature, SpaCy keeps only the best performing ones. Spacy supports a wide range of features. 

- GenSim

Unlike NLTK and SpaCy, GenSim specifically tackles the problem of information retrieval (IR). 

Developed with an emphasis on memory management, GenSim contains many models for document similarity, including Latent Semantic Indexing, Word2Vec, and FastText.


### Applications

Now that we have discussed pre-processing methods and Python libraries, we can put it all together with a few examples. 

For each example, we cover a couple of NLP algorithms, pick one based on our rapid development goals, and create a simple implementation using one of the libraries [6]:

- Pre-Processing
- Document Clustering
- Sentiment Analysis

- Document Clustering

The general idea with document clustering is to assign each document a vector representing the topics discussed. 

- Sentiment Analysis: Naive Bayes, gradient boosting, and random forest

- Keyword Extraction: Named Entity Recognition (NER) using SpaCy, Rapid Automatic Keyword Extraction (RAKE) using ntlk-rake

- Text Summarization: TextRank (similar to PageRank) using PyTextRank SpaCy extension, TF-IDF using GenSim

- Text Generation: SimpleNLG surface realization for children with cognitive problems.

- Spell Check: PyEnchant, SymSpell Python ports



## NLP Cheatsheet

This article is a checklist for the exploration needed to develop an NLP model that performs well. 

[NLP Cheatsheet](https://medium.com/javarevisited/nlp-cheatsheet-2b19ebcc5d2e)

[NLTK cheatsheet](https://medium.com/nlplanet/two-minutes-nlp-nltk-cheatsheet-d09c57267a0b)



----------



# Introduction to NLP

**Natural Language Processing (NLP)** is concerned with the analysis and building of intelligent systems that can function in languages that humans speak [2]. 

Processing of language is needed when a system wants to work based on input from a user in the form of text or speech and the user is adding input in regular use English.


**Natural Language Understanding (NLU):** the understanding phase is responsible for mapping the input that is given in natural language to a beneficial representation. 

NLU also analyzes different aspects of the input language that is given to the program.


**Natural Language Generation (NLG):** the generation phase of the processing is used in creating Natural Languages from the first phase. 

Generation starts with Text Planning which is the extraction of relevant content from the base of knowledge. 
  
Next, the Sentence Planning phase chooses the words that will form the sentence. 
  
Finally, the Text Realization phase is the final creation of the sentence structure.

  
### NLU vs NLG

NLP is used to turn sets of unstructured data into formats that computers can convert to speech and text.

Natural Language Understanding (NLU)

- NLU reads and makes sense of natural language. 
- NLU assigns meaning to speech and text. 
- NLU extracts facts from language. 

Natural Language Generation (NLG)

- NLG creates and outputs more language. 
- NLG outputs language with the help of machines. 
- NLG takes the insights that NLU extracts in order to create natural language. 



## Building a Natural Language Processor

There are a total of five execution steps when building a Natural Language Processor [2]:

1. **Lexical Analysis:** Processing of Natural Languages by the NLP algorithm starts with identifying and analyzing the input words’ structure. This part is called Lexical Analysis and Lexicon stands for an anthology of the various words and phrases used in a language. It is dividing a large chunk of words into structural paragraphs and sentences.

2. **Syntactic Analysis/Parsing:** Once the sentence structure is formed, syntactic analysis works on checking the grammar of the formed sentences and phrases. It also forms a relationship among words and eliminates logically incorrect sentences. For instance, the English Language analyzer rejects the sentence, ‘An umbrella opens a man’.

3. **Semantic Analysis:** In the semantic analysis process, the input text is now checked for meaning such as it draws the exact dictionary of all the words present in the sentence and subsequently checks every word and phrase for meaningfulness. This is done by understanding the task at hand and correlating it with the semantic analyzer. For example, a phrase like ‘hot ice’ is rejected.

4. **Discourse Integration:** The discourse integration step forms the story of the sentence. Every sentence should have a relationship with its preceding and succeeding sentences. These relationships are checked by Discourse Integration.

5. **Pragmatic Analysis:** Once all grammatical and syntactic checks are complete, the sentences are now checked for their relevance in the real world. During Pragmatic Analysis, every sentence is revisited and evaluated once again, this time checking them for their applicability in the real world using general knowledge.



## Tokenization, Stemming, and Lemmatization

Here are some tasks used for preprocessing text [2]:

### Tokenization

**Tokenization** or word segmentation is the process that breaks the sequence into smaller units called tokens in order to read and understand the sequence of words within the sentence, 

The tokens can be words, numerals, or even punctuation marks. 

Here is a sample example of how Tokenization works:

```
  Input: Cricket, Baseball and Hockey are primarly hand-based sports.

  Tokenized Output: “Cricket”, “Baseball”, “and”, “Hockey”, “are”, “primarily”, “hand”, “based”, “sports”
```

The start and end of sentences are called **word boundaries** which are used to understand the word boundaries of the given sentence(s).

- **Sent_tokenize package:** This package performs sentence tokenization and converts the input into sentences.

- **Word_tokenize package:** Similar to sentence tokenization, this package divides the input text into words. 

- **WordPunctTokenizer package:** In addition to the word tokenization, this package also works on punctuation marks as a token. 

```py 
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize
    from nltk.tokenize import WordPuncttokenizer
```

### Stemming

When studying the languages that humans use in conversations, some variations occur due to grammatical reasons.

For example, words such as virtual, virtuality, and virtualization all basically mean the same in but can have different meaning in varied sentences. 

For NLTK algorithms to work correctly, they must understand these variations. 

**Stemming** is a heuristic process that understands the word’s root form and helps in analyzing its meanings.

- **PorterStemmer package:** This package is built into Python and uses Porter’s algorithm to compute stems. Basically, the process is to take an input word of "running" and produce a stemmed word "run" as the output of the algorithm. 

- **LancasterStemmer package:** The functionality of the Lancaster stemmer is similar to Porter’s algorithm but has a lower level of strictness. It only removes the verb portion of the word from its source. 
  
  For example, the word ‘writing’ after running through the Lancaster algorithm returns ‘writ’. 

- **SnowballStemmer package:** This also works the same way as the other two and can be imported using the command . These algorithms have interchangeable use cases although they vary in strictness.

```py
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem.snowball import SnowballStemmer
```

### Lemmatization

Adding a morphological detail to words helps in extracting their base forms which is performed using lemmatization. 

Both vocabulary and morphological analysis result in lemmatization. 

This procedure aims to remove inflectional endings. The attained base form is called a lemma.

- **WordNetLemmatizer:** The wordnet function extracts the base form of a word depending on whether the word is being used as a noun or pronoun. 

```py
    from nltk.stem import WordNetLemmatizer
```


## Data Chunking

**Chunking** is the process of dividing data into chunks which is important in NLP. 

The primary function of chunking is to classify different parts of speech and short word phrases such as noun phrases. 

After tokenization is complete and input is divided into tokens, chunking labels them for the algorithm to better understand them. 

Two methodologies are used for chunking and we will be reading about those below [2]:

- **Chunking Up:** Going up or chunking upwards is zooming out on the problem. In the process of chunking up, the sentences become abstract and individual words and phrases of the input are generalized. 

  For example, the question "What is the purpose of a bus?"" after chunking up will answer "Transport"

- **Chunking Down:** The opposite of chunking up. During downward chunking, we move deeper into the language and objects become more specific. 

  For example, "What is a car?"" will yield specific details such as color, shape, brand, size, etc. of the car post being chunked down.

Example: Building Virtual Display Engines on Google Colab

**Noun-Phrase Chunking:** In the code below, we will perform Noun-Phrase (NP) chunking where we search for chunks corresponding to individual noun phrases. 

To create an NP-chunker, we will define a chunk grammar rule (shown in the code below). 

The flow of the algorithm will be as follows:


## Topic Modeling and Identifying Patterns in Data

Documents and discussions are usually revolve around topics. 

The base of every conversation is one topic and discussions revolve around it. 

For NLP to understand and work on human conversations, it needs to derive the topic of discussion within the given input. 

To compute the topic, algorithms run pattern matching theories on the input to determine the topic which is called **topic modeling**. 

Topic modeling is used to uncover the hidden topics/core of documents that need processing.

Topic modeling is used in the following scenarios [2]:

- **Text Classification:** This can improve the classification of textual data since modeling groups similar words, nouns, and actions together which does not use individual words as singular features.

- **Recommender Systems:** Systems based on recommendations rely on building a base of similar content. Therefore, topic modeling algorithms can best utilize recommender systems by computing similarity matrices from the given data.


## Challenges in Natural Language Processing

Here are some challenges that are often encountered with NLP [2]:

- **Lexical Ambiguity:** The first level of ambiguity that occurs generally in words alone. For example, when a code is given a word like "board" it would not know whether it is a noun or a verb which causes ambiguity in the processing of this piece of code.

- **Syntax Level Ambiguity:** This type of ambiguity involves the way phrases sound in comparison to how the machine perceives it. For example, a sentence such as "He raised the scuttle with a blue cap" could mean one of two things: Either he raised a scuttle with the help of a blue cap, or he raised a scuttle that had a red cap.

- **Referential Ambiguity:** This involves references made using pronouns. For instance, two girls are running on the track. Suddenly, she says, ‘I am exhausted’. It is not possible for the program to interpret, who out of the two girls is tired.


----------



# Conmon NLP Techniques

The article [4] discusses six fundamental techniques of NLP:


Here are some common NLP projects:

- NLP project to perform Information Extraction including Named Entity Recognition (NER) using simple regex named entity chunkers and taggers using Python and NLTK. 

- NLP project to categorize and tag words (N-Gram Tagging) and perform feature extraction using Python. 

- NLP project to create an embedding from one of the texts in the Gutenberg corpus and compute some statistics related to the embedding using the Gensim library.


## Lemmatization and Stemming

Stemming and lemmatization are probably the first two steps to build an NLP project — we often use one of the two. 

- Stemming: Stemming is a collection of algorithms that work by clipping off the end of the beginning of the word to reach its infinitive form. 

  These algorithms find the common prefixes and suffixes of the language being analyzed. 

Clipping off the words can lead to the correct infinitive form, but that is not always the case. 

There are many algorithms to perform stemming; the most common one used in English is the Porter stemmer which contains 5 phases that work sequentially to obtain the word’s root.

- Lemmatization: To overcome the flaws of stemming, lemmatization algorithms were designed. 

In these types of algorithms, some linguistic and grammar knowledge needs to be fed to the algorithm to make better decisions when extracting a word’s infinitive form. 

For lemmatization algorithms to perform accurately, they need to extract the correct lemma of each word. Thus, they often require a _dictionary_ of the language to be able to categorize each word correctly.


## Keyword extraction

Keyword extraction (keyword detection or keyword analysis) is an NLP technique used for text analysis. 

The main purpose of keyword extraction (KE) is to automatically extract the most frequent words and expressions from the body of a text. 

KE is often used as a first step to summarize the main ideas of a text and to deliver the key ideas presented in the text.


## Named Entity Recognition (NER)

Similar to stemming and lemmatization, named entity recognition (NER) is a technique used to extract entities from a body of text to identify basic concepts within the text such as names, places, dates, etc.

The NER algorithm mainly has two steps. 

  1. It needs to detect an entity in the text. 
  2. It categorizes the text into one category. 

The performance of NER depends heavily on the training data used to develop the model. The more relevant the training data to the actual data, the more accurate the results will be.


## Topic Modeling

We can use keyword extraction techniques to narrow down a large body of text to a handful of main keywords and ideas. Then, we can extract the main topic of the text.

Another, more advanced technique to identify the topic of text is topic modeling which is built upon unsupervised machine learning that does not require labeled data for training.


## Sentiment Analysis

The most famous and most commonly used NLP technique is sentiment analysis (SA). 

The core function of SA is to extract the sentiment behind a body of text by analyzing the containing words.

The most simple results of the the technique lay on a trinary scale: negative, positive, and neutral. 

The SA algorithm can be more complex and advanced; however, the results will be numeric in this case. 

If the result is a negative number, the sentiment behind the text has a negative tone to it, and if it is positive then some positivity is present in the text.


## Summarization

One of the useful and promising applications of NLP is text summarization which is reducing a large body of text into a smaller chunk containing the main message of the text. 

This technique is often used in long news articles and to summarize research papers.

Text summarization is an advanced technique that uses other techniques that we just mentioned to establish its goals such as topic modeling and keyword extraction. 

Summarization is accomplished in two steps: extract and abstract.



----------



## Guide to NLP

The article [5] summarizes some of the most frequently used algorithms in NLP:

### Bag of Words

The **bag-of-words** (BoW) is a commonly used model that allows you to count all words in a piece of text.

The unigram model is also called the **bag-of-words** model.

BoW creates an occurrence matrix containing word frequencies or occurrences for the sentence or document (disregarding grammar and word order) which are used as features for training a classifier.

BoW has several downsides such as the absence of semantic meaning and context and the fact that stop words (like “the” or “a”) add noise to the analysis and some words are not weighted accordingly (“universe” weights less than the word “they”).

However, there are techniques to overcome these issues.


**Bag of Words:** Converting words to numbers with no semantic information. 

BoW is simply an unordered collection of words and their frequencies (counts) where the tokens (words) have to be 2 or more characters in length.


fastText: Enriching Word Vectors with Subword Information [9]. 

fastText: Using Subword-Based Bag-of-Words Outperforms CBOW in Word2Vec

This is a limitation, especially for languages with large vocabularies and many rare words.

By considering subword units, and words are represented by a sum of its character n-grams. Models are trained on large corpora quickly, allows us to compute word representations for words that did not appear in the training data.


### TF-IDF

**TF-IDF:** Converting the words to numbers or vectors with some weighted information.

In Term Frequency-Inverse Document Frequency (TF-IDF), some semantic information is collected by giving importance to uncommon words than common words.

Instead of giving more weight to words that occur more frequently, TF-IDF gives a higher weight to words that occur less frequently (across the entire corpus). 

When have more domain-specific language in your text, this model performs better by giving weight to these less frequently occurring words. 


### Tokenization

**Tokenization** the process of segmenting running text into sentences and words. 

In essence, tokenization is the task of cutting a text into pieces called tokens and at the same time throwing away certain characters such as punctuation. 

Tokenization can remove punctuation too, easing the path to a proper word segmentation but also triggering possible complications. In the case of periods that follow abbreviation (e.g. dr.), the period following that abbreviation should be considered as part of the same token and not be removed.

The tokenization process can be particularly problematic when dealing with biomedical text domains which contain lots of hyphens, parentheses, and other punctuation marks.

### Stop Words Removal

Stop words removal includes getting rid of common language articles, pronouns, and prepositions such as “and”, “the” or “to” in English. 

In the process, some very common words that appear to provide little or no value to the NLP objective are filtered and excluded from the text to be processed, removing widespread and frequent terms that are not informative about the corresponding text.

Stop words can be safely ignored by carrying out a lookup in a pre-defined list of keywords, freeing up database space and improving processing time.

**There is no universal list of stop words.** 

A stop words list can be pre-selected or built from scratch. A potential approach is to begin by adopting pre-defined stop words and add words to the list later on. Nevertheless, it seems that the general trend has been to move from the use of large standard stop word lists to the use of no lists at all.

The problem is that stop words removal can remove relevant information and modify the context in a given sentence.

For example, if we are performing a sentiment analysis we might throw our algorithm off track if we remove a stop word like “not”. Under these conditions, you might select a minimal stop word list and add additional terms depending on your specific objective.


### Stemming

Stemming refere to the process of slicing the end or the beginning of words with the intention of removing _affixes_ (lexical additions to the root of the word).

The problem is that affixes can create or expand new forms of the same word called _inflectional affixes_ or even create new words themselves called _derivational affixes_. 

In English, prefixes are always derivational (the affix creates a new word as in the example of the prefix “eco” in the word “ecosystem”), but suffixes can be derivational (the affix creates a new word as in the example of the suffix “ist” in the word “guitarist”) or inflectional (the affix creates a new form of word as in the example of the suffix “er” in the word “faster”).

So if stemming has serious limitations, why do we use it? 

- Stemming can be used to correct spelling errors from the tokens. 
- Stemmers are simple to use and run very fast (they perform simple operations on a string).


### Lemmatization

The objective of **Lemmatization** is to reduce a word to its base form and group together different forms of the same word. 

For example, verbs in past tense are changed into present tense (e.g. “went” is changed to “go”) and synonyms are unified (such as “best” is changed to “good”). Thus, standardizing words with similar meaning to their root.

Although it seems closely related to the stemming process, lemmatization uses a different approach to reach the root forms of words.

Lemmatization resolves words to their dictionary form (known as lemma) which requires detailed dictionaries that the algorithm can use to link words to their corresponding lemmas.

Lemmatization also takes into consideration the context of the word to solve other problems such as disambiguation which means it can discriminate between identical words that have different meanings depending on the specific context. 

For example, think about words like “bat” (which can correspond to the animal or to the metal/wooden club used in baseball) or “bank” (corresponding to the financial institution or to the land alongside a body of water). 

By providing a part-of-speech parameter to a word (noun, verb, etc.) it is possible to define a role for that word in the sentence and remove disambiguation.

Thus, lemmatization is a much more resource-intensive task than performing a stemming process. At the same time, since it requires more knowledge about the language structure than a stemming approach, it demands more computational power than setting up or adapting a stemming algorithm.


### Topic Modeling

**Topic Modeling** (TM) is a method for discovering hidden structures in sets of texts or documents. 

In essence, TM clusters text to discover latent topics based on their contents, processing individual words and assigning them values based on their distribution. 

TM is based on the assumptions that each document consists of a mixture of topics and that each topic consists of a set of words which means that if we can spot these hidden topics we can unlock the meaning of our texts.

From the universe of topic modelling techniques, _Latent Dirichlet Allocation (LDA)_ is perhaps the most commonly used. 

LDA is a relatively new algorithm (invented less than 20 years ago) that works as an unsupervised learning method that discovers different topics underlying a collection of documents.

In unsupervised learning methods, there is no output variable to guide the learning process and data is explored by algorithms to find patterns. Thus, LDA finds groups of related words by:

Unlike other clustering algorithms such as K-means that perform hard clustering (where topics are disjointed), LDA assigns each document to a mixture of topics which means that each document can be described by one or more topics (say Document 1 is described by 70% of topic A, 20% of topic B, and 10% of topic C) and reflect more realistic results.

Topic modeling is extremely useful for classifying texts, building recommender systems (recommend based on your past readings) or even detecting trends in online publications.

In recent years, the field has come to be dominated by [deep learning](../dl/dl.md) approaches (see Collobert et al.).



## Data Mining

Data mining is the process of analyzing data by searching for patterns to turn the data into information and better decisions. 

Data mining is algorithm-based and finds patterns in large collections of data. 

Data mining is also important because it presents a potentially more efficient and thorough way of interpreting data.


## Pattern Recognition

Pattern recognition is a branch of ML that is focused on categorizing information or finding anomalies in data. For example, facial pattern recognition might be used to determine the age and gender of a person in a photo. 

Pattern recognition tends to be based on probability, so there is a chance that it does not accurately categorize information. 

Pattern recognition is also typically controlled by an algorithm which means that the computer will continue to make guesses until it finds a pattern that matches what we know is true or until the probability of any other pattern remaining is too small to be considered.



## Challenges in Natural Language Processing

- **Lexical Ambiguity:** This is the first level of ambiguity that occurs generally in words alone. For instance, when a code is given a word like ‘board’ it would not know whether to take it as a noun or a verb. This causes ambiguity in the processing of this piece of code.

- **Syntax Level Ambiguity:** This is another type of ambiguity that has more to do with the way phrases sound in comparison to how the machine perceives it. For instance, a sentence like, ‘He raised the scuttle with a blue cap’. This could mean one of two things. Either he raised a scuttle with the help of a blue cap, or he raised a scuttle that had a red cap.

- **Referential Ambiguity:** References made using pronouns constitute referential ambiguity. For instance, two girls are running on the track. Suddenly, she says, ‘I am exhausted’. It is not possible for the program to interpret, who out of the two girls is tired.


## Common Tasks

Some NLP use cases and tasks are:

- **Sentiment Analysis:** to understand the sentiment (negative, positive, neutral) a certain document/text holds.

Example: social media posts about Climate Change.

- **Topic Modeling:** to draw clusters or organize the data based on the topics (goal is to learn these topics) it contains.

Example: an insurance company wants to identify fraudulent claims by classifying causes into main labels and then further analyze the ones containing suspicious content/topics.

- **Text Generation:** to create new data (textual mainly) based on previous examples from the same domain. 

Example: chatbots, quotes, weather reports, email replies, …, etc.

- **Machine Translation:** to automatically convert from one language to another.

Example: English to German.



## NLP Workflow

In Data Science (and NLP) there is a workflow or pipeline that can be described as follows [3]:

1. **Define the question** that you want to answer out of your data.

Usually this question is given to you as the problem but sometimes it is your job to articulate it. 

2. Get and collect the data. 

If your problem is in the domain of movie reviews your data would be viewers posted reviews along with the ratings. ￼

It is critical that your data is in the same domain as your question/problem and comprehensive, most of times the data is provided or at least the resources that you should be looking at to obtain it.

3. Clean the data. ￼

Almost 90% of the time the data you have is raw, unclean, contains missing fields/outliers/misspellings and so on. 

4. Perform Exploratory Analysis of the Data (EDA). 

EDA is one of the most important steps in any Data Science or NLP task. 

After you have brought your data into a clean ready-to-use state, you want to explore it such that you understand more of its nature and content. 

Your analysis should keep the problem’s question in mind and your job is to try to connect the dots as this step might yield in finding useful correlations/outliers and trends in your data. 

5. Run the NLP technique which best suits the problem. 

This means deciding whether your problem requires sentiment analysis or topic modelling or any other advanced technique which deals with textual data. 

With some practice and experience you would be able to quickly identify the best NLP approach to solve a certain problem. Keep in mind that you can also perform multiple techniques on a single problem in order to be able to draw conclusions and to obtain insights that will answer the main question in step 1. ￼

Deciding on an approach/technique usually means choosing the suitable model or library/package to perform the task. 

6. Obtain knowledge and insights. 

In this step, you need to make use of your communication and representation skills as a data scientist. 


----------



## Tutorials

Here are some useful NLP tutorials and examples.  

### HuggingFace Transformers for NLP With Python

This article [10] explores the use of a simple pre-trained HuggingFace transformer language model for some common NLP tasks in Python.

- Text Classification
- Named Entity Recognition
- Text Summarization


### Haystack

Haystack is an open-source NLP framework that leverages Transformer models, designed to be the bridge between research and industry on neural search, question answering, semantic document search, and summarization. 

Haystack is a modular framework that integrates with other open-source projects such as Huggingface (Transformers, Elasticsearch, or Milvus).

**Haystack Use Cases**

The main use cases of Haystack are [10]:

- **Question Answering:** ask questions in natural language and find granular answers in your documents.

- **Semantic Search:** retrieve documents according to the meaning of the query, not its keywords.

- **Summarization:** ask a generic question and get summaries of the most relevant documents retrieved.

- **Question Generation:** take a document as input and return generated questions that the document can answer.

For example, Question Answering and Semantic Search can be used to better handle the long tail of queries that chatbots receive, or to automate processes by automatically applying a list of questions to new documents and using the extracted answers.

Haystack can also [10]:

- Use pre-trained models (such as BERT, RoBERTa, MiniLM) or fine-tune them to specific domains.

- Collect user feedback to evaluate, benchmark, and continuously improve the models.

- Scale to millions of docs via retrievers, production-ready backends such as Elasticsearch or FAISS, and a fastAPI REST API.

**How it works**

Haystack works by leveraging **Retriever-Reader** pipelines which harnesses the reading comprehension power of the Reader and applies it to large document bases with the help of the Retriever.

- **Readers** are Closed-Domain Question Answering systems: powerful models that analyze documents and perform the question answering task on them. 

Readers are based on the latest transformer-based language models which benefit from GPU acceleration. However, it is not efficient to use the Reader directly on a large collection of documents.

- **Retriever** helps the Reader by acting as a filter that reduces the number of documents that the Reader has to process. 

Retriever achieves this by scanning through all documents in the database, identifying the relevant ones (usually a small subset), and passing them to the Reader.

Here is a figure that summarizes the Retriever-Reader pipeline [4].

----------


## NLP Pretrained Models

- HuggingFace


## NLP Libraries

Here are some useful NLP libraries [12]: [13]:

- NLTK
- GenSim
- Polyglot
- SpaCy
- Textblob

- Pattern
- clean-text

- Presidio
- PySBD
- SymSpell
- TextAttack

The article [7] covers 5 useful Python recipes for your next NLP projects:

- Check metrics of text data with textstat
- Misspelling corrector with pyspellchecker
- Next word prediction with next_word_prediction
- Create an effective Word Cloud
- Semantic similarity analysis

Semantic similarity analysis

_semantic similarity_ measures the likeness of documents/sentences/phrases based on their meaning whereas 

_Lexical similarity_ is a measure of the degree to which the word sets or vocabulary of two given languages are similar. 

_Semantic similarity_ is a metric defined over a set of documents or terms where the idea of distance between items is based on the likeness of their meaning or semantic content. 

The most effective methodology is to use a powerful transformer to encode sentences, get their embeddings and then use cosine similarity to calculate their distance/similarity score.

Calculating the cosine distance between two embeddings gives us the similarity score which is widely used in information retrieval and text summarization such as extract top N most similar sentences from multiple documents. 

The similarity scores can also be used to reduce the dimensionality and to find similar resources.



## References

[1]: S. Bird, E. Klein, and E. Loper. Natural Language Processing with Python – Analyzing Text with the Natural Language Toolkit. Available online: https://www.nltk.org/book/.

[2]: [A Detailed, Novice Introduction to Natural Language Processing (NLP)](https://towardsdatascience.com/a-detailed-novice-introduction-to-natural-language-processing-nlp-90b7be1b7e54)

[3]: [Natural Language Processing (NLP) in Python — Simplified](https://medium.com/@bedourabed/natural-language-processing-nlp-in-python-simplified-b96b89c8be93)

[4]: [NLP Techniques Every Data Scientist Should Know](https://towardsdatascience.com/6-nlp-techniques-every-data-scientist-should-know-7cdea012e5c3)

[5]: [Guide to Natural Language Processing (NLP)](https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1)

[6]: [Natural Language Processing (NLP): Don’t Reinvent the Wheel](https://towardsdatascience.com/natural-language-processing-nlp-dont-reinvent-the-wheel-8cf3204383dd)

[7]: [Solve Complex NLP Tasks with 5 Lesser-known Python Libraries](https://towardsdatascience.com/solving-complex-nlp-tasks-with-5-simple-python-snippets-libraries-7d4dfab131e8)


[8]: [Exploring HuggingFace Transformers For NLP With Python](https://medium.com/geekculture/exploring-huggingface-transformers-for-nlp-with-python-5ae683289e67)

[9]: [fastText: Enriching Word Vectors with Subword Information](https://sh-tsang.medium.com/review-fasttext-enriching-word-vectors-with-subword-information-bafac50f22a8)

[10]: [Two minutes NLP — Quick Introduction to Haystack](https://medium.com/nlplanet/two-minutes-nlp-quick-introduction-to-haystack-da86d0402998)


[11]: [Get To Know Audio Feature Extraction in Python](https://towardsdatascience.com/get-to-know-audio-feature-extraction-in-python-a499fdaefe42)


[12]: [7 Amazing Python Libraries For Natural Language Processing](https://towardsdatascience.com/7-amazing-python-libraries-for-natural-language-processing-50ca6f9f5f11)

[13]: [4 More Little-Known NLP Libraries That Are Hidden Gems](https://towardsdatascience.com/4-more-little-known-nlp-libraries-that-are-hidden-gems-e77a71d1bc57)


[Everything You Need to Know to Get Started with NLP](https://towardsdatascience.com/nlp-survey-bde8a27e8ba)

[NLP Cheatsheet](https://medium.com/javarevisited/nlp-cheatsheet-2b19ebcc5d2e)

[Two minutes NLP — Python Regex Cheatsheet](https://medium.com/nlplanet/two-minutes-nlp-python-regular-expressions-cheatsheet-d880e95bb468)

[Two minutes NLP — NLTK cheatsheet](https://medium.com/nlplanet/two-minutes-nlp-nltk-cheatsheet-d09c57267a0b)

[How to tokenize text and pad sequences in Tensorflow](https://towardsdatascience.com/how-to-tokenize-and-pad-sequences-in-tensorflow-fcbbf2e8b3b5)


[Top 7 Applications of NLP (Natural Language Processing)](https://www.geeksforgeeks.org/top-7-applications-of-natural-language-processing/)

[3 Simple Ways to get started on NLP Sentiment Analysis](https://medium.com/geekculture/3-simple-ways-to-get-started-on-nlp-sentiment-analysis-d0d102ef5bf8)


[The Current Conversational AI and Chatbot Landscape](https://cobusgreyling.medium.com/the-current-conversational-ai-chatbot-landscape-c147e9bcc01b)


[Deep Learning for Natural Language Processing (NLP)](https://machinelearningmastery.com/start-here/#nlp)

[Collobert et. al: Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398)
