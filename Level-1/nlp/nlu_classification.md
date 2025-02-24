# NLU Intent Classification

## How Intent Classification Works

### Teaching semantics to a machine: word embeddings

**Natural Language Understanding (NLU)** is responsible for mapping the input that is given in natural language to a beneficial representation. 

How can we transform if “apple” is more similar to an “orange” than "hamburger" into a comparison with numbers?

It is possible for a computer to read the whole Internet and learn similarities from billions of words. 

In fact, embedding algorithms are used on huge text corpora such as Wikipedia or Commoncrawl to learn those similarities.

As you can see in the tables below those datasets contains millions of words in many languages and a lot can be learned by crawling them.

The first generation of such algorithms included Word2Vec and Glove.

Training means the algorithm reads the entirety of Wikipedia or Commoncrawl and learns the semantics of words from their context. 

The output of this training is _word embeddings_ where each word is a vector (an array of numbers). 

As you may remember from vectors, we can place vectors in a space and measure the distance between them, since similar words will have vectors close to each other. 

The distance between the vectors of “apple” and “orange” is smaller than between the vectors of “apple” and “hamburger”.


### Training a NLU model

NLU systems become better with more training which is why they need several examples for every intent. 

The examples should be similar in meanings, so if you were to plot all those sentence vectors, they should be close to each other. Thus, the examples should form a cloud of points (one cloud for each intent). 

When you train your NLU model, the model learns the boundaries between all the clouds, so when your system encounters a sentence it has never see it can map it to the closest cloud of points and determine its intent. 

The illustration below shows how such clouds and boundaries might look. 

Each point corresponds to a sentence or more precisely to a sentence’s vector. 

Blue crosses are vectors of sentences with the intent checkbalance and red circles correspond to vectors of sentences with the inten `_transfer`.

Word embeddings are good for comparing words but how do we compare sentences? 

We can perform arithmetic operations on vectors. 

A sentence is a group of words, the meaning of each word is captured in a vector, and averaging those vectors is a way to capture the meaning of a sentence.

To recap, the meaning of a word is captured by its _word embedding_ and the meaning of a sentence is captured by the average of the embedding of its words. 

So now we have something that can enable a machine to say “I am hungry” and “I want to eat” are similar sentences.

Intents are labels tagging sentences that has similar meaning. 

We have seen how meaning can be captured, so we can use that to train our assistant.


### Limitations of word embeddings

Word embeddings are great since they provide some sense of meaning for a wide vocabulary out of the box, but they also come with limitations that newer generations of NLU systems are addressing.

#### Homonyms

One limitation is that a single work can have several meanings. 

A “bank” can be a financial instutition or a follow the river. An embedding of bank would carry an average meaning based on the frequency of the context in which bank is used in the training data.

#### Plurals, abbreviations, and typos

Another limitation is that using whole words as features makes typos and abbreviations harder or impossible to understand: if a word not present in the original corpus (such as Wikipedia) it is unknown.

#### Heavy

Finally, pre-trained word embeddings bring a knowledge of the world inside your model. 

But your model does only need a fraction of this information and might require a more specific knowledge of the world that your chatbot is actually about: if your chatbot is not about a mainstream subject, chances are that the words that are important to your domain are underrepresented. 

Thus, the quality of the embeddings will be affected.


### Training your own embeddings

Training embeddings on your own dataset is a good shot at solving the issues raised above. 

You would have better vectors for your domain specific words and a lighter model that would not contain millions of word embeddings you have no use for.

Rasa's `EmbeddingsIntentClassifier` which is based on Facebook's StarSpace algorithm allows this: instead of using pre-trained embeddings, it learns embeddings directly from from your data and trains a classifier on it.

This dramatically increases the accuracy on domain specific datasets because it learns directly from the words included in your examples. 

However, since it has no pre-existing knowledge of the world (no pre-trained embeddings) it requires susbstantially more examples for each intent to get started.

A benefit from training embeddings on your vocubalary is that it can create features from n-grams and not just from words. N-grams are combination of letters inside the words which makes your model tolerant to small variations such as plurals or typing mistakes.

#### Mixing pre-trained with your own embeddings

The latest Rasa iteration is the DIETClassifier which brings the best of both worlds with the ability to mix pre-trained embeddings with your domain specific embeddings. 

This means you can still benefit from a general knowledge of the world and add the knowledge of your domain. 

General knowledge of the world means that your assistant will know that “beer” and “wine” are drinks and that “yes” and “sure” means affirmation. 

You can now build the knowledge of your domain on top of this general knowledge.

#### Giving context to embeddings with transformers

Another big improvement is that the DIETClassifier enriches embeddings with context from your data thanks to its transformers architecture.

**Transformers** are algorithm components designed to learn from sequences. 

Sentences are sequences in the sense that order matters and that each word is used in the context of the other words. 

Understanding a sentence properly involves understanding how each word relates to others.

The images above illustrate this. Since each word relates strongly to itself, we should not look too closely at that. 

If we look at the “The” token (first row), we see that “dog” is the darkest token (besides The of course) which indicates the “The” and “dog” share context, as do for “red” and “dog”.

Every embedding is changed to reflect how it relates to other words which means that NLU models are now able to understand that the bank you withdraw money from is different from the bank that follows the river.

#### Ordering and negation

Transformers looks at how each word infludence others in a sentence which means that a sentence is not a dumb average of all its word embeddings but a weighted average in which the weights represent how relevant a given word is to a particular intent.

For example, the word “play” may be more relevant in the sentence "I want to play chess" where the intent is play than in "I want to watch a play" where the intent is to watch.

A corollary is that _negation_ is better captured. 

With older approaches simply averaging word vectors, the only difference between "I am hungry" and "I am not hungry" was the value of the not vector which might not be a big enough shift to distinguish two opposite intents. 

If you have any experience with NLU, you know how handling negation has always been difficult.

With transformers, your model has a chance to understand that “not” is strongly related to “great” and weigh it differently. 

When you include many other examples with negations, it becomes a concept your model can learn.


----------



## Discover and Classify in-app Message Intent at Airbnb

Conversational AI is inspiring us to rethink the customer experience on our platform.

### Identifying Message Intent

Behind every message sent is an intent: to work out logistics, clarify details, or connect with the host. 

To improve the existing communication experience, it is vital for the AI to identify the "intent" correctly as a first step. 

However, this is a challenging task because it is difficult to identify the exhaustive set of intents that could exist within millions of messages.

To address this challenge, we set up our solutions in two phases: 

- Phase 1: we used a classic unsupervised approach called Latent Dirichlet Allocation (LDA) to discover potential topics (intents) in the large message corpus. 

- Phase 2: we moved to supervised learning techniques but used the topics derived from Phase 1 as intent labels for each message. 

  Thus, we built a multi-class classification model using a  Convolutional Neural Network (CNN) architecture. 

### Intent Discovery

The first challenge in this problem was to discover existing topics (intents) from the enormous messaging corpus without prior knowledge. 

You might think of using embedding techniques to generate message-level clusters and thus topics. However, a key assumption here is that only one primary topic exists in a message which does not hold for Airbnb data. 

On Airbnb, people tend to set up context before they start to type core messages and it is common to have one message containing several pieces of information that are not quite relevant to each other.

Here is an example:

What we really need is an algorithm that can detect distinct underlying topics and decide which one is the primary one based on probability scores.

Thus, LDA becomes a natural choice for our purposes. 

First, LDA is a probabilistic model which gives a probabilistic composition of topics in a message. 

Second, LDA assumes each word is drawn from a certain word distribution that characterizes a unique topic and each message can contain many different topics (see Figure 2 below for a graphical model representation along with the joint distribution of the variables). 

The word distribution allows human judgement to weigh in when deciding what each topic means.

Figure 2: A graphical model representation of LDA by David Blei et al. along with the joint probabilities of the observed (shaded nodes) and hidden (unshaded nodes) units.

Figure 3 shows a 2D visualization of the generated topics using pyLDAvis. We determined the number of topics (hyperparameter K) in LDA to be the one generating the highest coherence score on the validation set.

Figure 3: A 2D visualization of inter-topic distances calculated based on topic-term distribution and projected via principal component analysis (PCA). The size of the circle is determined by the prevalence of the topic.

Due to time constraints, we did not invest much time in methods such as doc2vec and BERT. 

Although these methods have constraints as mentioned above, they do take the word order into account and could be attractive alternatives for intent discovery purposes. We remain open to these methods and plan to revisit them at a later time.

### Labeling: From Unsupervised to Supervised

Labeling is a critical component of Phase 2 since it builds a key transition from an unsupervised solution to a supervised one. 

Although a sketch of the intent space in Phase 1 has already been detected, we do not have full control of the granularity due to its unsupervised nature. 

- This is problematic if a certain Airbnb product needs to address specific message intents which may not have been detected in Phase 1. 

- It is also hard to evaluate the efficacy of LDA results for each message without a clearly predefined intent label for that message as the ground truth.

Just like intent discovery, the first challenge of labeling is to determine what labels to define. So we need to ensure that the quality of the labels is high. 

Our solution is to perform an iterative process that starts from the topics discovered from LDA but leverages product feedback to generate a final set of labels.

During the labeling process, we found that about 13% of our target messages has multi-intent. 

**Multi-intent** is a situation where people ask questions with two or more different intents in one single message. 

When multi-intent occured, we asked our specialists to assign each specific intent to the corresponding sentences. 

Sentences assigned with one single intent were used as an independent training sample when building the intent classification model. 

We demonstrate how they are handled in real-time serving in the Productionization section (Figure 6).

### Intent Classification with CNN

Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) have been very popular methods for NLP tasks. 

In this work, we focus on CNN due to its implementation simplicity, high accuracy, and fast speed (at both training and inference time). 

Piero Molino et al., 2018 showed that Word CNN performs less than 1% worse than the Char C-RNN on the same dataset and hardware while being about 9 times faster during both training and inference. 

In our case, it takes us 10 minutes on average for the validation error to converge while it takes 60 minutes on average for RNN to converge to the same level. 

This results in a much slower model iteration and development when taking hyperparameter tuning into account.

In terms of model accuracy, Wenpeng Yin, et al.,2017 did a thorough comparison of CNN and RNN on different text classification tasks and found that CNN actually performs better than RNN when the classification is determined by some key phrases rather than comprehending the whole long-range semantics. 

In our case, we usually do not need to have the full context of a guest’s message in order to identify the intent of their question. The intent is mostly determined by the key phrases such as “how many beds do you have?” or “is street parking available?”

After extensive literature review, we decided to adopt Yoon Kim,2014 and Ye Zhang et al.,2016 where a simple one-layer CNN followed by one 1-max pooling layer was proposed. 

Unlike the original work, we designed 4 different filter sizes each with 100 filters.

To prepare for the embedding layer, we pre-trained word embeddings based on large out-of-sample Airbnb messaging corpus. 

We performed careful text preprocessing and found that certain preprocessing steps such as tagging certain information are especially helpful in reducing noise since they normalize information auch as URLs, emails, date, time, phone number, etc. 

Below is an example of the most similar words for the word house generated by word2vec models trained without and with such preprocessing steps:

To be consistent, we used the same preprocessing steps when training word embeddings, offline training for message intent classifier, as well as online inference for real-time messages. 

Our to-be-open-sourced Bighead Library made all these feasible.

The overall accuracy of the Phase-1 and 2 solution is around 70% and outperforms the Phase-1 only solution by a magnitude of 50–100%. 

It also exceeds the accuracy of predicting based on label distribution by a magnitude of `~400%`.

We evaluated classification accuracies class by class, especially when the dataset was imbalanced across different classes.

Figure 5: The normalized confusion matrix for the on-trip model results. 

Table 3: Example categories that are not so well predicted. 

There were two primary root causes for the misclassifications:

1. Human errors in labeling
2. Label ambiguity


----------


## Using the DIET classifier for intent classification

The task of intent classification comes under Natural Language Understanding (NLU) and the output of the intent classifier acts as the final interpretation of the query. 

The classification task is pretty straightforward and it can be addressed using various techniques. 

The biggest bottleneck can often be lack of training data. 

We can use Rasa’s DIET classifier and we can use Rasa’s NLU pipeline separately.

Rasa does not require writing any code, all the preprocessing and implementation is handled in the background. 

All we have to do is curate training samples carefully and experiment with various NLU pipelines to get the desired results. 

In this post, we will show how to prepare an NLU pipeline with the DIET classifier and spin up an NLU server to use it as an API.

### What is Rasa

Rasa is an open-source machine learning framework to automate text and voice-based conversations. 

Using Rasa, one can build contextual assistants capable of having layered conversations with lots of back-and-forths. 

Rasa is divided into two major components:

1. **Rasa NLU:** used to perform NLU tasks like intent classification and entity recognition on the user queries. Its job is to interpret messages.

2. **Rasa Core:** Rasa core is used to design the conversation. It handles the conversation flow, utterances, and actions based on the previous set of user inputs.

Since this post is about using DIET Classifier, a component of Rasa NLU, we are going to completely ignore the Rasa Core here.

### DIET: A Quick Introduction:

**Dual Intent and Entity Transformer(DIET)** is a transformer architecture that can handle both intent classification and entity recognition together. 

The best thing about DIET is its flexibility. 

DIET provides the ability to plug and play various pre-trained embeddings such as BERT, GloVe, ConveRT, etc. 

Based on your data and number of training examples, you can experiment with various SOTA NLU pipelines without even writing a single line of code.

If you look at the top right part of the diagram in the left, the total loss is being calculated from the summation of entity loss, mask loss, and intent loss. 

This is because DIET can be trained for these three NLP tasks simultaneously. 

The mask loss is turned off by default, so it only if you have a very large training dataset so that the model can adapt and become more domain-specific.

If you want to use DIET only for the intent classification then you have the option to turn off the entity recognition and vice versa. 

If you want to use the DIET classifier only for the one task, I would suggest you use it for both tasks because the final loss will be the sum of entity loss and intent loss, there is a probability that one task might influence the performance of the other:

Go to this link and based on your use case decide how you want to train your model. 

For example, we will train the DIET classifier for both entity recognition and intent classification.


----------



## Tutorials

Here are some useful NLU tutorials and examples.  

### Text Classification using Python

Here are some examples of Text Classification/Sentiment Analysis:

[NLP: Classification and Recommendation Project using scikit-learn](https://towardsdatascience.com/nlp-classification-recommendation-project-cae5623ccaae?gi=cb49766e5c29)


[Best Practices for Text Classification with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)

[How to Develop a Deep Learning Bag-of-Words Model for Sentiment Analysis (Text Classification)](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/)

[Deep Convolutional Neural Network for Sentiment Analysis (Text Classification)](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/)

[How to Use Word Embedding Layers for Deep Learning with Keras](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)

[Keras Code Examples](https://keras.io/examples/)


### Sentiment Analysis using Python

[3 Simple Ways to get started on NLP Sentiment Analysis](https://medium.com/geekculture/3-simple-ways-to-get-started-on-nlp-sentiment-analysis-d0d102ef5bf8)

[Twitter Sentiment Analysis using NLTK and Python](https://towardsdatascience.com/twitter-sentiment-analysis-classification-using-nltk-python-fa912578614c)

[Classifying Tweets for Sentiment Analysis: NLP in Python for Beginners](https://medium.com/vickdata/detecting-hate-speech-in-tweets-natural-language-processing-in-python-for-beginners-4e591952223)

[Tweet Classification and Clustering in Python](https://medium.com/swlh/tweets-classification-and-clustering-in-python-b107be1ba7c7)

[Identifying Tweet Sentiment in Python](https://towardsdatascience.com/identifying-tweet-sentiment-in-python-7c37162c186b)


### Text Classification using AutoML

[From raw text to model prediction in under 30 lines of Python using Atom](https://towardsdatascience.com/from-raw-text-to-model-prediction-in-under-30-lines-of-python-32133d853407)

[Powerful Twitter Sentiment Analysis in Under 35 Lines of Code](https://medium.com/thedevproject/powerful-twitter-sentiment-analysis-in-under-35-lines-of-code-a80460db24f6)


[A Gentle Introduction to PyCaret for Machine Learning](https://machinelearningmastery.com/pycaret-for-machine-learning/)

[NLP Text-Classification in Python: PyCaret Approach vs The Traditional Approach](https://towardsdatascience.com/nlp-classification-in-python-pycaret-approach-vs-the-traditional-approach-602d38d29f06)

[Natural Language Processing Tutorial (NLP101) - Level Beginner](http://www.pycaret.org/tutorials/html/NLP101.html)


### Sentiment Analysis using AutoML

[Complete Guide to Perform Classification of Tweets with SpaCy](https://towardsdatascience.com/complete-guide-to-perform-classification-of-tweets-with-spacy-e550ee92ca79)

[Sentiment Analysis of Tweets using BERT](https://thinkingneuron.com/sentiment-analysis-of-tweets-using-bert/)

[Fine-Tuning BERT for Tweet Classification with HuggingFace](https://www.kdnuggets.com/2022/01/finetuning-bert-tweets-classification-ft-hugging-face.html)

[How to use SHAP with PyCaret](https://astrobenhart.medium.com/how-to-use-shap-with-pycaret-dc9a31278621)


## References

[1]: [Discovering and Classifying In-app Message Intent at Airbnb](https://medium.com/airbnb-engineering/discovering-and-classifying-in-app-message-intent-at-airbnb-6a55f5400a0c)

[2]: [Using the DIET classifier for intent classification in dialogue](https://medium.com/the-research-nest/using-the-diet-classifier-for-intent-classification-in-dialogue-489c76e62804)

[3]: [How intent classification works in NLU?](https://botfront.io/blog/how-intent-classification-works-in-nlu/)
