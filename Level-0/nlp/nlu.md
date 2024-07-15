# Natural Language Understanding (NLU)

## Overview

NLU is a subset of NLP and its purpose is to extract the _meaning_ from the text. 

NLU is a subset of NLP and its purpose is to extract the _meaning_ from the text. 

For example, extracting the _sentiment_ from a sentence, the intention of the writer, or the semantics of the sentence. 

There are many applications for these specific techniques such as text paraphrasing, summarization, question answering, etc. 

Examples: Extracting the sentiment from a sentence, the intention of the writer, or the semantics of the sentence. 


## How do NLU services work

In the chatbot world, NLU is usually a _service_ that is called on each message received by a user that tries to understand the intention of the user. 

This will allow the automated agent to know what and how to reply. 

Let us dive into the explanation of how a NLU service works. 

First we need to understand the basic vocabulary:

- **Utterance:** The sentence sent by the user to the chatbot

- **Intent** The final goal of the sentence, that the NLU service will try to understand

- **Entity:** A parameter to specify information for the intent

Each utterance will be mapped to an intent and each intent can have 0, 1, or multiple entities. 

It is possible to create our own NLU service or we can use existing systems such as Microsoft LUIS, Google Dialogflow, IBM Watson NLU, Amazon Lex, Rasa NLU and more.


## Language Modeling

**Language modeling (LM)** involves developing a statistical model for predicting the next word in a sentence or next letter in a word given whatever has come before. 

LM is a precursor task in applications such as speech recognition and machine translation.


----------


## Steps to create an assistant

When creating a conversational agent with a NLU service, you have to follow different steps.


## Language Modeling

**Language modeling (LM)** or Text Generation involves developing a statistical model for predicting the next word in a sentence or next letter in a word given whatever has come before. 

LM involves developing a statistical model for predicting the next word in a sentence or next letter in a word given whatever has come before. 

[How to evaluate Text Generation Models](https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1)


## Text Generation Examples

[Text Generation With LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)

[Text Generation using Recurrent Long Short Term Memory Network](https://www.geeksforgeeks.org/text-generation-using-recurrent-long-short-term-memory-network/)

[Text Generation Using LSTM](https://bansalh944.medium.com/text-generation-using-lstm-b6ced8629b03)

[Text Generation through Bidirectional LSTM model](https://towardsdatascience.com/nlp-text-generation-through-bidirectional-lstm-model-9af29da4e520)


[Text Generation using FNet](https://keras.io/examples/nlp/text_generation_fnet/)

[How to Train Keras Deep Learning Models on AWS EC2 GPUs (step-by-step)](https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/)


----------


## What is intent recognition?

_Intent recognition_ or intent classification is the task of taking a written or spoken input and classifying it based on what the user wants to achieve. 

Intent recognition forms an essential component of chatbots and finds use in sales conversions, customer support, and many other areas.

**Natural Language Understanding (NLU)**: The understanding phase is responsible for mapping the input that is given in natural language to a beneficial representation. 

NLU also analyzes different aspects of the input language that is given to the program.

### How intent recognition works

Intent recognition works through the process of providing examples of text alongside their intents to a machine learning (ML) model.

### Training data

Training data is a representative sample of raw data that is manually labeled or organized in the way you eventually want your model to do automatically. 

Once you have labeled the data yourself, you feed it into your ML model to train a model.

### Creating training data

When it comes to intent recognition, the process of creating training data is the following:

1. Convert the data to text

If your input data is speech (such as audio files) you will need to convert these into text.

2. Choose the intents

For example, if you work in customer support, the intents might include “feature request”, “purchase”, and “account closure”

3. Assign each text input an intent

You will likely need many thousands of labeled data points in the training dataset for the model to become accurate enough to work in production.

The process of manually labeling training data is the most time-consuming and laborious aspect of utilizing ML which why we developed our text intent recognition data program. 
    
You input the raw data and the data program breaks it apart into discrete tasks to produce a coherent labeled output automatically. 
    
### Training your model

After the training data has been assembled we can to train a model that would be accessible via API, or you can create your own model or choose from many open-source models (such as one from Google’s BERT) and begin the training process yourself. 

Once your model performs well on your validation set, you are ready to let it loose in production.

More high quality training data makes for more finely tuned models

### Intent recognition use cases

Intent recognition finds a comfortable home in any situation where there are a large number of requests or questions that are often quite similar (Chatbots, Customer support). 


----------


## Challenges of NLU

All of the current solutions still use many traditional text mining algorithms to get meaning from unstructured text documents which is far from ideal.

To better understand the shortcomings of these algorithms used for natural language understanding (NLU), we take a detailed look at the problems of old school approaches at each step of flexible natural language processing. 

Thus, we get a better understanding of the need for an upper ontology to grasp the true meaning of a text. 

Upper ontologies bring value in offering universal general categories for semantic interoperability. 

They define concepts that are essential for understanding meaning.

When we look at the steps a traditional algorithm follows, we could identify seven different tasks:

1. Identify the language of the given text
2. Tokenization or how to separate words
3. Lemming or sentence breaking
4. Grammatical tagging or the parts-of-speech
5. Name Entity Extraction with chunking
6. Parse the syntax to extract meaning
7. Lexical chaining of sentences with sequences of related words



## References

[What is NLU? How to make chatbots understand what you want?](https://medium.com/empathic-labs/what-is-nlu-how-to-make-chatbots-understand-what-you-want-cecafff7aa7b)

[Challenges of Traditional Text Analytics used in NLP/NLU](https://constkogan.medium.com/challenges-of-traditional-text-analytics-used-in-nlp-nlu-b79904f9f9a9)

[What is intent recognition and how can I use it?](https://medium.com/mysuperai/what-is-intent-recognition-and-how-can-i-use-it-9ceb35055c4f)

