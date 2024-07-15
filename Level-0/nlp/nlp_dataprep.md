# NLP Text Preprocessing

## What is NLP Text Preprocessing

NLP Text preprocessing is a method to clean the text in order to make it ready to feed to models. Noise in the text comes in varied forms like emojis, punctuations, different cases. All these noises are of no use to machines and hence need to clean it.

Here are some text preprocessing steps that we can add or remove depending on the dataset [1]:

1. Remove newlines and tabs
2. Strip HTML Tags
3. Remove Links
4. Remove Whitespaces

### NLP Text Preprocessing Steps

Here are some of the key text preprocessing steps [1]:

1. Remove Accented Characters
2. Case Conversion
3. Reducing repeated characters and punctuations
4. Expand Contractions
5. Remove Special Characters
6. Remove Stopwords
7. Correct Misspelled words
8. Lemmatization / Stemming

### Key Findings

The data cleaning step entirely depends on the type of Dataset, so more steps can be included. 

It is a must to remove extra spaces so as to reduce file size.


## Pre-Processing

Pre-processing is perhaps the most important step to a NLP project which involves cleaning your inputs so your models can ignore the noise and focus on what matters most. 

A strong pre-processing pipeline will improve the performance of all your models.

Below are some common pre-processing steps [3]:

- **Segmentation:** Given a long list of characters, we might separate documents by white space, sentences by periods, and words by spaces. 

  Implementation details will vary based on the dataset.

- **Make Lowercase:** Just make everything lowercase. Capitalization generally does not add value and makes string comparison trickier. 

- **Remove Punctuation:** We may want to remove commas, quotes, and other punctuation that does not add to the meaning.
 
- **Remove Stopwords:** Stopwords are words like ‘she’, ‘the’, and ‘of’ that do not add to the meaning of a text, and can distract from the more relevant keywords.

- **Remove Other:** Depending on your application, you may want to remove certain words that do not add value. 

  For example, if evaluating course reviews, words like ‘professor’ and ‘course’ may not be useful.

- **Stemming/Lemmatization:** Both stemming and lemmatization generate the root form of inflected words (running to run). 

  Stemming is faster but does not guarantee the root is an English word. 
  
  Lemmatization uses a corpus to ensure the root is a word at the expense of speed.

- **Part of Speech Tagging:** POS tagging marks words with their part of speech (nouns, verbs, prepositions) based on definition and context. 

  For example, we can focus on nouns for keyword extraction.
  


## Preprocessing Textual Data with Pandas

The most significant difference between textual data and numeric data is the amount of cleaning and preprocessing they require [2]. In addition, textual data might comes in a variety of formats.

Numbers usually come in a format that can be directly used in analysis or modeling, perhaps with a few modification. 

Textual data is represented as strings in Python. A string is a sequence of unicode characters. Unlike some other programming languages, Python does not have a character data type so a single character is a string of length 1.

### String

Strings are stored with the “object” data type by default. It may cause some drawbacks because non-string data can also be stored with this data type. Thus, with Pandas version 1.0, a new data type for strings was introduced: “StringDtype”.

### Split

A string might include multiple pieces of information. For instance, a typical address shows street, city, and state information. In our mock DataFrame, the group column consists of two parts combined with a hyphen. If we want to represent groups in two separate columns, we can split it.

We can create separate columns by using the expand parameter.

```py
     df["group1"] = df["group"].str.split("-", expand=True)[0]
     df["group2"] = df["group"].str.split("-", expand=True)[1]

     # output
     #      groupq1   group2
     # 0      A         1B
     # 1      B         1B
     # 2      A         1C
     # 3      A         1B
     # 4      C         1C
     # Name: group, dtype: object
```

### Combine

Just like we split strings, we sometimes need to combine them. Let’s create a name column that includes both the first and last names. 

```py
     # method 1
     df["first_name"].str.cat(df["last_name"], sep=" ")

     # method 2
     df["first_name"] + " " + df["last_name"]

     # output of both 1 and 2
     # 0     John Doe
     # 1     jane doe
     # 2    emily uth
     # 3     Matt Dan
     # 4     Alex mir
     # dtype: object
```

## Lowercase

```py
     # lowercase
     df["first_name"].str.lower() + " " + df["last_name"].str.lower()

     # output
     # 0     john doe
     # 1     jane doe
     # 2    emily uth
     # 3     matt dan
     # 4     alex mir
     # dtype: object
```

### Capitalize

```py
     df["first_name"].str.capitalize() + " " + df["last_name"].str.capitalize()

     # output
     # 0     John Doe
     # 1     Jane Doe
     # 2    Emily Uth
     # 3     Matt Dan
     # 4     Alex Mir
     # dtype: object
```

```py
     # insert column at beginning
     name = df["first_name"].str.lower() + " " + df["last_name"].str.lower()
     df.insert(0, "name", name)
```

### Index

We may need to extract numerical data from strings. 

The `salary` column is an example where we need to remove the currency sign and the comma.

We have mentioned that strings are a sequence of characters, so we can use _indexing_ to access characters. 

Since the currency signs are the first characters, we can remove them by selecting the characters starting from the second one.

```py
     df["salary"].str[1:]
     
     # output
     # 0     75000
     # 1     72000
     # 2     45000
     # 3     77000
     # 4    58,000
     # Name: salary, dtype: object
```

In one of the values, a comma is used as a thousand separator. 

We can remove it by using the `replace` method.

```py
     df["salary"].str[1:].str.replace(",","")

     # output
     # 0     75000
     # 1     72000
     # 2     45000
     # 3     77000
     # 4     58000
     # Name: salary, dtype: object
```

The comma has been replaced by an empty string which is equal to removing it. 

Since multiple string operations can be done at once, we can perform all this in a single step.

```py
     df["salary_numeric"] = df["salary"].str[1:].str.replace(",","").astype("int")

     df.dtypes

     # output
     # name              object
     # first_name        object
     # last_name         object
     # group             object
     # salary            object
     # group1            object
     # group2            object
     # salary_numeric     int64
     # dtype: object
```

### Encoding categorical values

Some algorithms do not accept string values, so we need to convert them to numeric values by label encoding or one-hot encoding.

**Label encoding** is just replacing the strings with numbers. 

We can perform label encoding on the “group1” column. 

We can manually replace values with numbers but this can be tedious. Since the number of distinct values is high, this method is not very practical.

A better option is to change the data type of this column to category and then use the category codes.

```py
     df["group1"] = df["group1"].astype("category")
     df["group1_numeric"] = df['group1'].cat.codes
     df[["group1", "group1_numeric"]]
```

Each category has been replaced by a number. 

Unless there is a hierarchy among the categories, label encoding is not applicable for some algorithms. Here, category "C" might be given higher importance.

In this case, we should do **one-hot encoding** which means creating a new column for each distinct value. 

The value in the “group1” column is "A", the value in column "A" is 1, etc.

```py
     pd.get_dummies(["group1"])
```


## References

[1]: [Cleaning and Preprocessing Text Data by Building NLP Pipeline](https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a)

[2]: [If You Work With Textual Data, Learn These Pandas Methods](https://towardsdatascience.com/if-you-work-with-textual-data-learn-these-pandas-methods-3f224122ebaf?gi=21487bb52ed1)

[3]: [Natural Language Processing (NLP): Don’t Reinvent the Wheel](https://towardsdatascience.com/natural-language-processing-nlp-dont-reinvent-the-wheel-8cf3204383dd)

[4]: [Text Augmentation in a few lines of Python Code](https://towardsdatascience.com/text-augmentation-in-few-lines-of-python-code-cdd10cf3cf84)

[5]: [Data Augmentation for Speech Recognition](https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78)

[6]: [How To Perfectly Clean Your Text Data For NLP](https://heartbeat.comet.ml/how-to-perfectly-clean-your-text-data-for-nlp-29cfae79e243)

[7]: [Understanding Regular Expression for Natural Language Processing](https://heartbeat.comet.ml/understanding-regular-expression-for-natural-language-processing-ce9c4e272a29)

[8]: [Two minutes NLP — Python Regular Expressions cheatsheet](https://medium.com/nlplanet/two-minutes-nlp-python-regular-expressions-cheatsheet-d880e95bb468)


[Complete Guideline to Implementation of Basic NLP Techniques with NLTK and spaCy](https://towardsdatascience.com/hands-on-implementation-of-basic-nlp-techniques-nltk-or-spacy-687099e02816)

[The Ultimate Guide To Different Word Embedding Techniques In NLP](https://www.kdnuggets.com/2021/11/guide-word-embedding-techniques-nlp.html)

[Implementing Various NLP Text Representation in Python](https://towardsdatascience.com/implementing-various-nlp-text-representation-in-python-84781da0ec2d)

[Difference between Bag of Words (BOW) and TF-IDF in NLP with Python](https://pub.towardsai.net/difference-between-bag-of-words-bow-and-tf-idf-in-nlp-with-python-97d3e75a9fd)


[Boost Performance of Text Classification tasks with Easy Data Augmentation](https://towardsdatascience.com/boost-performance-of-text-classification-tasks-with-easy-data-augmentation-1420d45b914a)
