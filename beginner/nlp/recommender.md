# Recommender Systems

## Recommender Systems

In a nutshell, recommender systems predict user interests and recommend relevant items [1].

Recommender systems rely on a combination of data stemming from explicit and implicit information on users and items:

- **Characteristic information:** information about items such as categories, keywords, etc. and users with their preferences and profiles. 

The systems make recommendations based on the user’s item and profile features based on the assumption that if a user was interested in an item in the past, they would be interested in similar items later.

- **User-item interactions:** information about ratings, number of purchases, likes, etc.

Thus, recommender systems fall into two categories: **content-based** systems that use characteristic information and **collaborative filtering** systems based on user-item interactions. 

In addition, there is a complementary method called the **knowledge-based** system that rely on explicit knowledge about the item, the user, and recommendation criteria as well as **hybrid** systems that combine different types of information.


### Content-based algorithms
 
**Cosine similarity:** the algorithm finds the cosine of the angle between the profile vector and item vector:

Based on the cosine value (which ranges between -1 to 1) the items are arranged in descending order and one of the two approaches is used for recommendations:

- Top-n approach: the top n items are recommended. 

- Rating scale approach: all items above a set threshold are recommended.

**Euclidean Distance:** since similar items lie in close proximity to each other if plotted in n-dimensional space, we can calculate the distance between items and use the distance to recommend items to the user:

However, Euclidean Distance performance falls in large-dimensional spaces which limits the scope of its application.

**Pearson’s Correlation:** the algorithm shows how much two items are correlated or similar:

A major drawback of Pearson's algorithm is that it is limited to recommending items that are of the same type.


### Collaborative filtering systems

Unlike content-based systems, collaborative filtering systems utilize user interactions and the preference of other users to filter for items of interest. 

The baseline approach to collaborative filtering is _matrix factorization_. 

The goal is to complete the unknowns in the matrix of user-items interactions (call it RR).

Suppose we have two matrices UU and II such that U x IU d I is equal to RR in the known entries. Using the U x IU x I product we will also have values for the unknown entries of RR which can be used to generate the recommendations.

A smart way to find matrices UU and II is using a neural network. In fact, we can think of this approach as a generalization of classification and regression. 

However, we need to have enough information to work which means a cold start for new e-commerce websites and new users.

There are two types of collaborative models: 

**Memory-based methods** offer two approaches:

- Identify **clusters of users** and utilize the interactions of one specific user to predict the interactions of the cluster. 

- Identifiy **clusters of items** that have been rated by a certain user and use them to predict the interaction of the user with a similar item. 

Memory-based techniques are simple to implement and transparent but they encounter major problems with large sparse matrices since the number of user-item interactions can be too low for generating high-quality clusters.


**Model-based methods** are based on machine learning and data mining techniques to predict user ratings of unrated items. 

These methods are able to recommend a larger number of items to a larger number of users compared to other methods such as memory-based. 

Examples of model-based methods: decision trees, rule-based models, Bayesian methods, and latent factor models.


### Knowledge-based systems

**Knowledge-based** recommender systems use explicit information about the item assortment and the client’s preferences to generate corresponding recommendations. 

If no item satisfies all the requirements, products satisfying a maximal set of constraints are ranked and displayed. 

Unlike other approaches, KB does not depend on large bodies of statistical data about items or user ratings which makes them useful for rarely sold items such as houses or when the user wants to specify requirements manually. 

Such an approach allows avoiding a ramp-up or cold start problem since recommendations do not depend on a base of user ratings. 

Knowledge-based recommender systems have a conversational style offering a dialog that effectively walks the user down a discrimination tree of product features.

Knowledge-based systems work on two approaches: 

- constraint-based: rely on an explicitly defined set of recommendation rules. 

- case-based: uses intelligence from different types of similarity measures and retrieving items similar to the specified requirements.


### Data collection
 
Since a product recommendation engine mainly runs on data, data mining and storage are of primary concern.

The data can be collected explicitly and implicitly:

- Explicit data is information that is provided intentionally which means input from the users such as movie ratings. 

- Implicit data is information that is not provided intentionally but gathered from available data streams such as search history, clicks, order history, etc. 

Data scraping is one of the most useful techniques to extract these types of data from the website.


----------


## Recommender using Scikit-Learn and Tensorflow

The article [6] describes how to recommend sales items to customers by using customer’s individual purchase history using Scikit-Learn and Tensorflow.

Collaborative Filtering for Sales Items sold (binary) per Customer

### Background

The article [6] discusses the dataset and many concepts of recommending engines in detail:

Sparse is better than dense (Zen of Python)

Explicit is better than Implicit (Zen of Python)
.. but implicit is better than nothing. 

In a nutshell: recommendation is based on the assumption that customers who purchase products in a similar quantity share one or more hidden preferences. Due to this shared latent or hidden features, customers will likely purchase similar products.


Binary data vs SalesAmount: Collaborative Filtering in this example is based on binary data (a set of just two values). We add a 1 as purchased which means the customer has purchased this item, no matter how many the customer actually has purchased in the past. 

Another approach would be to use the SalesAmount and normalize it if we want to treat the Amount of SalesItems purchased as a kind of taste factor which means that someone who bought SalesItem x 100 times (while another Customer bought that same SalesItem x only 5 times) does not like it as much. 

### What similarity measure to use

It is recommended to use Pearson if the data is subject to grade-inflation (different customers may be using different scales). 

If our data is dense (almost all attributes have non-zero values) and the magnitude of the attribute values is important, use distance measures such as Euclidean or Manhattan. 

Here we have a sparse univariate data example, so we will use Cosine Similarity.


### Scikit-Learn Recommender

We gave the following: Sales per Sales Day, Customer (such as Customer code 0 is one specific customer), and Sales Item (similar to Customer, Sales Item 0 is one specific Sales Item).

We want to see what Sales Items have been purchased by what Customer:

Our Collaborative Filtering will be based on binary data. 

For every dataset we will add a 1 as purchased which means this customer has purchased this item, no matter how many the customer actually has purchased in the past. 

Another approach would be to use the SalesAmount and normalize it, in case you want to treat the Amount of SalesItems purchased as a kind of taste factor which means that someone who bought SalesItem x 100 times (while another Customer bought that same SalesItem x only 5 times) does not like it as much. 


For better reference, we add "I" as a prefix for Item for each SalesItem. Otherwise, we would only have customer and SalesItem numbers which can be confusing:

We calculate the Item-Item cosine similarity:

Using the top 10 SalesItem recommendations per Customer in a dataframe, we use the Item-Item Similarity Matrix from above cell by creating a SalesItemCustomerMatrixs (SalesItems per rows and Customer as columns filled binary incidence).

We are now finished with recommending Sales Items to Customers which they should be highly interested in (but have not already purchased), using Scikit-Learn.

### Tensorflow Recommender

Now we will use Tensorflow Recommender (tensorflow-recommenders). 

The original code has been taken from Google’s Brain Team and has only been slightly adopted to our dataset:

We add C (for Customer) as a prefix to every customer id:

Define the Customer and SalesItem models:

Finally, we create the model, train it, and generate predictions:


----------


# Building a Recommender System for Implicit Feedback

There are three main algorithms for building a recommender system [4]:

- Content-based uses characteristics of an item to provide a recommendation. 

CB uses the descriptions of the items to build the profile of the user’s preferences.

- Collaborative Filtering uses your behavior or related user attributes to provide a recommendation. 

CF is based on the assumption that people who agreed in the past will agree in the future and that they will like similar kinds of items that they liked in the past.

There are two types of collaborative filtering: user-based and item-based.

Item-based CF is generally preferred over user-based CF since the user-based is a more computationally expensive since we usually have more users than items.

- Hybrid combines collaborative filtering, content-based filtering, and other approaches.

The article [4] covers Collaborative Filtering and how to evaluate the model.

### The Feedback Loop

In the real world, recommendation systems are not easy to develop as they sound such as After watching a movie on Netflix or Amazon Prime,
would you end up rating it? How many times do you rate a movie on Netflix? perhaps one out of twenty times which is where the concept of Explicit and Implicit feedback comes into the picture [4].

The explicit technique relies more on the user’s preference for an item often measured by ratings. Usually, ratings can be captured using a Likert scale of 5 or 10. 

The implicit feedback mechanism is simpler and relies heavily on views, clicks, likes, shares, etc.

### Explicit Feedback vs Implicit Feedback

Most examples use explicit feedback: star rating of movies on IMDB. It is clear what content the user likes (rated 5 stars) and what they do not like (rated 1 star). In this case, we could try to predict the rating the user would give to an unseen movie and recommend those close to 5 stars.

The article [4] focuses on an approach using only implicit feedback such as if a user purchased a product or not which is implicit because we only know that the customer bought the items, but we cannot tell if they liked or which one they preferred. 

More xamples of implicit feedback: number of clicks, number of page visits, the number of times a song was played, etc. 

When dealing with implicit feedback, we can look at the number of occurrences to infer user preference but this can lead to bias towards categories bought on a daily basis.

### The Process

Collaborative Filtering requires [4]:

1. Collection of data about user behavior and associated data items

2. Processing and cleaning the collected data (as per need)

3. Recommendation Calculation Framework

4. Recommended Resultsl

Collaborative Filtering can be both user-based or item-based which means calculate the similarity between users to make an implicit recommendation or calculate the similarity between items to make an implicit recommendation.

### How to measure Similarity

Similarity can be measured using two widely used techniques [4]:

- Euclidean Distance
- Cosine Distance

Cosine Similarity is easier to interpret as theoretically the value can lie between 0 and 1. 

### Evaluation

After building the model, we need to evaluate it to check its quality. 

We could use standard metrics such as MSE for explicit feedback and F1-score for implicit feedback.

However, recommender systems models are different since order matters when giving a list of recommendations.

Therefore, a more suitable evaluation metric is MAP@K which stands for Mean Average Precision @ (cutoff) K.

Using MAP@K to evaluate a recommender algorithm implies that we are treating the recommendation like a ranking task which usually makes sense: we want the most likely/relevant items to be shown first.

A quick fresh-up on the definition of Precision and Recall:

In the context of a recommender system for movies, would be:

- Precision = out of the total movies recommended, # of movies the user did like.

- Recall = out of the total movies the user would like, # of movies that we recommended.

These two are great metrics but they do not care about ordering which is when MAP@K and MAR@K are useful.

### Code

We can build a simple recommender system with just a few lines of code, using Turicreate in Python. 

For evaluation, the ml_metrics package can be used and an easy demo can be found here.

The Instacart Market Basket Analysis dataset from Kaggle is a nice dataset to use to try predicting which products will be in a user’s next order.

----------


## Music Recommender using Pyspark

The article [4] develops a use case for Music recommendations. 

The data is extracted from last.fm API which is a social networking and music recommendation service that uses a variety of data sources. 

Users can ‘scrobble’ (listen to a song) tracks they like on iTunes, Spotify, and other web services and apps. 

Last.fm analyses which music have been scrobbled to each user’s profile and searches for patterns in their listening habits. 

Users can make friends with people they know or meet on the web. 

Last.fm may locate musical ‘neighbors’ who listen to similar music and propose new music based on which artists a user listens to the most (Cornell University).

The idea is to use the data to construct a recommendation system that uses collaborative filtering and social influence data to identify artists that a user might like but has not heard yet. 

### Creating ALS Algorithm

Recommender systems frequently employ collaborative filtering methods which are used to fill in the gaps or missing entries in a user-item association matrix. 

Model-based collaborative filtering is now supported by spark.ml in which people and products (items) are defined by a collection of latent characteristics or features used to forecast missing entries. 

The alternating least squares (ALS) approach is used by spark.ml to learn these latent components. 

### Evaluate the Output

Root Mean Square Error (RMSE) is used to evaluate the performance of the recommender system but RMSE is not a reliable score since the model does not have information about items that are disliked. 

Unfortunately, pyspark does not support any other function to evaluate the performance of the ALS model. 

We can create ROEM (Rank Ordering Error Metric) on the prediction data to evaluate the performance of the model.

----------


## Turning Web Browsing Activity into Product Ratings with Survival Analysis

Suppose an e-commerce website is selling a fairly diverse assortment of products. As the site collects browsing activity, the company would like to exploit it to derive interpretable insights on how users interaction with the web pages affect purchases. 

In addition, the company would like to turn these interactions into ratings through a scoring system which aims to improve both the quality of the recommendations and the user experience.

The article [5] describes how to achieve both objectives by leveraging survival analysis through a practical example.

Despite the complexity of recommender systems and the variety of approaches¹, there are three basic entities:

- Users
- Products
- Ratings

While users and products are straightforward concepts, ratings hide more subtleties since their implementation depends on several factors such as the business domain, context, user experience (UX) and data availability.

### Survival Analysis

Survival analysis covers a collection of statistical methods for describing time to event data that is widely used in clinical studies, but it is also used in other fields (such as predictive maintenance).

Here, we are interested in how the browsing activity affects the time to the event. 

We know that some users made a purchase while other users abandoned the website after a certain amount of time (right censoring):

### Kaplan-Meier estimator

We use the Kaplan-Meier curve to estimate the survival function S(t) which gives the probability that customers will survive ( = not buy) past a certain time after landing on the product page. 

In fact, the unconditional probability of survival up to time t is estimated as the product of conditional probabilities of surviving to the different event times:

We can stratify the Kaplan-Meier curve for a given condition, and thus verify whether the condition affects the survival estimate or not.

Thus, we have identified the positive effect on purchases (or negative effect on the purchase-free survival) represented by adding the product to a wishlist at a glance.

The log-rank statistical test is performed to evaluate differences between the two curves and the p-value allows us to reject the null hypothesis (H₀: the curves are identical) which confirms a statistically significant difference between survival rates in presence vs. absence of wishlist use.

But how can we build a model that takes into account the effect of multiple variables? How can we turn the interactions with web pages elements into measures of hazard or ratings?

### The Cox proportional hazards model

The Cox proportional hazards model³ can be used to assess the association between variables and survival rate, and is defined as:

This, we can use survival analysis to gain interpretable hazard measures that provide insights on the association between customers interaction with the web pages and survival rate.

For a given web browsing session, we may calculate the partial hazards as exp(β’xᵢ), thus neglecting the baseline hazard h₀(t), and therefore estimate the relative risk associated to product purchase:

We can compare it with the hazard from a different web session on the same product page:

We notice that the first session is associated to a higher partial hazard (4.93) than the second (1.17), suggesting that the first user implicitly manifested a higher preference for that product than the second user, and that he may be more likely to buy it.

In fact, we can use the partial hazards as a naive estimate for ratings turning our browsing activity dataset into a triad of user, product, and ratings which are essential to any recommendation strategy:

### Summary

In summary, we applied survival analysis techniques to browsing information collected by a fictional e-commerce website which can provide the following benefits [5]:

- Incorporating of web browsing activity (users behaviour) into existing ratings calculations to improve recommendation results.

- Interpretable hazards can be used to explain what kind of interaction with the UX may lead to an event (purchase).

- Web browsing activity is independent from user identity which can remain anonymous.

- Highlight potential areas of improvement for the existing UI (such as the interaction between customers and a new web page element is associated with a decreased risk of purchase).

However, it is important to remember the following limitations:


----------


## How to Develop Recommender Systems

The article [7] provides a high level overview of recommender systems which may give some ideas for approaches. 

We can also track web browser activity in various ways to build a custom dataset but we should be able to find a toy dataset to work with initially once we decide on an approach.  

Once we decide on an approach (there are many such collaborative filtering for example), we should be able to determine the type of dataset that we need so we can find a toy dataset to use for experimentation and prototyping; it will most likely take a lot of time and effort to build a custom dataset for many of the approaches such as web browser tracking.

Then, we should have a toy dataset that we can use with some AutoML tools to evaluate many different models and be able to narrow the choices to just a few models.


## Recommender Systems using Python

[Learning to Rank for Product Recommendations with XGBRanker](https://towardsdatascience.com/learning-to-rank-for-product-recommendations-a113221ad8a7)

[Machine Learning Streaming with Kafka, Debezium, and BentoML](https://towardsdatascience.com/machine-learning-streaming-with-kafka-debezium-and-bentoml-c5f3996afe8f)

[Recommender System using Collaborative Filtering in Pyspark](https://angeleastbengal.medium.com/recommender-system-using-collaborative-filtering-in-pyspark-b98eab2aea75)



## Performance Metrics for Ranking

The most common performance measures are [10]:

- Normalized Discounted Cumulative Gain (NDCG) 
- Mean Average Precision (MAP)

NDCG is the enhanced version of CG (Cumulative Gain). 

In CG, recommending an order does not matter. If the results contain relevant items in any order, this will result in a higher value and indicate that our predictions are good. 

We need prioritize relevant items when recommending. Therefore, we need to penalize when low relevance items appear earlier in the results which is what DCG achieves. 

However, DCG suffers when different users have a different set of item/interaction counts which is where Normalized Discounted Cumulative Gain (NDCG) comes into play.


We also need to evaluate the coverage of the recommender model using the coverage metric. 

The coverage metric computes the percentage of training products in the test set which means the more coverage better the model. 

In some cases, the model tries to predict popular merchants to maximize NDCG and MAP@k. 

When we have doubts about our evaluation metric, we can quickly check the coverage of the model. Here, we get around 2% coverage which indicates that the model can be improved.



## References

[1] [Recommender Systems — A Complete Guide to Machine Learning Models](https://towardsdatascience.com/recommender-systems-a-complete-guide-to-machine-learning-models-96d3f94ea748)

[2] [Inside recommendations: how a recommender system recommends](https://www.kdnuggets.com/2021/11/recommendations-recommender-system.html)


[3] [Recommend using Scikit-Learn and Tensorflow Recommender](https://towardsdatascience.com/recommend-using-scikit-learn-and-tensorflow-recommender-bc659d91301a)

[4] [Building (and Evaluating) a Recommender System for Implicit Feedback](https://medium.com/@judaikawa/building-and-evaluating-a-recommender-system-for-implicit-feedback-59495d2077d4)


[5] [Recommender System using Collaborative Filtering in Pyspark](https://angeleastbengal.medium.com/recommender-system-using-collaborative-filtering-in-pyspark-b98eab2aea75)

[5] [Turning Web Browsing Activity into Product Ratings with Survival Analysis](https://towardsdatascience.com/turning-web-browsing-activity-into-product-ratings-with-survival-analysis-5d5842af2a6d)

[6] [Recommender Systems: Item-Customer Collaborative Filtering](https://towardsdatascience.com/recommender-systems-item-customer-collaborative-filtering-ff0c8f41ae8a)

[7] [Inside recommendations: how a recommender system recommends](https://www.kdnuggets.com/inside-recommendations-how-a-recommender-system-recommends.html/)

[8] [Learning to rank: A primer](https://towardsdatascience.com/learning-to-rank-a-primer-40d2ff9960af)

[9] [Learning to Rank: A Complete Guide to Ranking using Machine Learning](https://towardsdatascience.com/learning-to-rank-a-complete-guide-to-ranking-using-machine-learning-4c9688d370d4)

[10] [Learning to Rank for Product Recommendations with XGBRanker](https://towardsdatascience.com/learning-to-rank-for-product-recommendations-a113221ad8a7)


[An Intuitive Explanation of Collaborative Filtering](https://www.kdnuggets.com/2022/09/intuitive-explanation-collaborative-filtering.html?utm_source=rss&utm_medium=rss&utm_campaign=an-intuitive-explanation-of-collaborative-filtering)

[Session-Based Recommender Systems with Word2Vec](https://towardsdatascience.com/session-based-recommender-systems-with-word2vec-666afb775509)

