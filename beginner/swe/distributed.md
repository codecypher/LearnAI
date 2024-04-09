# Distributed Systems

## Kubernetes

**Kubernetes (K8S)** is an open-source container orchestration system for automating software deployment, scaling, and management. 

Kubernetes works with Docker, Containerd, and CRI-O.

Kubernetes defines a set of building blocks or primitives that collectively provide mechanisms that deploy, maintain, and scale applications based on CPU, memory, or custom metrics.

Kubernetes is loosely coupled and extensible to meet different workloads. 

The internal components as well as extensions and containers that run on Kubernetes rely on the Kubernetes API.

The platform exerts its control over compute and storage resources by defining resources as **Objects** that can be managed.

Kubernetes follows the **primary/replica architecture**. 

The components of Kubernetes can be divided into those that manage an individual node and those that are part of the control plane.


## Kubernetes vs Docker

Kubernetes is an open-source platform used to maintain and deploy a _group of containers_. 

Kubernetes is a container management system developed in the Google platform that helps manage a containerized application in various types of physical, virtual, and cloud environments.


Docker is a tool used to automate the deployment of applications as lightweight, portable containers that can be deployed to various environments.


## Hadoop

Hadoop allows the user to utilise a network of many computers with the aim of harnessing their combined computational power to tackle problems involving huge amounts of data. 

The distributed storage uses the Hadoop Distributed File System (HDFS) while the processing implements the MapReduce programming model using Yet Another Resource Negotiator (YARN) to schedule tasks and allocate resources.

The design of HDFS is with batch processing in mind, rather than interactive use by the user. 

The real basis of Hadoop is **MapReduce** and its key characteristics are batch processing, no limits on passes over the data, no time or memory constraints. 

There are a number of ideas that enable these characteristics and define Hadoop MapReduce. 

- The design is such that hardware failure is expected and will be handled quickly and without losing or corrupting data. 

- Priority is on scaling out rather than up meaning that adding more commodity machines is preferable to fewer high-end ones. Thus, scalability in Hadoop is relatively cheap and seamless. 

- Hadoop processes data sequentially, avoiding random access, and also promotes data locality awareness. 

These properties ensure that processing is orders of magnitude faster and the expensive process of moving large amounts of data is avoided where possible.


## Spark

The simple MapReduce programming model of Hadoop is attractive and is utilized extensively in industry, but performance on certain tasks remain sub-optimal which gave rise to Spark to provide a speedup over Hadoop. 

**Spark** is a data processing engine for big data sets that is also open-source and maintained by the Apache Foundation. 

The introduction of an abstraction called **resilient distributed datasets (RDDs)** was the foundation that allowed Spark to excel and gain a huge speedup over Hadoop at certain tasks.

RDDs are fault-tolerant collections of elements that can be executed in parallel by distribution among multiple nodes in a cluster. 

The key to the speed of Spark is that any operation performed on an RDD is done in **memory** rather than on disk.


**Spark allows two types of operations on RDDs:** transformations and actions. 

Actions are used to apply computation and obtain a result while transformations result in the creation of a new RDD. 

The distribution of these operations is done by Spark and does not need direction from the user.

The operations performed on an RDD are managed by using a directed acyclic graph (DAG). 

In a Spark DAG, each RDD is represented as a node while the operations form the edges. 

The fault-tolerant property of RDDs comes from the fact that in part of an RDD is lost then it can be re-computed from the original dataset by using the lineage of operations which are stored in the graph.


## Spark vs Hadoop

Spark results in a massive speedup for certain tasks due to the fact that Spark processes data in RAM (random access memory) while Hadoop reads and writes files to HDFS which is on disk 

Spark can use HDFS as a data source but will still process the data in RAM rather than on disk as is the case with Hadoop. 

RAM is much faster than disk for two reasons:

- RAM uses solid-state technology to store information while disk does this magnetically. 

- RAM is much closer to the CPU than information stored on disk and has a faster connection, so data in RAM is accessed much faster.

This technical difference results in speedups of many orders of magnitude for applications where the same dataset is reused multiple times. 

Hadoop results in significant delays (latency) for these tasks because a separate MapReduce job is required for each query which involves reloading the data from disk each time. 

With Spark, the data remains in RAM; so it can be read from memory instead of disk which results in Spark exhibiting speedups of up to 100x over Hadoop in certain cases where we reuse the same data multiple times. 

In cases where there is **data reuse**, Spark should be chosen over Hadoop. 

Examples: iterative jobs and interactive analysis.

A common example of an iterative task that repeatedly uses the same dataset is the training of a machine learning (ML) model. 

ML models are often trained by iteratively passing over the same training dataset in order to try and reach the global minimum of the error function by using an optimisation algorithm such as gradient descent. 

The level of increased performance achieved by Spark in a ML task becomes more prominent the more times the data is queried. 

There would be no speedup evident if you were to train an ML model on Hadoop and Spark using only one pass over the data (epoch) since the data needs to be loaded from disk into RAM for the first iteration on Spark. 

However, each subsequent iteration on Spark will run in a fraction of the time while each subsequent Hadoop iteration will take the same amount of time as the very first iteration as the data is retrieved from disk each time. 

Thus, Spark is generally preferable to Hadoop when dealing with ML applications.

There are circumstances where the in-memory computation of Spark falls short. 

If the data sets we are dealing with are so large that they exceed available RAM then Hadoop is the preferred choice. In addition, Hadoop is relatively easy and cheap to scale when compared with Spark. 

Thus, a business under time constraints would likely be best served with Spark, but a business with capital constraints may be better served by the cheaper setup and scalability of Hadoop.


----------


## Tips for Optimizing a Spark Job

Spark is commonly used to apply transformations on data (usually structured data). 

There are two scenarios in which it is particularly useful. 

1. When the data to be processed is too large for the available computing and memory resources. 

2. Spark can also be used to accelerate a calculation by using several machines within the same network.

The objective here is to develop a strategy for optimizing a Spark job when resources are limited. Thus, we can influence many Spark configurations before using cluster elasticity. 

The strategy here is greedy - we make the best choice at each stage of the process without going backwards. 

The purpose here is to provide a clear methodology that is easy to test on various use cases. Then test these recommendations on a shareable example and give a template code that allows the experiment to be repeated on another Spark job.

### Application on a toy use case

We use a toy use case to design a Spark job. The processing groups French cities according to weather and demographic variables. This task is called unsupervised classification or clustering in Machine Learning. The example illustrates common features of a Spark pipeline, i.e. a data pre-processing phase (loading, cleaning, merging of different sources, feature engineering), the estimation of the Machine Learning model parameters, and finally the storage of the results to disk. More details on this experiment are available on the code repository.

The configuration settings found are probably a sub-optimal solution, but it offers a much faster alternative than an exhaustive search, especially for big data processing.

Apache Spark is an analytics engine for large-scale data processing. It provides an interface for programming entire clusters with implicit data parallelism and fault tolerance and stores intermediate results in memory (RAM and disk).

The processing at the heart of Spark makes extensive use of **functional programming** to solve the problem of scaling up in Big Data. 

The source code is mainly in Scala, but Spark also offers APIs in Python, Java, R and SQL at a higher level, offering almost equivalent possibilities without any loss of performance in most cases, except for user-defined functions (UDF). 

The Spark driver called the master node orchestrates the execution of the processing and its distribution among the Spark executors (also called slave nodes). 

The driver is not necessarily hosted by the computing cluster, it can be an external client. 

The cluster manager manages the available resources of the cluster in real time which allocates the requested resources to the Spark driver if they are available.

### Decomposition of a Spark job

A Spark job is a sequence of **stages** that are composed of **tasks** which can be represented by a Directed Acyclic Graph (DAG). 

An example of a Spark job is an Extract Transform Log (ETL) data processing pipeline. 

_Stages_ are often delimited by a data transfer in the network between the executing nodes such as a join operation between two tables. 

A _task_ is a unit of execution in Spark that is assigned to a partition of data.

### Lazy Evaluation

_Lazy Evaluation_ is a trick commonly used for large data processing. Indeed, when data exceeds memory resources, a strategy is needed to optimise the computation. 

Lazy Evaluation means triggering processing only when a Spark _action_ is run and not a Spark _transformation_.

Transformations are not executed until an action has been called which allows Spark to prepare a logical and physical execution plan to perform the action efficiently.

### Example of Lazy Execution

An action is called to return the first row of a dataframe to the driver after several transformations. 

Spark can reorganize the execution plan of the previous transformations to get the first transformed row more quickly by managing the memory and computation. 

In fact, only the partition of data containing the first row of that dataframe needs to be processed which  frees memory and prevents unnecessary processing.

### Wide and narrow transformations

The Spark transformations are divided into two categories: wide and narrow transformations. 

The difference between these two types is the need for a redistribution of the data partitions in the network between the executing nodes which is called a **shuffle&l** in Spark terminology.

**Wide transformations** requiring a shuffle are the most expensive, and the processing time is longer depending on the size of the data exchanged and the network latency in the cluster.

### How to modify the configuration settings of a Spark job?

There are three ways to modify the configurations of a Spark job:

- By using the configuration files present in the Spark root folder. These changes affect the Spark cluster and all its applications.

- In the Spark application code. 

- The values defined in the configuration files are considered first, then the arguments passed as parameters to ´lspark-submit, then the values which are configured directly in the application code are last.  

These configuration parameters are visible as read-only in the Environment tab of the Spark GUI.

In the code associated with this article, the parameters are defined directly in the Spark application code.

### Measure if an optimization is necessary

Optimizing a process is a time-consuming and therefore costly step in a project.

Usually, the constraints are linked to the use case and are defined in the service level agreement (SLA) with the stakeholders. We monitor the relevant metrics (such as processing time, allocated memory) while checking their compliance with the SLA.

Estimating the time needed to optimise an application and reach an objective is mot an easy task. This article does not pretend to do so, but aims to suggest actions that can be used quickly instead. 

These recommendations provide areas for improvement. 


### Recommendation 1: Use the Apache Parquet file format

The Apache Parquet format is officially a column-oriented storage which is more of a hybrid format between row and column storage that is used for tabular data where the data in the same column are stored contiguously.

This format is particularly suitable when performing queries (transformations) on a subset of columns and on a large dataframe because it loads only the data associated with the required columns into memory.

As the compression scheme and the encoding are specific to each column according to the typing, it improves the reading and writing of these binary files and their size on disk.

These advantages make it a very interesting alternative to the CSV format, so parquet is the format recommended by Spark and the default format for writing.

### Recommendation 2: Maximise parallelism in Spark

Spark’s efficiency is based on its ability to **process several tasks in parallel** at scale which is why optimizing a Spark job often means reading and processing as much data as possible in parallel. To achieve this goal, it is necessary to **split a dataset into several partitions**.

Partitioning a dataset is a way of arranging the data into configurable, readable subsets of contiguous data blocks on disk which can be read and processed independently and in parallel. It is this independence that enables massive data processing. 

Ideally, Spark organises one thread per task and per CPU core where each task is related to a single partition. 

Thus, a first intuition is to configure a number of partitions at least as large as the number of available CPU cores. 

All cores should be occupied most of the time during the execution of the Spark job. If one of the cores is available at any time, it should be able to process a job associated with a remaining partition. 

The goal is to avoid bottlenecks by splitting the Spark job stages into a large number of tasks which is crucial in a distributed computing cluster. 

The following diagram illustrates this division between the machines in the network.

Partitions can be created:

- When reading the data by setting the spark.sql.files.maxPartitionBytes parameter (default is 128 MB).

  A good situation is when the data is already stored in several partitions on disk such as a dataset in parquet format with a folder containing data partition files between 100 and 150 MB in size.

- Directly in the Spark application code using the Dataframe API.

This last method coalesce decreases the number of partitions while avoiding a shuffle in the network.

We might be tempted to increase the number of partitions by lowering the value of the spark.sql.files.maxPartitionBytes parameter, but this choice can lead to the small file problem. 

There is a deterioration of I/O performance due to the operations performed by the file system (e.g. opening, closing, listing files) which is often amplified with a distributed file system like HDFS. 

Scheduling problems can also be observed if the number of partitions is too large.

In practice, the maxPartitionBytes parameter should be defined empirically according to the available resources.

### Recommendation 3: Beware of shuffle operations

There is a specific type of partition in Spark called a shuffle partition which are created during the stages of a job involving a shuffle - when a wide transformation (such as groupBy() or join()) is performed. 

The setting of these partitions impacts both the network and the read/write disk resources.

The value of spark.sql.shuffle.partitions can be modified to control the number of partitions. By default, this is set to 200 which may be too high for some processing and results in too many partitions being exchanged in the network between the executing nodes. 

The partitions parameter should be adjusted according to the size of the data - start with a value at least equal to the number of CPU cores in the cluster.

Spark stores the intermediate results of a shuffle operation on the local disks of the executor machines, so the quality of the disks (especially the I/O quality) is really important. For example, the use of SSD disks will significantly improve performance for this type of transformation.

The table below describes the main parameters that we can also influence.

### Recommendation 4: Use Broadcast Hash Join

A join between several dataframe is a common operation. 

In a distributed context, a large amount of data is exchanged in the network between the executing nodes to perform the join. 

Depending on the size of the tables, this exchange causes network latency which slows down processing. 

Spark offers several join strategies to optimise this operation such as Broadcast Hash Join (BHJ).

BHJ is suitable when one of the merged dataframe is “sufficiently” small to be duplicated in memory on all the executing nodes (broadcast operation). 

The diagram below illustrates how this strategy works.

The second dataframe is classically decomposed into partitions distributed among the cluster nodes. 

By duplicating the smallest table, the join no longer requires any significant data exchange in the cluster except the broadcast of the table beforehand which greatly improves the speed of the join. 

The Spark configuration parameter to modify is spark.sql.autoBroadcastHashJoin. 

The default value is 10 MB which means this method is chosen if one of the two tables is smaller than this size. 

If sufficient memory is available, it may help to increase this value or set it to -1 to force Spark to use it.

### Recommendation 5: Cache intermediate results

To optimise computations and manage memory resources, Spark uses lazy evaluation and a DAG to describe a job which offers the possibility of quickly recalculating the steps before an action if necessary by executing only part of the DAG. 

To take full advantage of this functionality, it is best to store expensive intermediate results if several operations use them downstream of the DAG. If an action is run, its computation can be based on these intermediate results and only replay a part of the DAG before an action.

If caching can speed up execution of a job, we pay a cost when these results are written to memory and/or disk, so it should be tested at different locations in the processing pipeline whether the total time saving outweighs the cost which is especially relevant when there are several paths on the DAG.

A table can be cached using the following command:

The different caching options are described in the table below:

> Caching is a kind of Spark transformation, so it is performed when an action is run. If the computation of this action involves only a sub-part of the data, then only the intermediate results for that sub-part will be stored. For example, if the take(1) action collecting the first row had been called, only the partition containing that first row would have been cached.

### Recommendation 6: Manage the memory of the executor nodes

The memory of a Spark executor is broken down as follows:

By default, the spark.memory.fraction parameter is set to 0.6 which means that 60% of the memory is allocated for execution and 40% for storage, once the reserved memory is removed. 

The default is 300 MB and is used to prevent out of memory (OOM) errors.

We can modify the following two parameters:

- spark.executor.memory
- spark.memory.fraction


The six recommendations use a greedy approach aims to maximise the probability of reducing computation time. 

The following diagram summarises the method proposed by associating the recommendations at each stage:



## Spark Performance Tips

Here are some tips to avoid performance problems while running PySpark code.

> Start queries with filter and select data to shorten the size of the datasets

Golden rule: you will always want to filter and select only variables you’re actually using when creating scripts. That simple action reduces the size of the data, which converts into faster scripts.

> groupBy is painful for large data

Yes, it will be a slow grouping proportionally to the size of your dataset. Even though that is a lazy function — meaning it will only actually be performed once you request an action like display() or count() or collect() — it still needs to read all the data and that takes time.

> Processing happens on memory. More memory lead time is faster.

The great catch of Spark is that the processing happens on memory rather than on disk, thus it is much faster. Ergo, the more memory your cluster has, the faster it will be.

> Avoid from loops as much as possible. 

If you are used to performing loop operations in your Python scripts, remember that PySpark is not the place to run loops since it will take a long time to run, given the size of the data and that this command will be split in many nodes.

> Always filter by partition of the table

Many tables in Databricks are partitioned by a given field. If you know which one that is, use that variable as your filter.

> PySpark uses Camel Case

Most of the functions will be camelCase. 

### Useful Code Snippets

Here are some useful code snippets when working with Databricks:



## PySpark Example

This is a step-by-step tutorial on how to use Spark to perform exploratory data analysis on larger than memory datasets.

Analyzing datasets that are larger than the available RAM memory using Jupyter notebooks and Pandas Data Frames is a challenging issue. 

Here we present a method for performing exploratory analysis on a large data set to identify and filter out unnecessary data.

The result is a pyspark.sql.dataframe variable. At this point the data is not actually loaded into the RAM memory. 

Data is only loaded when an action is called on the pyspark variable, an action that needs to return a computed value. If I ask for instance for a count of the number of products in the dataset, Spark is smart enough not to try and load the whole dataset to compute the value. 

```py
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col

    # create a spark session
    sc = SparkSession.builder.master("local").appName("Test").getOrCreate()

    raw_data = sc.read.options(delimiter="\t",header=True).csv("en.openfoodfacts.org.products.csv")

    raw_data.printSchema()

    # compute number of products per country 
    BDD_countries = raw_data.groupBy("countries_tags").count().persist()

    BDD_countries.printSchema()

    # filter dataframe to only countries with at least 5000 products
    BDD_res = BDD_countries.filter(col("count") > 5000).orderBy("count",ascending = False).toPandas()
```

Now we can for instance filter out all the products that are not available in France and perform the rest of the analysis on a smaller, easier to use dataset.



## Kafka

Apache Kafka publishes and subscribes (pub/sub) to streams of records based on a **message queue** mechanism. 

Publish-Subscribe (pub/sub) is a popular messaging pattern that is commonly used in to help distribute data and scale. 

The pub/sub messaging pattern can be easily implemented using an event broker such as Solace PubSub+, Kafka, RabbitMQ, and ActiveMQ. 

With an event broker, there is a set of applications known as **producers** and another set of applications known as **consumers**. 

Producers are responsible for publishing data to the broker and consumers are responsible for consuming data from the event broker. Thus, Kafka has an event-driven architecture. 

By introducing a message broker, we have removed the need for producers to directly communicate with the consumers which ensures a loosely coupled architecture. In addition, the broker is responsible for managing connections, security, and subscription interests rather than implementing this logic in the applications.

Since Kafka is a broker-based solution, it aims to maintain streams of data as records within a cluster of servers. 

Kafka servers provide data persistence by storing messages across multiple server instances in topics.

The major tasks that Kafka performs are: read messages, process messages, write messages to a topic, or aggregate messages for a certain period of time. 

The time window can be fixed (hourly aggregation) or sliding (hourly aggregation starting from a certain point in time).

Kafka was developed based on five core APIs for Java and Scala:

1. **Admin API:** Used to manage topics, brokers and other Kafka objects

2. **Producer API:** Used to write a stream of data to the Kafka topics

3. **Consumer API:** Used to read streams of data from topics in the Kafka cluster

4. **Kafka Streams API:** Used for stream processing applications and microservices

5. **Kafka Connect API:** Used to build and run reusable data import/export connectors that consume or produce data streams from and to external applications in order to integrate with Kafka.


### Key Concepts

A **topic** is a category of messages that a consumer can subscribe to and it’s identified by its name. This way, consumers aren’t automatically receiving every message that has been published to the cluster, but instead they subscribe only to the relevant ones.

Topics are split into ordered **partitions** which are numbered, starting incrementally from 0. 

When creating a new topic we must explicitly specify its number of partitions.

Each message within a partition gets an incremental ID called an **offset**. 

The data will be stored only for a limited period of time and once written to a partition, it cannot be modified. 

Unless a key is provided, data is automatically assigned to a random partition. 

There can be any number of partitions per topic, and having more partitions leads to more parallelism.

A Kafka cluster consists of one or multiple **brokers**, each of them being identified by a unique ID. 

After connecting to any broker, the user is connected to the entire cluster.

### Distribution

Every partition is copied across one or more servers in order to assure fault tolerance. 

There is only one server that has the role of the **leader** which handles all read/write requests for the partition, and 0 or more servers that act as the **followers** which have to replicate the leader.

If the leader fails, the new leader will be automatically assigned from the list of followers.

### Messaging

Messaging is the act to send a message from one place to another which has three principal actors:

- Producer: Who produces and send the messages to one or more queues. 

- Queue: A buffer data structure that receives (from the producers) and delivers messages (to the consumers) in a FIFO (First-In First-Out) way. When a message is delivered, it id removed forever from the queue (there is no chance to get it back). 

- Consumer: Who is subscribed to one or more queues and receives their messages when published.

### Producer/Consumer

This follows the publish/subscribe design pattern.

The **producer** writes data to topics and can choose to receive three types of acknowledgements for data writes:

- Acks = 0: will not wait for acknowledgements which may cause loss of data

- Acks = 1: will wait only for leader’s acknowledgement which may cause limited data loss

- Acks = all: will wait for acknowledgement from leader and replicas which results in no data loss

The **consumer** reads from a server and can be on separate processes or separate machines. 

Consumers have a group-name label and each message published is sent to one consumer instance within each consumer group that is subscribed to it.

----------


## Distributed Tools

- [Envoy](https://www.envoyproxy.io/)


## References

[1]: [Hadoop vs Spark: Overview and Comparison](https://towardsdatascience.com/hadoop-vs-spark-overview-and-comparison-f62c99d0ee15)

[2]: [6 recommendations for optimizing a Spark job](https://towardsdatascience.com/6-recommendations-for-optimizing-a-spark-job-5899ec269b4b)

[3]: [Useful Code Snippets for PySpark](https://towardsdatascience.com/useful-code-snippets-for-pyspark-c0e0c00f0269)

[4]: [PySpark Example for Dealing with Larger than Memory Datasets](https://towardsdatascience.com/a-pyspark-example-for-dealing-with-larger-than-memory-datasets-70dbc82b0e98)


[5]: [What Is Apache Kafka?](https://medium.com/cognizantsoftvision-guildhall/what-is-apache-kafka-a52a206d14c8)

[6]: [Building Python Microservices with Apache Kafka: All Gain, No Pain](https://towardsdatascience.com/building-python-microservices-with-apache-kafka-all-gain-no-pain-1435836a3054)

[7]: [Kafka: Learn All About Consumers](https://medium.com/@maverick11/learn-all-about-kafka-consumers-fba36b94e724)


[Serverless computing vs containers](https://www.cloudflare.com/learning/serverless/serverless-vs-containers/)
