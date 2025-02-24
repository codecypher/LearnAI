# SQL


## SQL Tips

Here are some tips on SQL optimization [1]:

### Avoid using SELECT *

```sql
  # correct example
  select name, age from user where id = 1;
```

### Avoid Using SELECT DISTINCT; Use GROUP BY

SELECT DISTINCT can be costly because it requires sorting and filtering the results to remove duplicates [3]. 

It is better to ensure that the data being queried is unique by design—using primary keys or unique constraints.

```sql
  SELECT DISTINCT department FROM employees;
```

The following query with the GROUP BY clause is much more helpful:

```sql
  SELECT department FROM employees GROUP BY department;
```

GROUP BY can be more efficient, especially with proper indexing.


### Replace UNION with UNION ALL

```sql
# correct example
(select * from user where id=1)
union all
(select * from user where id=2);
```

### Small table drives big table

If we want to check the list of orders placed by all valid users.

This can be achieved using the in keyword:

```sql
select * from order
where user_id in (select id from user where status=1)
```

This can also be achieved using the exists keyword:

```sql
select * from order
where exists (select 1 from user where order.user_id = user.id and status=1)
```

Because of the in a keyword is included in the SQL statement, it will execute the subquery statement in first and then execute the statement outside in. If the amount of data in is small, the query speed is faster as a condition.

And if the SQL statement contains the exists keyword, it executes the statement to the left of exists first (main query statement).

Then use it as a condition to match the statement on the right. If it matches, you can query the data. If there is no match, the data is filtered out.

Here, the order table has 10,000 pieces of data and the user table has 100 pieces of data. Thus, the order is a large table and the user is a small table.

- in applies to the large table on the left and the small table on the right.

- exists applies to the small table on the left and the large table on the right.


### Bulk operations

What if you have a batch of data that needs to be inserted after business processing?

We could insert data one by one in a loop.

```sql
insert into order(id,code,user_id)
values(123,'001',100);
```

However, this operation requires multiple requests to the database to complete the insertion of this batch of data. Every time we request the database remotely, it will consume a certain amount of performance.

If our code needs to request the database multiple times to complete this business function, it will inevitably consume more performance.

The correct approach is to provide a method to insert data in batches.

```sql
# correct example
orderMapper.insertBatch(list);
# insert into order(id,code,user_id)
# values(123,'001',100),(124,'002',100),(125,'003',101);
```

Then we only need to request the database remotely once, and the SQL performance will be improved. The more data, the greater the improvement.

However, it is not recommended to operate too much data in batches at one time. If there is too much data, the database response will be very slow.

Batch operations need to grasp a degree, and it is recommended that each batch of data be controlled within 500 as much as possible. If the data is more than 500, it will be processed in multiple batches.

### Limit Query Results

Sometimes, we need to query the first item of some data: query the first order placed by a user and we want to see the time of his first order.

We could query orders according to the user id, sort by order time, find out all the order data of the user and get an order set.

Although this approach has no problem in function, it is very inefficient. It needs to query all the data first which is a waste of resources.

```sql
# correct example
select id, create_date
 from order
where user_id=123
order by create_date asc
limit 1;
```

Use limit 1 to return only the data with the smallest order time of the user.

When deleting or modifying data, in order to prevent misoperation, resulting in deletion or modification of irrelevant data, the limit can also be added at the end of the SQL statement.

```sql
update order
set status=0,edit_time=now(3)
where id>=100 and id<200 limit 100;
```

Then, even if the wrong operation (such as the id is wrong) it will not affect too much data.

### Do not use too many values with IN

For the batch query interface, we usually use th in keyword to filter out data.

Suppose we want to query user information in batches through some specified ids.

If we do not impose any restrictions, the query statement may query a lot of data at one time, which may easily cause the interface to time out.

```sql
select id,name from category
where id in (1,2,3...100)
limit 500;
```

We can limit the data with the limit in SQL.

If there are more than 500 records in ids, we can use multiple threads to query the data in batches. Only 500 records are checked in each batch and the queried data are aggregated and returned.

However, this is only a temporary solution and is not suitable for scenes with too many ids. Because there are too many ids, even if the data can be quickly detected, if the amount of data returned is too large, the network transmission is very performance-intensive and the interface performance is not much better.


### Incremental query

Sometimes, we need to query data through a remote interface and synchronize it to another database.


We could get all the data directly then sync it. But if there is a lot of data, the query performance will be very poor.

```
select * from user
where id>#{lastId} and create_time >= #{lastCreateTime}
limit 100;
```

In ascending order of id and time, only one batch of data is synchronized each time and this batch of data has only 100 records.

After each synchronization is completed, save the largest id and time of the 100 pieces of data for use when synchronizing the next batch of data.

This incremental query method can improve the efficiency of a single query.


### Efficient paging

When querying data on the list page, we generally paginate the query interface to avoid returning too much data at one time and affecting the performance of the query.

The limit keyword commonly used for paging in MySQL:

```sql
select id,name,age
from user limit 10,20;
```

If the amount of data in the table is small, using the limit keyword for paging is no problem. But if there is a lot of data in the table, there will be performance problems.

```sql
# better
select id,name,age
from user where id > 1000000 limit 20;
```

First, we find the largest id of the last paging and use the index on the id to query. However, the id is required to be continuous and ordered.

We can also use between to optimize pagination.

```sql
select id,name,age
from user where id between 1000000 and 1000020;
```

The between should be paginated on the unique index or there will be an inconsistent size of each page.


### Replacing subqueries with join queries

If we need to query data from more than two tables in MySQL, there are generally two implementation methods: subquery and join query.

```sql
select * from order
where user_id in (select id from user where status=1)
```

Sub-query statements can be implemented using rhe in the keyword and the conditions of one query statement fall within the query results of another select statement. The program runs on the nested innermost statement first and then runs the outer statement.

The advantage of a subquery statement is that it is simple and structured if the number of tables involved is small.

But when MySQL executes sub-queries, temporary tables will need to be created. After the query is completed, these temporary tables need to be deleted which has some additional performance consumption.

Therefore, the query can be changed to a connection query.

```sql
select o.* from order o
inner join user u on o.user_id = u.id
where u.status=1
```

### Join tables should not be too many

If there are too many join, MySQL will be very complicated when selecting indexes and it is easy to choose the wrong index.

If there is no hit, the nested loop join is to read a row of data from the two tables for pairwise comparison, and the complexity is n².

So we should try to control the number of joined tables.

```sql
# correct example
select a.name,b.name.c.name,a.d_name
from a
inner join b on a.id = b.a_id
inner join c on c.b_id = b.id
```

If we need to query the data in other tables in the implementation of the business scenario, we can have redundant special fields in the a, b, and c tables such as the redundant d_name field in the table a to save the data to be queried.

However, the number of joined tables should be determined according to the actual situation of the system. It cannot be generalized. In general, the less the better.

### Inner join note

We generally use the join keyword to query  multiple tables.

The most commonly used joins are `left join` and `inner join`.

- left join: Find the intersection of two tables plus the remaining data in the left table.

- inner join: finds the data of the intersection of two tables.

```sql
select o.id,o.code,u.name
from order o
inner join user u on o.user_id = u.id
where u.status=1;
```

If two tables are related using inner join, MySQL will automatically select the small table in the two tables to drive the large table, so there will not be too many performance problems.

```sql
select o.id,o.code,u.name
from order o
left join user u on o.user_id = u.id
where u.status=1;
```

If two tables are associated using left join, MySQL will use the left join keyword to drive the table on the right by default. If there is a lot of data in the left table, there will be performance problems.

When using left jointo query, use a small table on the left and a large table on the right. In general, try to use left join as little as possible.


### Use Indexes for Faster Retrieval

Indexes can improve query performance by allowing the database to quickly find rows rather than scanning the entire table. 

Indexes are useful for columns frequently used in WHERE, JOIN, and ORDER BY clauses.


### Limit the number of indexes

We all know that indexes can significantly improve the performance of query SQL but more number  indexes is not the better.

When new data is added to the table, an index needs to be created for it at the same time and the index requires additional storage space and a certain performance consumption.

The number of indexes in a single table should be less than 5 if possible and the number of fields in a single index should not exceed 5.

If your system has low concurrency and the amount of data in the table is not too much then more than 5 indexes can be used.

For high-concurrency systems, try to limit to using 5 indexes on a single table.

How can a high concurrency system optimize the number of indexes?

If we can build a joint index, do not build a single index and we can delete a useless single index.

### Choose the appropriate field type

char represents a fixed string type and the storage space of the field of this type is fixed which will waste storage space.

```sql
alter table order
add column code char(20) NOT NULL;
```

varchar represents a variable-length string type, and the field storage space of this type will be adjusted according to the length of the actual data, without wasting storage space.

```sql
alter table order
add column code varchar(20) NOT NULL;
```

If it is a field with a fixed length (such as the user’s mobile phone number), it is usually 11 bits and can be defined as a char type with a length of 11 bytes.

It is recommended to change the enterprise name to varchar type. The storage space of variable-length fields is small, which can save storage space, and for queries, the search efficiency in a relatively small field is obviously higher.

When we choose field types, we should follow these principles:

If we can use numeric types, we do need strings since characters tend to be slower to process than numbers.

Use small types as much as possible such as  bit to store boolean values, tinyint to store enumeration values, etc.

- A fixed-length string field, of type char.

- A variable-length string field, of type varchar.

- decimal is used for the amount field to avoid the problem of loss of precision.


### Improve the efficiency of GROUP BY

We have many business scenarios that need to use the group by keyword which is used to group and avoid duplicates.

It is often used in conjunction with having which means grouping and then filtering data according to certain conditions.

```sql
# incorrect example
select user_id,user_name from order
group by user_id
having user_id <= 200;
```

However, this query has poor performance. It first groups all orders according to the user id and then filters users whose user id is greater than or equal to 200.

We can use the where condition to filter out the redundant data before grouping to improve efficiency when grouping.

```sql
# correct example
select user_id,user_name from order
where user_id <= 200
group by user_id
```

Before our SQL statements do some time-consuming operations, we should reduce the data range as much as possible which can improve the overall performance of SQL.

### Index optimization

In SQL optimization, there is a very important content: index optimization.

In many cases, the execution efficiency of SQL statements is very different when the index is used and the index is not used. Therefore, index optimization is the first choice for SQL optimization.

The first step in index optimization is to check whether the SQL statement is indexed.

How to check whether the query has used the index?

We can use the explain command to view the execution plan of MySQL.

```sql
  explain select * from `order` where code='002';
```

Sometimes MySQL chooses the wrong index, but we can use force index to force the query to use a certain index.



## N+1 Query Problem

Here are two solutions to the problem:

1. Join the authors in the SQL query.

2. Get the meals and then join the drinks with your programming language

We do not query the database for every drink. In one query, we get all the meals and in the other we query the drinks and join them to the corresponding meals.

When to use one solution or the other?

In this app, every meal includes just one drink, but what if a meal includes more than one drink?

In that case, the first solution cannot help us since the SQL query is going to repeat the record for every drink in a meal.

Thus, we should use the second solution when we want to first query the meals and then get the drinks to join them to the corresponding meals.


## Advanced SQL

Here are some tips on advanced SQL techniques [4]:

### Window Functions

A window function in SQL is a function that uses values from one or multiple rows to return a value for each row. 

In contrast, an aggregate function returns a single value for multiple rows.

Window functions have an OVER clause. Otherwise, it is an aggregate or single-row (scalar) function.

Window functions can perform calculations such as running totals, averages, counts, ranking, and more. 

```sql
SELECT 
    Sale_Person_ID, 
    Department, 
    Sales_Amount,
    SUM(Sales_Amount) OVER (PARTITION BY Department) AS dept_total
FROM 
    promo_sales;
```

Common types of window functions include ranking functions, aggregate functions, offset functions, and distribution functions [4].

1. Ranking Functions: Ranking functions assign a rank or row number to each row within a partition of a result set.

2. Aggregate Functions: Aggregate functions are used to perform calculation or run statistics across a set of rows related to the current row.

3. Offset Functions: Offset functions allow access to data from other rows in relation to the current row. They are used when you need to compare values between rows or when you run time-series analysis or trend detection.

4. Distribution Functions: Distribution functions calculate the relative position of a value within a group of values and also help you understand the distribution of values.


### Subqueries

A subquery is a query within another SQL query; often called a nested query or inner query [4]. 


A subquery can be used to generate a new column, new table, or some conditions to further restrict the data to be retrieved in the main query.

But it is important to use subqueries sparingly since overuse can lead to performance issues, especially with large datasets.

#### Subquery for new column generation

This time we’d like to add a new column to show the difference between each sales person’s sales amount and the department average.
SELECT 
    Sale_Person_ID, 
    Department, 
    Sales_Amount,
    Sales_Amount - (SELECT AVG(Sales_Amount) OVER (PARTITION BY Department) FROM promo_sales) AS sales_diff
FROM 
    promo_sales;

#### Subquery to create a new table

To determine which department is the most cost-efficient, we need to calculate the return on advertising spend for each department. 

We can use a subquery to create a new table that includes the total sales amounts and marketing costs for these departments [4].

```sql
SELECT 
    Department, 
    dept_ttl,
    Mkt_Cost,
    dept_ttl/Mkt_Cost AS ROAS
FROM 
    (SELECT
        s.Department,
        SUM(s.Sales_Amount) AS dept_ttl,
        c.Mkt_Cost
     FROM 
        promo_sales s
     GROUP BY s.Department
     LEFT JOIN 
        mkt_cost c
     ON s.Department=c.Department
     ) 
```

#### Subquery to create restrictive conditions

A subquery can also be used to select sales persons whose sales amount exceeded the average amount of all sales persons. 

```sql
SELECT 
    Sale_Person_ID, 
    Department, 
    Sales_Amount
FROM 
    promo_sales
WHERE 
    Sales_Amount > (SELECT AVG(salary) FROM promo_sales);
```

#### Correlated Subquery

A correlated subquery (CSQ) depends on the outer query for its values and is executed once for each row in the outer query.

CSQ can be used to find the sales persons whose sales performance were above the average of their department during the promotion.

```sql
SELECT 
    ps_1.Sale_Person_ID, 
    ps_1.Department, 
    ps_1.Sales_Amount
FROM 
    promo_sales ps_1
WHERE 
    ps_1.Sales_Amount > (
          SELECT AVG(ps_2.Sales_Amount) 
          FROM promo_sales ps_2 
          WHERE ps_2.Department = ps_1.Department
);
```

### Common Table Expressions

A Common Table Expression (CTE) is a named temporary result set that exists within the scope of a single SQL statement [4]. 

CTEs are defined using a WITH clause and can be referenced one or more times in a subsequent SELECT, INSERT, UPDATE, DELETE, or MERGE statement [4].

There are primarily two types of CTEs in SQL:

Non-recursive CTEs: used to simplify complex queries by breaking them down the query into more manageable parts.

Recursive CTEs: reference themselves within their definitions which allows for hierarchical or tree-structure data.

Here is a non-recursive CTEs to calculate the average sales amount from each department and compare it with the store average during the promotion.

```sql
WITH dept_avg AS (
    SELECT 
        Department,
        AVG(Sales_Amount) AS dept_avg
    FROM 
        promo_sales
    GROUP BY 
        Department
),

store_avg AS (
    SELECT AVG(Sales_Amount) AS store_avg
    FROM promo_sales
)

SELECT 
    d.Department,
    d.dept_avg,
    s.store_avg,
    d.dept_avg - s.store_avg AS diff
FROM 
    dept_avg d
CROSS JOIN 
    store_avg s;
```

Since a recursive CTE can deal with hierarchical data, we can generate a sequence of numbers from 1 to 10.

```sql
WITH RECURSIVE sequence_by_10(n) AS (
    SELECT 1
    UNION ALL
    SELECT n + 1
    FROM sequence_by_10
    WHERE n < 10
)
SELECT n FROM sequence_by_10;
```

CTE improves the readability and maintainability of complex queries by simplifying them.



## References

[1]: [15 Best Practices for SQL Optimization](https://betterprogramming.pub/15-best-practices-for-sql-optimization-956759626321)

[2]: [How the N+1 Query Can Burn Your Database](https://betterprogramming.pub/how-the-n-1-query-can-burn-your-database-3841c93987e5)

[3]: [5 Tips for Improving SQL Query Performance](https://www.kdnuggets.com/5-tips-for-improving-sql-query-performance)

[4]: [The Most Useful Advanced SQL Techniques to Succeed in the Tech Industry](https://towardsdatascience.com/the-most-useful-advanced-sql-techniques-to-succeed-in-the-tech-industry-0f0690e8386c)

