# Plots and Graphs

Matplolib is the basis of image visualization in Python but there are other alternatives such as seaborn and plotly (which is browser-based).

## Choosing Graphs

[This Is How You Should Be Visualizing Your Data](https://towardsdatascience.com/this-is-how-you-should-be-visualizing-your-data-4512495c007b)

[An Atlas of Simple Matplotlib Charts](https://medium.com/geekculture/an-atlas-of-simple-matplotlib-charts-2f6fd32ca4cf)


## Python Graphing Libraries

- Plotly
- Seaborn
- Bokeh
- Cufflinks

- Altair
- Ydata Profiling

- SciPy
- Statsmodels


## Tips for Improving Charts with Matplotlib

### Remove Chart Junk

Here are a few things you can do to make your figures clearer [20]:

- Using titles and labels sparingly but effectively
- Avoid complex vocabulary and jargon
- Remove unnecessary gridlines and borders
- Remove background images
- Avoid overly ornate fonts
- Avoid using unnecessary special effects such as 3D effects and shadows

### Choose Appropriate Colors

Theew are a few general rules that can help make your figures appear more polished and professional:

- Use colour to highlight information, not distract: Colors should be used to draw attention to the most crucial aspects of your data.

- Be Consistent: When creating multiple visualisations, maintaining consistency helps your audience quickly understand new visualisations based on experience with previous ones. For example, if you are using blue for a particular category in one chart, try to use the same colour for the same category in other charts.

- Consider Color Vision Issues / Blindness: It is important to consider people with colour blindness when creating your charts. For example, avoid colours that are known to be problematic, such as red and green or blue and yellow.

Understand Color Psychology: The meaning behind colours can have important implications and can also vary between cultures. For example, red is often seen as a negative colour or a warning of danger, whereas green is seen as a positive or an indicator of growth.

### Applying a Matplotlib Theme

If you are a regular reader of my articles, you will have seen I have covered several matplotlib theme libraries in recent months [21]. 

These theme libraries allow you to instantly transform your figures from the boring standard colour scheme of matplotlib into something that is much more aesthetically pleasing.

Not only do they help with how the figures look, but they can also help improve interpretability.

There are numerous matplotlib theme libraries available including `mplcyberpunk` which lets you transform your matplotlib figure into a futuristic graph with glowing neon colors. 

### Consider Your Audience and the Story You Are Telling

When creating data visualisations, one of the most important things to keep in mind is who your audience is and the story that you want to tell.

Instead of presenting all the available data to the user in a large range of confusing and complex charts, it is best to distil the data and information down to the most relevant parts. 

This will depend on what the objective of the data analysis is, which could be defined by a client, a research project or an event organiser.



## Plotting Multiple Graphs in Matplotlib

In general, matplotlib seems to have the most functionality with low-level control. However, the syntax is verbose and inconsistent between thw different types of graphs. 

Here are four tips for plotting multiple graphs [1].

### Import libraries

```py
    import seaborn as sns # v0.11.2
    import matplotlib.pyplot as plt # v3.4.2

    sns.set(style='darkgrid', context='talk', palette='rainbow')
```

### Load datasets

```py
    # Load data using pandas
    nifty_data = pd.read_csv('NIFTY_data_2020.csv', parse_dates=["Date"], index_col='Date')
    nifty_data.head()

    # Load data using seaborn
    df = sns.load_dataset('tips')
    df.head()
```

### plt.subplots()

One way to plot multiple subplots is to use `plt.subplots()`.

```py
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    sns.histplot(data=df, x='tip', ax=ax[0])
    sns.boxplot(data=df, x='tip', ax=ax[1])

    # Set title for subplots
    ax[0].set_title("Histogram")
    ax[1].set_title("Boxplot")
```

Visualize the same set of graphs for all numerical variables in a loop:

```py
    numerical = df.select_dtypes('number').columns
    for col in numerical:
        fig, ax = plt.subplots(1, 2, figsize=(10,4))
        sns.histplot(data=df, x=col, ax=ax[0])
        sns.boxplot(data=df, x=col, ax=ax[1])
```

### plt.subplot()

Another way to visualise multiple graphs is to use `plt.subplot()`.

```py
    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    sns.histplot(data=df, x='tip', ax=ax1)
    ax2 = plt.subplot(1,2,2)
    sns.boxplot(data=df, x='tip', ax=ax2)
```

Visualize the same set of graphs for all numerical variables in a loop:

```py
    plt.figure(figsize=(14,4))
    for i, col in enumerate(numerical):
        ax = plt.subplot(1, len(numerical), i+1)
        sns.boxplot(data=df, x=col, ax=ax)
        ax.set_title(f"Boxplot of {col}")
```

### Comparison between plt.subplots() and plt.subplot()

<img width=600 src="https://miro.medium.com/max/1206/1*TH6KO5j_pKHV30MWzCS3lg.png" />


### plt.tight_layout()

When plotting multiple graphs, it is common to see labels of some subplots overlapping on their neighbor subplots:

```py
    categorical = df.select_dtypes('category').columns
    plt.figure(figsize=(8, 8))
    for i, col in enumerate(categorical):
        ax = plt.subplot(2, 2, i+1)
        sns.countplot(data=df, x=col, ax=ax)

    # Fix overlap
    plt.tight_layout()
```

### Set Title for Figure

```py
    plt.figure(figsize=(8, 8))
    for i, col in enumerate(categorical):
        ax = plt.subplot(2, 2, i+1)
        sns.countplot(data=df, x=col, ax=ax)

    # Set title for figure
    plt.suptitle('Category counts for all categorical variables')

    plt.tight_layout()
```



## Subplots

> One Figure, many Axes.  

We need to define two fundamental Matplotlib object types: Figure and Axes [18].

The _Figure_ can be considered as the outermost container that holds everything together; All other objects stay alive in this container. 

A Figure can have one or more Axes objects. 

We need an _Axes_ object to draw something; Each subplot on a Figure is actually an Axes object.

### Overlapping Axes objects

We can add multiple Axes objects to a Figure by placing them on top of each other. 

Their sizes are actually the same. 

The Axes objects share the same x-axis and y-axis by default.

```py
    import matplotlib.pyplot as plt

    # create a Figure object
    plt.figure(figsize=(14,5))

    # Figure title
    plt.title("September Sales", fontsize=15)

    # first Axes
    plt.plot(df["date"], df["store1"])

    # second Axes
    plt.plot(df["date"], df["store2"])

    # add legend
    plt.legend(["Store1", "Store2"])

    plt.show()
```

### Axes objects in different positions

We can also place Axes objects in different positions on a Figure.

There are two steps in this process:

1. Arrangement of the positions of Axes objects
2. Creating plots on each Axes object

The arrangement can be made using the `add_subplot` or `subplot` functions. 

The `subplot` function is a wrapper of `add_subplot` which is more frequently used.

We first create the schema for subplots. 

The nrows and ncols parameters can be used for this task. 

We can create as many Axes objects as needed. Each ax can be accessed with its index.

```py
    # create Figure with 4 Axes objects.
    fig, axs = plt.subplots(nrows=2, ncols=2)
```

```py
    # Figure with 2 Axes objects
    fig, axs = plt.subplots(
        figsize=(10,5),
        ncols=2,
        sharey=True
    )

    # Figure title
    fig.suptitle("Histogram", fontsize=16)

    # First Axes
    axs[0].hist(sales["price"])
    axs[0].set_title("Price", fontsize=14)

    # Second Axes
    axs[1].hist(sales["cost"], color="green")
    axs[1].set_title("Cost", fontsize=14)

    plt.show()
```

The `sharex` and `sharey` parameters can be used to eliminate redundant axis ticks. 

In the Figure above, both plots can have the same y-axis, so we can remove the y-axis from the left one by setting the value of the `sharey` parameter as `True`.


### Subplot2grid

The `subplot` function divides the Figure into cells where each subplot occupies one cell. 

The `subplot2grid` function allows subplots occupy multiple cells so we can create grid structures. 

Here we define a two-by-two grid. The position of the subplot on top is defined as the first row and first column that spans over 2 columns. The ones on the bottom are the second-row plots and each occupies a single cell.

```py
    # Create Figure
    plt.figure(figsize=(8,4))

    # top
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

    # bottom left
    ax2 = plt.subplot2grid((2, 2), (1, 0))

    # bottom right
    ax2 = plt.subplot2grid((2, 2), (1, 1))

    plt.show()
```

The `colspan` and `rowspan` parameters are used for creating subplots that occupy more than one cell.




## How to Customize Matplotlib Plots

### Colors

We can set the plotting style and customize matplotlib parameters using `rcParams` [3].

The articles [3] and [13] discuss how to use matplotlib colormaps which are used to make plots more accessible for people with color vision deficiencies.

```py
    plt.style.use('fivethirtyeight')  # customize the plotting style
    plt.style.use('default')  # reset to default style

    # show available styles
    print(plt.style.available)
```


### Data Point Labels

Here are some examples of how to label the values of data points on plots with matplotlib [4].

```py
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(x, y)
    plt.xlabel("x values", size=12)
    plt.ylabel("y values", size=12)
    plt.title("Learning more about pyplot with random numbers chart", size=15)
  
    # increase the frequency of the x and y ticks to match the actual values of x and the possible values of y
    plt.xticks(x, size=12)
    plt.yticks([i for i in range(20)], size=12)
    plt.show()
```

We can also use the `.annotate()` method [17]:

```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates

    np.random.seed(106)

    df = pd.DataFrame({
        'Date': pd.date_range('2022-04-01', '2023-03-31'),
        'Amount': np.random.randn(365).cumsum()})

    fig, ax = plt.subplots(figsize=(10,6))

    date_format = DateFormatter('%Y-%b')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # Add some room
    ax.set_ylim(-20, 15)

    # Add arrow at max value
    max_idx = np.argmax(df['Amount'])
    max_loc = df['Date'].iloc[max_idx], df['Amount'].iloc[max_idx]
    text_loc = df['Date'].iloc[max_idx+20], 10

    ax.annotate(
        text='Woah!', 
        xy=(max_loc),
        xytext=(text_loc),
        arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90")
        )

    ax.plot(df['Date'], df['Amount'])
```

### Customize Date Labels

Here is an example of customizing rhe date x-axis labels [17]:

```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates

    np.random.seed(106)

    df = pd.DataFrame({
        'Date': pd.date_range('2022-04-01', '2023-03-31'),
        'Amount': np.random.randn(365).cumsum()})

    fig, ax = plt.subplots(figsize=(10,6))

    date_format = DateFormatter('%Y-%b')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.plot(df['Date'], df['Amount'])
```


## Make Plots Prettier

Matplotlib provides a lot of customizability which can often seem overwhelming,

Here we perform the following [17]:

- We remove the spines from the top and the right sides. 

- We use the `.text()` method to add both a title and a subtitle. 

The process of adding text can be a bit trial-and-error. 

```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates

    # Import timedelta
    from datetime import timedelta

    np.random.seed(106)

    df = pd.DataFrame({
        'Date': pd.date_range('2022-04-01', '2023-03-31'),
        'Amount': np.random.randn(365).cumsum()})

    fig, ax = plt.subplots(figsize=(10,6))

    date_format = DateFormatter('%Y-%b')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.set_ylim(-20, 15)

    max_idx = np.argmax(df['Amount'])
    max_loc = df['Date'].iloc[max_idx], df['Amount'].iloc[max_idx]
    text_loc = df['Date'].iloc[max_idx+20], 10

    ax.annotate(text='Woah!', xy=(max_loc),xytext=(text_loc),arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90"))

    # Remove Spines
    ax.spines[['top', 'right']].set_visible(False)

    # Add Titles
    ax.text(
      x=df['Date'].min()-timedelta(20), 
      y=18.5, s='Analyzing Stock Prices', 
      ha='left', fontsize=18, weight='bold')
    ax.text(
      x=df['Date'].min()-timedelta(20), 
      y=16.5, s='April 2022 - Mar 2023', 
      ha='left', fontsize=15)

    ax.plot(df['Date'], df['Amount'], color='black')
```


----------



## Seaborn

### Overview

In general, Seaborn seems to be the easiest graph library to use with consistent syntax.

Seaborn allows for creating the common plots with just 3 functions [8]:

- Relplot: Used for creating relational plots
- Displot: Used for creating distributions plots
- Catplot: Used for creating categorical plots

### Import Dataset

Here we load some sample datasets.

```py
    import pandas as pd
    import pandas_datareader as pdr
    import seaborn as sns

    sns.set(style="darkgrid")

    # read sample dataset in [5] and [6]
    penguins = sns.load_dataset(name="penguins")
    taxis = sns.load_dataset(name="taxis")
    print(penguins.head())

    # read stock datasets in [7]
    start = '2020-1-1'
    end = '2021-6-30'
    source = 'yahoo'
    stocks = pd.DataFrame(columns=["Date","Close","Volume","Stock"])
    stock_list = ["AAPL", "IBM", "MSFT", "MRNA"]

    for stock in stock_list:
        df = pdr.data.DataReader(stock,
                                 start=start ,
                                 end=end,
                                 data_source=source).reset_index()
        df["Stock"] = stock
        df = df[["Date","Close","Volume","Stock"]]
        stocks = pd.concat([stocks, df], ignore_index=True)
        stocks.head()
```
### Scatter Plot

```py
    # scatter plot of the bill length and bill depth column [6]
    sns.relplot(data=penguins,
                x="bill_length_mm",
                y="bill_depth_mm",
                kind="scatter",
                height=6,
                aspect=1.4)

    # show different categories in different colors
    sns.relplot(data=df,
                x="bill_length_mm",
                y="bill_depth_mm",
                hue="sex",
                kind="scatter",
                height=6,
                aspect=1.4)
```

### Line Plot

We can use the `relplot` or `lineplot` functions of Seaborn to create line plots.

The `relplot` function is a figure-level interface for drawing relational plots including line plot and scatter plot.

```py
    sns.relplot(
      data=stocks[stocks.Stock == "AAPL"],
      x="Date", y="Close",
      kind="line",
      height=5, aspect=2
      )

    # increase font size of the axis titles and legend
    sns.set(font_scale=1.5)

    # plot all stocks
    sns.relplot(
      data=stocks,
      x="Date", y="Close", hue="Stock",
      height=5, aspect=2,
      kind="line",
      palette="cool"
      ).set(
        title="Stock Prices",
        ylabel="Closing Price",
        xlabel=None
      )

    # create line plot for each stock using row and/or col
    sns.relplot(
      data=stocks, x="Date", y="Close",
      row="Stock",
      height=3, aspect=3.5,
      kind="line"
      )
```

```py
    taxis["date"] = taxis["pickup"].astype("datetime64[ns]").dt.date
    taxis_daily = taxis.groupby(["date","payment"], as_index=False).agg(
        total_passengers = ("passengers","sum"),
        total_amount = ("total","sum")
    )
    taxis_daily.head()

    # show line plot that shows daily total passenger counts [6]
    sns.relplot(data=taxis_daily,
                x="date", y="total_passengers",
                hue="payment",
                kind="line",
                height=5,
                aspect=2)
```

### Histogram

```py
    # show histogram of daily passenger counts and total amounts [6]
    sns.displot(data=taxis_daily,
                x="total_amount",
                kind="hist",
                height=5,
                aspect=1.5,
                bins=12)
```

### Box Plot

```py
    # show box plot of the body mass of penguins and the differences based on islands [6]
    sns.catplot(data=penguins,
                x="island",
                y="body_mass_g",
                kind="box",
                height=5,
                aspect=1.5)
```

### Seaborn Tips

Here are four ways to change the font size of axis labels and title in a Seaborn plot [9].

#### Changing the Font Size in Seaborn

1. Set_theme function

<img width=600 src="https://miro.medium.com/max/2100/1*z3Bu9_mGcoVNt7ueVwz5nQ.png" />

2. Axis level functions

<img width=600 src="https://miro.medium.com/max/2100/1*HXJRNs84Wc23NnmJ9OomKA.png" />

3. Set_axis_labels function

<img width=600 src="https://miro.medium.com/max/2100/1*ELwHL4YC5ombql6yJKLdGA.png" />

4. Matplotlib functions

<img width=600 src="https://miro.medium.com/max/2100/1*WgJOamXxhwolExnNZEzl-g.png" />



## Plots for Multivariate Data Analysis using Seaborn

The article [14] discusses how to visualize data using Seaborn axes-level and figure-level plots.

Seaborn is an interface built on top of Matplotlib that uses short lines of code to create and style statistical plots from Pandas datafames. Seaborn utilizes Matplotlib, so it is best to have a basic understanding of the figure, axes, and axis objects.

Seaborn plots belong to one of two groups.

  1. Axes-level plots: These mimic Matplotlib plots and can be bundled into subplots using the `ax` parameter. They return an `axes` object and use normal Matplotlib functions to style.

  2. Figure-level plots: These provide a wrapper around axes plots and can only create meaningful and related subplots because they control the entire figure. They return either `FacetGrid`, `PairGrid`, or `JointGrid` objects and do not support the `ax` parameter. They use different styling and customization inputs.


We will use the [vehicles dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?ref=hackernoon.com&select=Car+details+v3.csv) from Kaggle that is under the Open database license. 

```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('darkgrid')
    sns.set(font_scale=1.3)
    cars = pd.read_csv('edited_cars.csv')
```

### Explore relationships between numeric columns

Numeric features contain continuous data or numbers as values.

The first two plots will be matrix plots we pass the whole dataframe to visualize all the pairwise distributions in one plot.

#### Pair plot

A **pair plot** creates a grid of scatter plots to compare the distribution of pairs of numeric variables which also features a histogram for each feature in the diagonal boxes.

```py
    sns.pairplot(cars)
```

What to look for:

- Scatter plots showing either positive linear relationships (if x increases, y increases) or negative (if x increases, y decreases).

- Histograms in the diagonal boxes that show the distribution of individual features.

In the pair plot below, the circled plots show an apparent linear relationship. The diagonal line points out the histograms for each feature, and the pair plot’s top triangle is a mirror image of the bottom.

```py
    # add a third variable that segments the scatter plots by color using the parameter
    sns.pairplot(
        data=cars, 
        aspect=.85,
        hue='transmission')
```

What to look out for:

- Clusters of different colors in the scatter plots.


#### Heat map

A **heat map** is a color-coded graphical representation of values in a grid. It’s an ideal plot to follow a pair plot because the plotted values represent the correlation coefficients of the pairs that show the measure of the linear relationships.

In short, a pair plot shows the intuitive trends of the data, while a heat map plots the actual correlation values using color.

Functions to use:

- sns.heatmap(): axes-level plot

First, we run `df.corr()` to get a table with the correlation coefficients which is also known as a correlation matrix.

```py
    cars.corr()
```

sns.heatmap() — Since the table above is not very intuitive, we create a heatmap.

```py
    sns.set(font_scale=1.15)
    plt.figure(figsize=(8,4))
    sns.heatmap(
        cars.corr(),        
        cmap='RdBu_r', 
        annot=True, 
        vmin=-1, vmax=1)
```

The cmap=’RdBu_r’ parameter sets the color scheme, `annot=True` draws the values inside the cells, and vmin and vmax ensures the color codes start at -1 to 1.


What to look out for:

- Highly correlated features. These are the dark-red and dark-blue cells. Values close to 1 mean a high positive linear relationship, while close to -1 show a high negative relationship.


#### Scatter plot

A scatter plot shows the relationship between two numeric features by using dots to visualize how these variables move together.

Functions to use:

- sns.scatterplot(): axes-level plot
- sns.relplot(kind=’line’): figure-level

Functions with regression line;

- sns.regplot(): axes-level
- sns.lmplot(): figure-level

```py
    # Two numeric columns (bivariate)
    sns.set(font_scale=1.3)

    sns.scatterplot(
        x='engine_cc', 
        y='mileage_kmpl', 
        data=cars)

    plt.xlabel(
        'Engine size in CC')
    plt.ylabel(
        'Fuel efficiency')
```

A reg plot draws a scatter plot with a regression line showing the trend of the data.

```py
    sns.regplot(
        x='engine_cc', 
        y='mileage_kmpl', 
        data=cars)

    plt.xlabel(
        'Engine size in CC')
    plt.ylabel(
        'Fuel efficiency')
```

We can further segment the scatter plot by a categorical variable using hue.

```py
    # Three columns (multivariate): two numeric and one categorical.
    sns.scatterplot(
        x='mileage_kmpl',
        y='engine_cc', 
        data=cars,
        palette='bright',
        hue='fuel')
```

A rel plot or relational plot is used to create a scatter plot using kind=’scatter’ (default), or a line plot using kind=’line’.

In our plot below, we use kind='scatter' and hue=’cat_col’ to segment by color. Note how the image below has similar results to the one above.

```py
    sns.relplot(
        x='mileage_kmpl', 
        y='engine_cc', 
        data=cars, 
        palette='bright',
        kind='scatter', 
        hue='fuel')
```

We can also create subplots of the segments column-wise using col=’cat_col’ and/or row-wise using row=’cat_col’. 

The plot below splits the data by the transmission categories into different plots.

```py
    sns.relplot(
        x='year', 
        y='selling_price', 
        data=cars, 
        kind='scatter', 
        col='transmission')
```

The lmplot is the figure-level version of a regplot that draws a scatter plot with a regression line onto a Facet grid. It does not have a kind parameter.

```py
    sns.lmplot(
        x="seats", 
        y="engine_cc", 
        data=cars,
        palette='bright',
        col="transmission", 
        hue="fuel")
```


### Binned Scatterplot

The [binned scatterplot][^binned_scatterplot] is a very powerful tool that provides a flexible and parsimonious way of visualizing and summarizing conditional means (and not only) in large datasets.

The idea is to divide the conditioning variable (age) into equally sized bins or quantiles and  plot the conditional mean of the dependent variable (sales) within each bin.

The most important choice when building a binned scatterplot is the **number of bins** which is the usual bias-variance trade-off. 

> It is strongly recommended to use the default optimal number of bins.

By picking a higher number of bins, we have more points in the graph but in the extreme case we will have a standard scatterplot (assuming the conditioning variable is continuous). 

On the other hand, by decreasing the number of bins, the plot will be more stable but in the extreme case we will have a single point representing the sample mean.

Binned scatterplots can also provide inference for the consitional means by computing confidence intervals around each data point using the `binsreg` package with the option `ci` adds confidence intervals to the estimation results. By default, the confidence intervals are not included in the plot.

> The recommended specification is ci=c(3,3) which adds confidence intervals based on cubic B-spline estimate of the regression function of interest to the binned scatter plot.

One problem with the Python version of the package is that is not very Python-ish. Therefore, we can wrap the binsreg package into a `binscatter` function that takes care of cleaning and formatting the output in a nicely readable Pandas DataFrame.

We can now proceed to estimate and visualize the binned scatterplot for age based on sales.

```py
# Estimate binsreg
df_est = binscatter(x='age', y='sales', data=df, ci=(3,3))
df_est.head()
```

The binscatter function outputs a dataset in which  we have values and confidence intervals for the outcome variable (sales) for each bin of the conditioning variable (age),.

We can now plot the estimates.

```py
# Plot binned scatterplot
sns.scatterplot(x='age', y='sales', data=df_est);
plt.errorbar('age', 'sales', yerr='ci', data=df_est, ls='', lw=2, alpha=0.2);
plt.title("Sales by firm's age")
```

The plot shows that the relationship looks extremely non-linear with a sharp increase in sales at the beginning of the lifetime of a firm followed by a plateau.

Moreover, the plot is also telling us information about the distributions of age and sales. The plot is denser on the left where the distribution of age is concentrated. Also, confidence intervals are tighter on the left where most of the conditional distribution of sales lies.

It might be important to control for other variables such as the number of products since firms that sell more products probably survive longer in the market and also make more sales.

binsreg allows us to condition the analysis on any number of variables using the w option.

```py
# Estimate binsreg
df_est = binscatter(x='age', y='sales', w=['products'], data=df, ci=(3,3))

# Plot binned scatterplot
sns.scatterplot(x='age', y='sales', data=df_est);
plt.errorbar('age', 'sales', yerr='ci', data=df_est, ls='', lw=2, alpha=0.2);
plt.title("Sales by firm's age");
```

Conditional on the number of products, the shape of the sales life-cycle changes further. After an initial increase in sales, we observe a gradual decrease over time.



#### Line Plot

A line plot comprises dots connected by a line that shows the relationship between the x and y variables. 

The x-axis usually contains time intervals and the y-axis holds a numeric variable whose changes we want to track over time.

Functions to use:

- sns.lineplot() — axes-level plot
- sns.relplot(kind=’line’) — figure-level plot

```py
    # Two columns (bivariate): numeric and time series.
    sns.lineplot(
        x="year", 
        y="selling_price",
        data=cars)
```

We split can split the lines by a categorical variable using hue.

```py
    # Three columns (multivariate): time series, numeric, and categorical column.
    sns.lineplot(
        x="year", 
        y="selling_price",
        data=cars,
        palette='bright',
        hue='fuel')
```

The results above can be obtained using sns.relplot with kind=’line’ and the hue parameter.

As mentioned earlier, a rel plot’s kind=’line’ parameter plots a line graph. We will use col=’transmission’ to create column-wise subplots for the two transmission classes.

```py
    sns.relplot(
        x="year", 
        y="selling_price",
        data=cars,
        color='blue', height=4
        kind='line',
        col='transmission')
```

#### Joint plot

A joint plot comprises three charts in one. 

- The center contains the bivariate relationship between the x and y variables. 

- The top and right-side plots show the univariate distribution of the x-axis and y-axis variables, respectively.

Functions to use:

- sns.jointplot() — figure-level plot

By default, the center plot is a scatter plot, (kind=’scatter’) while the side plots are histograms.

```py
    # Two columns (bivariate): two numeric
    sns.jointplot(
        x='max_power_bhp', 
        y='selling_price', 
        data=cars)
```

The joint plots in the image below utilize different kind parameters (‘kde’, ‘hist’, ‘hex’, or ‘reg’) as annotated in each figure.

sns.jointplot(x, y, data, hue=’cat_col’)

```py
    # Three columns (multivariate): two numeric, one categorical
    sns.jointplot(
        x='selling_price', 
        y='max_power_bhp', 
        data=cars,  
        palette='bright',
        hue='transmission')
```

### Exploring the relationships between categorical and numeric relationships

In the following charts, the x-axis will hold a categorical variable and the y-axis a numeric variable.

#### Bar plot

The bar chart uses bars of different heights to compare the distribution of a numeric variable between groups of a categorical variable.

By default, bar heights are estimated using the “mean”. The estimator parameter changes this aggregation function by using python’s inbuilt functions such as estimator=max or len, or NumPy functions like np.max and np.median.

Functions to use:

- sns.barplot() — axes-level plot
- sns.catplot(kind=’bar’) — figure-level plot

```py
    # Two columns (bivariate): numeric and categorical
    sns.barplot(
        x='fuel', 
        y='selling_price', 
        data=cars, 
        color='blue',
        # estimator=sum,
        # estimator=np.median)
```

```py
    # Three columns (multivariate): two categorical and one numeric.
    sns.barplot(
        x='fuel', 
        y='selling_price', 
        data=cars, 
        palette='bright'
        hue='transmission')
```

A catplot or categorical plot uses the kind parameter to specify what categorical plot to draw with options being ‘strip’(default), ’swarm’, ‘box’, ‘violin’, ‘boxen’, ‘point’ and ‘bar’.

The plot below uses catplot to create a similar plot to the one above.

```py
    sns.catplot(
        x='fuel', 
        y='selling_price', 
        data=cars,
        palette='bright',
        kind='bar',
        hue='transmission')
```

#### Point plot

Instead of bars like in a bar plot, a point plot draws dots to represent the mean (or another estimate) of each category group. A line then joins the dots, making it easy to compare how the y variable’s central tendency changes for the groups.

Functions to use:

- sns.pointplot() — axes-level plot
- sns.catplot(kind=’point’) — figure-level plot

```py
    # Two columns(bivariate): one categorical and one numeric
    sns.pointplot(
        x='seller_type', 
        y='mileage_kmpl', 
        data=cars)
```

When you add a third category using hue, a point plot is more informative than a bar plot because a line is drawn through each “hue” class, making it easy to compare how that class changes across the x variable’s groups.

Here, catplot is used with kind=’point’ and hue=’cat_col’. The same results can be obtained using sns.pointplot and the hue parameter.

```py
    # Three columns (multivariate): two categorical and one numeric
    sns.catplot(
        x='transmission', 
        y='selling_price', 
        data=cars, 
        palette='bright',
        kind='point', 
        hue='seller_type')
```

Here, we use the same categorical feature in the hue and col parameters.

```py
    sns.catplot(
        x='fuel', 
        y='year', 
        data=cars, 
        ci=None,  
        height=5, #default 
        aspect=.8,
        kind='point',
        hue='owner', 
        col='owner', 
        col_wrap=3)
```

### Box plot

A box plot visualizes the distribution between numeric and categorical variables by displaying the information about the quartiles.

From the plots, we can see the minimum value, median, maximum value, and outliers for every category class.

Functions to use:

- sns.boxplot() — axes-level plot
- sns.catplot(kind=’box’) — figure-level plot

```py
    # Two columns (bivariate): one categorical and one numeric
    sns.boxplot(
        x='owner', 
        y='engine_cc', 
        data=cars, 
        color='blue')

    plt.xticks(rotation=45, 
               ha='right')
```

hese results can also be recreated using sns.catplotusing kind=’box’ and hue.

```py
    # Three columns (multivariate): two categorical and one numeric
    sns.boxplot(
        x='fuel', 
        y='max_power_bhp', 
        data=cars,
        palette='bright',
        hue='transmission')
```

Use the catplot function with kind=’box’ and provide col parameter to create subplots.

```py
    sns.catplot(
        x='fuel', 
        y='max_power_bhp',
        data=cars,
        palette='bright',
        kind = 'box', 
        col='transmission')
```

#### Violin plot

In addition to the quartiles displayed by a box plot, a violin plot draws a Kernel density estimate curve that shows probabilities of observations at different areas.

#### Strip plot

A strip plot uses dots to show how a numeric variable is distributed among classes of a categorical variable. 

Think of a strip plot as a scatter plot where one axis is a categorical feature.

Functions to use:

- sns.stripplot() — axes-level plot
- sns.catplot(kind=’strip’) — figure-level plot

```py
    # Two variables (bivariate): one categorical and one numeric
    plt.figure(
        figsize=(12, 6))

    sns.stripplot(
        x='year', 
        y='km_driven', 
        data=cars, 
        linewidth=.5, 
        color='blue')

    plt.xticks(rotation=90)
```

Use the catplot function using kind=’strip’ (default) and provide the hue parameter. The argument dodge=True (default is dodge=False) can be used to separate the vertical dots by color.

```py
    # Three columns (multivariate): two categorical and one numeric
    sns.catplot(
        x='seats', 
        y='km_driven', 
        data=cars, 
        palette='bright', 
        height=3,
        aspect=2.5,
        # dodge=True,
        kind='strip',
        hue='transmission')
```

### Remarks

- For categorical plots such as bar plots and box plots, the bar direction can be re-oriented to horizontal bars by switching up the x and y variables.

- The row and col parameters of the FacetGrid figure-level objects used together can add another dimension to the subplots. However, col_wrap cannot be with the row parameter.

- The FacetGrid supports different parameters depending on the underlying plot. For example, sns.catplot(kind=’violin’) will support the split parameter while other kinds will not. More on the kind-specific options in this documentation.

- Figure-level functions also create bivariate plots. For example, sns.catplot(x=’fuel’, y=’mileage_cc’, data=cars, kind=’bar’) creates a basic bar plot.



----------



## Bokeh and Cufflinks

Bokeh: Interactive Web-Based Visualization

Bokeh is ideal for creating interactive visualizations for modern web browsers. It allows you to build elegant, interactive plots, dashboards, and data applications.

Here we create plotly and bokeh charts using the basic pandas plotting syntax.

We also discuss the pandas plotting functions.

### Import the Dataset

```py
    # Reading in the data
    df = pd.read_csv('NIFTY_data_2020.csv', parse_dates=["Date"], index_col='Date')

    # resample/aggregate the data by month-end
    df_resample = nifty_data.resample(rule = 'M').mean()
```

### Plotting using Pandas

Perhaps the most straightforward plotting technique is to use the pandas plotting functions [9].

### Plotting using Pandas-Bokeh

```py
    import pandas as pd
    import pandas_bokeh as pb

    # embedding plots in Jupyter Notebooks
    pb.output_notebook()

    # export plots as HTML
    pb.output_file(filename)
```

```py
    df.plot_bokeh(kind='line')
    df.plot_bokeh.line()  # same thing

    # scatter plot
    df.plot_bokeh.scatter(x='NIFTY FMCG index', y='NIFTY Bank index')

    # histogram
    df[['NIFTY FMCG index','NIFTY Bank index']].plot_bokeh(kind='hist', bins=30)

    # bar plot df_resample.plot_bokeh(kind='bar',figsize=(10,6))
```


### Plotting using Cufflinks

Cufflinks is an independent third-party wrapper library around Plotly that is more versatile, has more features, and has an API similar to pandas plotting [12].

```py
    import pandas as pd
    import cufflinks as cf

    # making all charts public and setting a global theme
    from IPython.display import display,HTML

    cf.set_config_file(sharing='public',theme='white',offline=True)
```

```py
    df.iplot(kind='line')

    # scatter plot
    df.iplot(kind='scatter',x='NIFTY FMCG index', y='NIFTY Bank index',mode='markers')

    # histogram
    df[['NIFTY FMCG index','NIFTY Bank index']].iplot(kind='hist', bins=30)

    # bar plot
    df_resample.iplot(kind='bar')
```


## Plotnine

plotnine is the implementation of the R package ggplot2 in Python.

[Introduction to Plotnine as the Alternative of Data Visualization Package in Python](https://towardsdatascience.com/introduction-to-plotnine-as-the-alternative-of-data-visualization-package-in-python-46011ebef7fe?gi=c5498e05addd)

[ggplot: Grammar of Graphics in Python with Plotnine](https://towardsdatascience.com/ggplot-grammar-of-graphics-in-python-with-plotnine-2e97edd4dacf)

[Why Is ggplot2 So Good For Data Visualization?](https://towardsdatascience.com/why-is-ggplot2-so-good-for-data-visualization-b38705f43f85)

[ggplot (plotnine) in action!](https://medium.com/geekculture/ggplot-in-action-7008f304bee1)



## Advanced Data Visualizations

Here are some advanced data visualizations that are good to know [16]:

- Cohort Chart
- Correlation Matrix
- Distplots



---------



## Data Visualization Packages

Here we discuss three visualization python packages [11].

### AutoViz

AutoViz is an open-source visualization package under the AutoViML package library designed to automate many data scientists’ works. Many of the projects were quick and straightforward but undoubtedly helpful, including AutoViz.

AutoViz is a one-liner code visualization package that would automatically produce data visualization.


### Missingno

missingno is a package designed to visualize your missing data.

This package provides an easy-to-use insightful one-liner code to interpret the missing data and shows the missing data relationship between features.

### Yellowbricks

Yellowbrick is a library to visualize the machine learning model process.

Yellowbrick is an open-source package to visualize and work as diagnostic tools that build on top of Scikit-Learn.

Yellowbrick was developed to help the model selection process using various visualization APIs that extended from Scikit-Learn APIs.


### Mito

Mito is a spreadsheet that lives inside JupyterLab notebooks.

Mito allows you to edit Pandas dataframes like an Excel file and generates Python code that corresponds to each of your edits.

[How to Make Basic Visualizations in Python without Coding using Mito](https://towardsdatascience.com/how-to-make-basic-visualizations-in-python-without-coding-f1da689d838e)

[Mito — Part 1: An Introduction to a Python Package Which Will Improve And Speed Up Your Analysis](https://towardsdatascience.com/mito-part-1-an-introduction-a-python-package-which-will-improve-and-speed-up-your-analysis-17d9001bbfdc?source=rss----7f60cf5620c9---4)




## Trees and Graphs

DSPlot is a Python package that draws and renders images of data structures.

[Visualize Trees and Graphs in Seconds With DSPlot](https://betterprogramming.pub/visualize-trees-and-graphs-in-seconds-with-dsplot-9112f465da8f?source=rss---p-d0b105d10f0a---4)



## References

[1]: [4 simple tips for plotting multiple graphs in Python](https://towardsdatascience.com/4-simple-tips-for-plotting-multiple-graphs-in-python-38df2112965c)

[2]: [Customizing Plots with Python Matplotlib](https://towardsdatascience.com/customizing-plots-with-python-matplotlib-bcf02691931f)

[3]: [Matplotlib Styles for Scientific Plotting](https://towardsdatascience.com/matplotlib-styles-for-scientific-plotting-d023f74515b4)

[4]: [How To Label The Values of Data Points With Matplotlib](https://towardsdatascience.com/how-to-label-the-values-plots-with-matplotlib-c9b7db0fd2e1?source=rss----7f60cf5620c9---4)

[5]: [Seaborn Data Visualization: A Complete Overview](https://medium.com/@alains/seaborn-data-visusalisation-a-complete-overview-that-will-blow-your-mind-2a6256cd065)

[6]: [1 Line of Seaborn is What Need for Data Visualization](https://sonery.medium.com/1-line-of-seaborn-is-what-need-for-data-visualization-e66e0b0f2add)

[7]: [7 Examples to Master Line Plots With Python Seaborn](https://towardsdatascience.com/7-examples-to-master-line-plots-with-python-seaborn-42d8aaa383a9?gi=9da22d442565)

[8]: [3 Seaborn Functions That Cover All Your Visualization Tasks](https://towardsdatascience.com/3-seaborn-functions-that-cover-almost-all-your-visualization-tasks-793f76510ac3)

[9]: [4 Different Methods for Changing the Font Size in Python Seaborn](https://sonery.medium.com/4-different-methods-for-changing-the-font-size-in-python-seaborn-fd5600592242)

[10]: [Get Interactive Plots Directly With Pandas](https://www.kdnuggets.com/get-interactive-plots-directly-with-pandas.html/)

[11]: [The Easiest Way to Make Beautiful Interactive Visualizations With Pandas using Cufflinks](https://towardsdatascience.com/the-easiest-way-to-make-beautiful-interactive-visualizations-with-pandas-cdf6d5e91757)

[12]: [Top 3 Visualization Python Packages to Help Your Data Science Activities](https://towardsdatascience.com/top-3-visualization-python-packages-to-help-your-data-science-activities-168e22178e53)

[13]: [How to Use Colormaps with Matplotlib to Create Colorful Plots in Python](https://betterprogramming.pub/how-to-use-colormaps-with-matplotlib-to-create-colorful-plots-in-python-969b5a892f0c)

[14]: [10 Must-know Seaborn Visualization Plots for Multivariate Data Analysis in Python](https://towardsdatascience.com/10-must-know-seaborn-functions-for-multivariate-data-analysis-in-python-7ba94847b117)

[15]: [Visualizing Multidimensional Categorical Data using Plotly](https://towardsdatascience.com/visualizing-multidimensional-categorical-data-using-plotly-bfb521bc806f)

[16]: [Five Advanced Data Visualizations All Data Scientists Should Know](https://towardsdatascience.com/five-advanced-data-visualizations-all-data-scientists-should-know-e042d5e1f532)

[17]: [3 Matplotlib Tips You Need to Know](https://towardsdatascience.com/3-matplotlib-tips-you-need-to-know-1b24e41552d5)

[18]: [Super Flexible Matplotlib Structure for Subplots](https://pub.towardsai.net/super-flexible-matplotlib-structure-for-subplots-d26b005252f1)

[19]: [Creating Beautiful Histograms with Seaborn](https://www.kdnuggets.com/2023/01/creating-beautiful-histograms-seaborn.html)

[20]: [4 Easy Ways to Instantly Improve Your Data Visualisations](https://towardsdatascience.com/4-easy-ways-to-instantly-improve-your-data-visualisations-2a5fc3a22182)

[21]: [Upgrade Your Data Visualisations: 4 Python Libraries to Enhance Your Matplotlib Charts]


[Holoviz Is Simplifying Data Visualization in Python](https://towardsdatascience.com/holoviz-is-simplifying-data-visualization-in-python-d51ca89739cf)

[Four Useful Seaborn Visualization Templates](https://towardsdatascience.com/grab-and-use-4-useful-seaborn-visualization-templates-6e5f11a210c9#45f6)

----------

[^binned_scatterplot]: https://towardsdatascience.com/goodbye-scatterplot-welcome-binned-scatterplot-a928f67413e4 "The Binned Scatterplot"
