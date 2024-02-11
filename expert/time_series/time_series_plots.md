# Time Series Plots


## Time Series Data Visualization in Python

Visualization plays an important role in time series analysis and forecasting [1][2].

Plots of the raw sample data can provide valuable diagnostics to identify temporal structures such as trends, cycles, and seasonality that can influence the choice of model.

```py
    from pandas import read_csv
    from matplotlib import pyplot
    
    series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
    series.plot()
    pyplot.show()
```
    
The tutorial in [2] discusses six different types of visualizations that we can use on fo time series data:

1. Line Plots

2. Histograms and Density Plots: visualize the distribution of observations themselves.

3. Box and Whisker Plots: visualize the distribution of values by time interval.

4. Heat Maps: compare observations between intervals using a heat map (similar to box and whisker).

5. Lag Plots or Scatter Plots: explore the relationship between each observation and a lag of that observation.

6. Autocorrelation Plots: quantify the strength and type of relationship between observations and their lags.

The focus is on univariate time series, but the techniques are just as applicable to multivariate time series when you have more than one observation at each time step.

----------

Here are some additional techniques for visualizing time series data in Python [1].

### Line Plots

```py
    # Create time-series line plots
    df = pd.read_csv('discoveries.csv', parse_dates=['date'], index_col='date')

    plt.style.use('fivethirtyeight')
    df.plot(figsize=(10,10))
    plt.show()

    # show all of the available styles
    print(plt.style.available)

    # change the color of the plot using the color parameter
    ax = df.plot(color='blue')

```

It is crucial that each of your plots is carefully annotated with axis labels and legends.

The `.plot()` method in pandas returns a matplotlib AxesSubplot object, and it is common practice to assign this returned object to a variable called `ax`.

```py
    ax = df.plot(color='blue', figsize=(10,10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of great discoveries')
    ax.set_title('Number of great inventions and scientific discoveries from 1860 to 1959')
    plt.show()
```

```py
    # We can slice the data using strings that represent the period in which we are interested.
    df_subset = df['1860':'1870']
    ax = df_subset.plot(color='blue', fontsize=14)
    plt.show()
```

Additional annotations can also help emphasize specific observations or events in your time series which can be achieved with `matplotlib`.

```py
    # adding markers
    ax = df.plot(color='blue', figsize=(12,10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of great discoveries')
    ax.axvline('1920-01-01', color='red', linestyle='--')
    ax.axhline(4, color='green', linestyle='--')
```

We can also highlight regions of interest to your time series plot which can help provide more context around the data and really emphasize the story we are trying to convey with the graph.

```py
    # Highlighting regions of interest
    ax = df.plot(color='blue', figsize=(15,10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of great discoveries')
    ax.axvspan('1890-01-01', '1910-01-01', color='red', alpha=0.3)
    ax.axhspan(8, 6, color='green', alpha=0.3)
```


### Summary Statistics and Diagnostics

Here we discuss how to gain a deeper understanding of your time-series data by computing summary statistics and plotting aggregated views of your data.

The number of missing values is 59 rows. 

To replace the missing values in the data we can use different options such as the mean value, value from the preceding time point, or the value from time points that are coming after. 

In order to replace missing values in your time series data, we can use the `.fillna()` method in pandas. It is important to notice the argument which specifies how we want to deal with our missing data. 

Using the method `bfill` (backfilling) will ensure that missing values are replaced by the next valid observation. On the other hand, `ffill` (forward- filling) will replace the missing values with the most recent non-missing value. 

Here, we use the `bfill` method.

```py
    # count missing values 
    print(co2_levels.isnull().sum())

    # Replacing missing values in a DataFrame
    co2_levels = co2_levels.fillna(method='bfill')
```

### Plot aggregates of the data

A moving average or rolling mean is a commonly used technique in the field of time series analysis. It can be used to smooth out short-term fluctuations, remove outliers, and highlight long-term trends or cycles. 

Taking the rolling mean of your time series is equivalent to “smoothing” your time series data. 

In pandas, the `.rolling()` method allows you to specify the number of data points to use when computing your metrics.

Here, you specify a sliding window of 52 points and compute the mean of those 52 points as the window moves along the date axis. The number of points to use when computing moving averages depends on the application, and these parameters are usually set through trial and error or according to some seasonality. 

For example, we could take the rolling mean of daily data and specify a window of 7 to obtain weekly moving averages. 

In our case, we are working with weekly data so we specified a window of 52 (because there are 52 weeks in a year) in order to capture the yearly rolling mean. 

```py
    # The moving average model
    # The rolling mean of a window of 52 is applied to the data
    co2_levels_mean = co2_levels.rolling(window=52).mean()
    ax = co2_levels_mean.plot(figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("The values of my Y axis")
    ax.set_title("52 weeks rolling mean of my time series")
    plt.show()
```

Another useful technique to visualize time series data is to take aggregates of the values in your data. 

For example, the co2_levels data contains weekly data but we may wish to see how these values behave by month of the year. 

Because we set the index of your co2_levels DataFrame as a DateTime type, it is possible to directly extract the day, month, or year of each date in the index. 

For example, you can extract the month using the command co2_levels `.index.month`. Similarly, we can extract the year using the command co2_levels `.index.year`.

Aggregating values in a time series can help answer questions such as “what is the mean value of our time series on Sundays”, or “what is the mean value of our time series during each month of the year”. 

If the index of the pandas DataFrame consists of DateTime types, we can extract the indices and group your data by these values. Here, you use the `.groupby()` and `.mean()` methods to compute the monthly and yearly averages of the CO2 levels data and assign that to a new variable called co2_levels_by_month and co2_levels_by_year. 

The .groupby() method allows us to group records into buckets based on a set of defined categories. In this case, the categories are the different months of the year and for each year.

```py
    # Plot aggregate values of your time series (co2_levels_by_month)
    index_month = co2_levels.index.month
    co2_levels_by_month = co2_levels.groupby(index_month).mean()
    co2_levels_by_month.plot(figsize=(12,10))
    plt.show()

    # Plott co2_levels_by_year
    index_year = co2_levels.index.year
    co2_levels_by_year = co2_levels.groupby(index_year).mean()
    co2_levels_by_year.plot(figsize=(12,10))
    plt.title('Yearly aggregation of the co2 level time series')
    plt.show()
```

### Box Plot

An important step to understanding the data is to create plots of the summary statistics of the time series.

There are three fundamentals plots to visualize the summary statistics of the data: the box plot, histogram plot, and density plot.

A boxplot provides information on the shape, variability, and median of your data which is  useful to display the range of your data and for identifying any potential outliers.

The lines extending parallel from the boxes are commonly referred to as _whiskers_ which are used to indicate variability outside the upper (which is the 75% percentile) and lower (which is the 25% percentile) quartiles (outliers_. The outliers are usually plotted as individual dots that are in line with whiskers.

```py
    # Summarize the data with boxplots
    ax1 = co2_levels.boxplot(figsize=(12,10))
    ax1.set_ylabel('Co2 levels')
    ax1.set_title('Boxplot for the co2 levels data')
```

### Histogram

Histograms are a type of plot that allows us to inspect the underlying distribution of the data. 

Histograms visualize the frequency of occurrence of each value in the data. 

In pandas, it is possible to produce a histogram by simply using the standard .plot() method and specifying the kind argument as hist. 

```py
    # Summarize the data with histograms
    ax2 = co2_levels.plot(kind='hist', bins=100, figsize=(15,10))
    ax2.set_xlabel('Co2 levels value')
    ax2.set_ylabel('Frequency of values in the co2 levels data')
    ax2.set_title('Histogram of the co2 levels data 100 bins')
    plt.show()
```

### Kernel Density Plot

Since it can be difficult to identify the optimal number of bins, histograms can be a cumbersome way to assess the distribution of your data. Instead, we can make use of kernel density plots to view the distribution of your data. 

Kernel density plots are a variation of histograms which use kernel smoothing to plot the values of your data and allow for smoother distributions by dampening the effect of noise and outliers while displaying where the mass of your data is located.

It is simple to generate density plots with the pandas library since we only need to use the standard `.plot()` method while specifying the kind argument as **density**.

```py
    # Summarize the data with density plots
    ax3 = co2_levels.plot(kind='density', linewidth=2, figsize=(15,10))
    ax3.set_xlabel('co2 levels data values')
    ax3.set_ylabel('Density values of the co2 levels data')
    ax3.set_title('Density plot of the co2 levels data')
```

### Seasonality, Trend, and Noise

Now we go beyond summary statistics by learning about autocorrelation and partial autocorrelation plots. We also discuss how to automatically detect seasonality, trend, and noise in your time series data. 

The autocorrelation and partial autocorrelation were covered in more detail in the previous article of this series.


**Autocorrelation** is a measure of the correlation between your time series and a delayed copy of itself. 

For example, an autocorrelation of order 3 returns the correlation between a time series at points t(1), t(2), t(3), and its own values lagged by 3 time points, t(4), t(5), t(6). 

Autocorrelation is used to find repeating patterns or periodic signals in time series data. 

The principle of autocorrelation can be applied to any signal, not just time series. Therefore, it is common to encounter the same principle in other fields where it is also sometimes referred to as autocovariance.

```py
    # Plot the autocorrelation of the co2 level time series 
    import matplotlib.pyplot as plt
    from statsmodels.graphics import tsaplots

    fig = tsaplots.plot_acf(co2_levels['co2'], lags=40)
    plt.show()
```

Since autocorrelation is a correlation measure, the autocorrelation coefficient can only take values between -1 and 1. 

An autocorrelation of 0 indicates no correlation while 1 and -1 indicate strong negative and positive correlations. 

In order to help assess the significance of autocorrelation values, the `.plot_acf()` function also computes and returns margins of uncertainty which are represented in the graph as blue shaded regions. 

Values above these regions can be interpreted as the time series having a statistically significant relationship with a lagged version of itself.

Going beyond autocorrelation, _partial autocorrelation_ measures the correlation coefficient between a time series and lagged versions of itself. However, it extends this idea by also removing the effect of previous time points.

```py
    # Plot the partial autocorrelations for the first 40 lags of the co2 level time series
    import matplotlib.pyplot as plt
    from statsmodels.graphics import tsaplots

    fig = tsaplots.plot_pacf(co2_levels['co2'], lags=40)
    plt.show()
```

If partial autocorrelation values are close to 0, we can conclude that values are not correlated with one another. Inversely, partial autocorrelations that have values close to 1 or -1 indicate that there exist strong positive or negative correlations between the lagged observations of the time series. 

If partial autocorrelation values are beyond the margins of uncertainty which are marked by the blue-shaded regions, we can assume that the observed partial autocorrelation values are statistically significant.


When looking at time-series data, we may notice some clear patterns that they exhibit. As we can see in the co2 levels time series shown below, the data displays a clear upward trend as well as a periodic signal.


In general, most time series can be decomposed in three major components: seasonality, trend, and noise.

- The **seasonality** describes the periodic signal in your time series.
- The **trend** describes whether the time series is decreasing, constant, or increasing over time. 
- The **noise** describes the unexplained variance and volatility of your time series.

```py
    # Time series decomposition 
    rcParams['figure.figsize'] = 11, 15  # resizing the image to be big enough for us 
    
    decomposition = sm.tsa.seasonal_decompose(co2_levels['co2'])
    fig = decomposition.plot()
    plt.show()
```

It is easy to extract each individual component and plot them. 

Here, we use the `dir()` command to print out the attributes associated with the decomposition variable generated before and to print the seasonal component we use the `decomposition.seasonal` command.

```py
    # Plot only easonality component in time series
    decomp_seasonal = decomposition.seasonal
    ax = decomp_seasonal.plot(figsize=(14, 2))
    ax.set_xlabel('Date')
    ax.set_ylabel('Seasonality of time series')
    ax.set_title('Seasonal values of the time series')
    plt.show()
```

A seasonal pattern exists when a time series is influenced by seasonal factors. 

Seasonality should always be a fixed and known period. 

For example, the temperature of the day should display clear daily seasonality, as it is always warmer during the day than at night. Alternatively, it could also display monthly seasonality, since it is always warmer in summer compared to winter.

Now we can extract the trend values of the time series decomposition. The trend component reflects the overall progression of the time series and can be extracted using the decomposition `.trend` command.

```py
    # Trend component in time series
    decomp_trend = decomposition.trend
    ax = decomp_trend.plot(figsize=(14, 2))
    ax.set_xlabel('Date')
    ax.set_ylabel('Trend of time series')
    ax.set_title('Trend values of the time series')
    plt.show()
```

Finally, we can extract the noise or the residual component of a time series. The residual component describes random, irregular influences that could not be attributed to either trend or seasonality.

```py
    # Plot the noise component in time series
    decomp_resid = decomposition.resid
    ax = decomp_resid.plot(figsize=(14, 2))
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual of time series')
    ax.set_title('Residual values of the time series')
    plt.show()
```

### Analyzing airline data

We will hone our skills with the famous [airline dataset](https://github.com/youssefHosni/Time-Series-With-Python/tree/main/Time%20Series%20Data%20Visualization) which consists of monthly totals of airline passengers from January 1949 to December 1960. The dataset contains 144 data points and is often used as a standard dataset for time series analysis.

```py
    # Load the airline data
    airline = pd.read_csv('airline_passengers.csv', parse_dates=['Month'], index_col='Month')

    # Plot the time series in your DataFrame
    ax = airline.plot(color='blue', fontsize=12, figsize=(12,10))

    # Add a red vertical line at the date 1955-12-01
    ax.axvline('1955-12-01', color='red', linestyle='--')

    # Specify the labels in your plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Monthly Airline Passengers', fontsize=12)
    ax.set_title('Number of Monthly Airline Passengers')
    plt.show()
```

Print the summary of the data and the number of missing values. 

```py
    # Print out the number of missing values
    print(airline.isnull().sum())

    # Print out summary statistics of the airline DataFrame
    print(airline.describe())
```

Plot the box plot of the time series data.

```py
    # Display boxplot of airline values
    ax = airline.boxplot()

    # Specify the title of your plot
    ax.set_title('Boxplot of Monthly Airline\nPassengers Count', fontsize=20)
    plt.show()
```

Create and plot the monthly aggregation of the airline passenger data.

```py
    # Get month for each dates from the index of airline
    index_month = airline.index.month

    # Compute the mean number of passengers for each month of the year
    mean_airline_by_month = airline.groupby(index_month).mean()

    # Plot the mean number of passengers for each month of the year
    mean_airline_by_month.plot()
    plt.legend(fontsize=20)
    plt.show()
```

Finally, decompose the time series and plot the trend and seasonality in the data.

```py
    import statsmodels.api as sm

    # Perform time series decompositon
    decomposition = sm.tsa.seasonal_decompose(airline)

    # Extract the trend and seasonal components
    trend = decomposition.trend
    seasonal = decomposition.seasonal

    airline_decomposed = pd.DataFrame({'Trend':trend, 'Seasonal':seasonal})

    # Print the first 5 rows of airline_decomposed
    print(airline_decomposed.head(5))

    # Plot the values of the airline_decomposed DataFrame
    ax = airline_decomposed.plot(figsize=(12, 6), fontsize=15)

    # Specify axis labels
    ax.set_xlabel('Date', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
```

The trend shows that the number of passengers is increasing over the years 1949 to 1959, which is reasonable as the number of airplanes itself increased. There is also seasonality in the data which is expected as shown in the monthly aggregation plot.

### Visualizing Multiple Time Series

It is common to be involved in projects where multiple time series need to be studied simultaneously. 

Here, we discuss you how to plot multiple time series at once and how to discover and describe relationships between multiple time series. We will be working with a new dataset that contains volumes of different types of meats produced in the United States between 1944 and 2012. The dataset can be downloaded from [here](https://github.com/youssefHosni/Time-Series-With-Python/tree/main/Time%20Series%20Data%20Visualization).


A convenient aspect of pandas is that dealing with multiple time series is very similar to dealing with a single time series. As usual, we can leverage the `.plot()` and `.describe()` methods to visualize and produce statistical summaries of the data.

```py
    meat = pd.read_csv("meat.csv", parse_dates=['date'], index_col='date')
    print(meat.head(5))

    # Plot multiple time series
    ax = meat.plot(figsize=(15, 10), fontsize=14)
    plt.show()
```

Another interesting way to plot multiple time series is to use area charts. Area charts are commonly used when dealing with multiple time series and can be leveraged to represent cumulated totals. With the pandas library, you can simply leverage the .area() method as shown on this slide to produce an area chart.

```py
    # Area charts
    ax = meat.plot.area(figsize=(15, 10), fontsize=14)
```

#### Plot multiple time series

When plotting multiple time series, matplotlib will iterate through its default color scheme until all columns in the DataFrame have been plotted. Therefore, the repetition of the default colors may make it difficult to distinguish some of the time series. 

For example, since there are seven time series in the meat dataset, some time series are assigned the same blue color. In addition, matplotlib does not consider the color of the background, which can also be an issue.

To remedy this, the .`plot()` method has an additional argument called **colormap** which allows us to assign a wide range of color palettes with varying contrasts and intensities. 

We can either define your own Matplotlib colormap or use a string that matches a colormap registered with matplotlib. Here, we use the `Dark2` color palette.

```py
    ax = meat.plot(colormap='Dark2', figsize=(14, 7))
    ax.set_xlabel('Date')
    ax.set_ylabel('Production Volume (in tons)')
    plt.show()
```

#### Subplots

To overcome issues with visualizing datasets containing time series of different scales, we can leverage the subplots argument which will plot each column of a DataFrame on a different subplot. In addition, the layout of the subplots can be specified using the `layout` keyword,which accepts two integers specifying the number of rows and columns to use. 

It is important to ensure that the total number of subplots is greater than or equal to the number of time series in the DataFrame. 

We can also specify if each subgraph should share the values of their x-axis and y-axis using the `sharex` and `sharey` arguments. 

Finally, we need to specify the total size of the graph (which will contain all subgraphs) using the `figsize` argument.

```py
    # Facet plots
    meat.plot(subplots=True,
            linewidth=0.5,
            layout=(2, 4),
            figsize=(16, 12),
            sharex=False,
            sharey=False)
    plt.show()
```

#### Visualizing the relationships between multiple time series

One of the most widely used methods to assess the similarities between a group of time series is by using the correlation coefficient. 

The **correlation coefficient** is a measure used to determine the strength or lack of relationship between two variables. 

The standard way to compute correlation coefficients is by using Pearson’s coefficient which should be used when we think that the **relationship between the variables of interest is linear**. Otherwise, we can use the Kendall Tau or Spearman rank coefficient methods when the relationship between your variables of interest is thought to be non-linear. 

In Python, we can quickly compute the correlation coefficient between two variables by using the `pearsonr`, `spearmanr`, or `kendalltau` functions in the s`cipy.stats.stats` module. 

All three of these correlation measures return both the correlation and p-value between the two variables x and y.

If we want to investigate the dependence between multiple variables at the same time, we need to compute a correlation matrix which is a table containing the correlation coefficients between each pair of variables. 

- Correlation coefficients can take any values between -1 and 1. 

- A correlation of 0 indicates no correlation while 1 and -1 indicate strong positive and negative correlations.

```py
    # Compute the correlation matrices
    from scipy.stats.stats import pearsonr
    from scipy.stats.stats import spearmanr
    from scipy.stats.stats import kendalltau
    
    corr_p = meat[['beef', 'veal','turkey']].corr(method='pearson')
    print('Pearson correlation matrix')
    print(corr_p)
    
    corr_s = meat[['beef', 'veal','turkey']].corr(method='spearman')
    print('Spearman correlation matrix')
    print(corr_s)
```

Once we have stored your correlation matrix in a new DataFrame, it may be easier to visualize it instead of trying to interpret several correlation coefficients at once using a **heatmap** of the correlation matrix.

```py
    import seaborn as sns
    
    corr_mat = meat.corr(method='pearson')
    sns.heatmap(corr_mat)

    # Clustermap
    sns.clustermap(corr_mat)
```

The heatmap is a useful tool to visualize correlation matrices, but the lack of order can make it difficult to read or even identify which groups of time series are the most similar. Therefore, it is recommended to use the `.clustermap()` function in the seaborn library which applies hierarchical clustering to your correlation matrix to plot a sorted heatmap where similar time series are placed closer to one another.


### Case Study: Unemployment Rate

In this section, we practice all the concepts covered in the course. 

We will visualize the [unemployment rate](https://github.com/youssefHosni/Time-Series-With-Python/tree/main/Time%20Series%20Data%20Visualization) in the US from 2000 to 2010. The jobs dataset contains time series for 16 industries across a total of 122 time points one per month for 10 years.


----------


## Multiple Time-Series Plot

The article [3] discusses techniques for plotting multiple time-series data. 

Here, we use the [Air Pollution in Seoul](http://www.kaggle.com/datasets/bappekim/air-pollution-in-seoul) dataset from Kaggle. The data was provided by the Seoul Metropolitan Government. It is about air pollution information which consists of SO2, NO2, CO, O3, PM10, and PM2.5 between 2017 and 2019 from 25 districts in Seoul, the capital city of South Korea.

Here, PM2.5 from 25 districts will be the primary variable plotted as multiple time-series lines. PM2.5 is defined as a fine particle matter with a diameter smaller than 2.5 µm. It is considered a type of pollution that causes short-term health effects.

Visualizing PM2.5 from many locations helps compare how pollution affects the city.

```py
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    # load the dataset
    df = pd.read_csv('<file locaion>/Measurement_summary.csv')
    print(df.head())

    # check data types and missing values
    print(df.info())

    # check number of distinct station codes
    station_code = df['Station code'].nunique()
    print(station_code)

    # check the station codes
    list_scode = list(set(df['Station code']))
    print(list_scode)
```

From 101 to 125, the Station codes represent the districts in Seoul. Personally, using the district names is more convenient for labeling the visualization since it is more convenient to read.

 The names will be exacted from the ‘Address’ column to create the ‘District’ column.

```py
    list_add = list(df['Address'])
    District = [i.split(', ')[2] for i in list_add]
    df['District'] = District

    # create list with the 25 district names for use later
    list_district = list(set(District))
```

Prepare another three columns, YM(Year-Month), Year, and Month to apply with some graphs. For easier visualizing, we will group them into average monthly DataFrame.

```py
    list_YM = [i.split(" ")[0][:-3] for i in  list(df['Measurement date'])]
    list_Year = [i.split(" ")[0][0:4] for i in  list(df['Measurement date'])]
    list_Month = [i.split(" ")[0][5:7] for i in  list(df['Measurement date'])]

    df['YM'] = list_YM
    df['Year'] = list_Year
    df['Month'] = list_Month

    # create a monthly dataframe
    df_monthly = df.groupby(['Station code', 'District', 'YM', 'Year', 'Month']).mean()
    df_monthly = df_monthly[['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']].reset_index()

    df_monthly.head()
```

### Plot the multiple time-series data

The visualizations recommended in this article are mainly for coping with the overlapping plots since it is a main problem in plotting multiple time-series data, as we have already seen.

Each graph has its pros and cons. Some may be just for an eye-catching effect, but all of them have the same purpose: comparing sequences between categories.

```py
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(14,8)})

    ax = sns.lineplot(data=df_monthly, x ='YM', y = 'PM2.5',
                      hue='District', palette='viridis',
                      legend='full', lw=3)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel('PM2.5 (µg/m3)')
    plt.xlabel('Year-Month')
    plt.show()
```

### 1. Changing nothing but making the plot interactive.

Plotly is a graphing library for making interactive graphs. The interactive chart helps zoom in on the area with overlapping lines.

### 2. Comparing one by one with Small Multiple Time Series.

With the Seaborn library, we can plot small multiple time series. The idea is to plot each line one by one while still comparing them with the silhouette of the other lines. 

```py
    g = sns.relplot(data = df_monthly, x = "YM", y = "PM2.5",
                    col = "District", hue = "District",
                    kind = "line", palette = "Spectral",   
                    linewidth = 4, zorder = 5,
                    col_wrap = 5, height = 3, aspect = 1.5, legend = False
                   )

    #add text and silhouettes
    for time, ax in g.axes_dict.items():
        ax.text(.1, .85, time,
                transform = ax.transAxes, fontweight="bold"
               )
        sns.lineplot(data = df_monthly, x = "YM", y = "PM2.5", units="District",
                     estimator = None, color= ".7", linewidth=1, ax=ax
                    )

    ax.set_xticks('')
    g.set_titles("")
    g.set_axis_labels("", "PM2.5")
    g.tight_layout()
```

### 3. Changing the point of view with Facet Grid

FacetGrid from Seaborn can be used to make multi-plot grids. 

In this case, the ‘Month’ and ‘Year’ attributes are set as rows and columns, respectively. 

The values can also be simultaneously compared monthly in vertical and yearly in horizontal.

```py
    g = sns.FacetGrid(df_monthly, col="Year", row="Month", height=4.2, aspect=1.9)
    g = g.map(sns.barplot, 'District', 'PM2.5', palette='viridis', ci=None, order = list_district)

    g.set_xticklabels(rotation = 90)
    plt.show()
```

### 4. Using color with Heatmap

A heatmap represents the data into a two-dimensional chart showing values in colors. 

To deal with the Time Series data, we can set the groups on the vertical and the timeline on the horizontal dimensions. The difference in color helps distinguish between groups.

```py
    df_pivot = pd.pivot_table(df_monthly,
                              values='PM2.5',
                              index='District',
                              columns='YM')
    print(df_pivot)

    plt.figure(figsize = (40,19))
    plt.title('Average Monthly PM2.5 (mircrogram/m3)')

    sns.heatmap(df_pivot, annot=True, cmap='RdYlBu_r', fmt= '.4g',)
    plt.xlabel('Year-Month')
    plt.ylabel('District')
    plt.show()
```

### 5. Applying angles with a Radar chart

We can set the angular axis on the scatter plot in Plotly to create an interactive Radar Chart. Each month will be selected as a variable on the circle.

For example, in this article, we will create a radar chart comparing the average monthly PM2.5 of the 25 districts in 2019.

### 6. Fancy the bar plot with Circular Bar Plot (Race Track Plot)

The concept of a Circular Bar Plot (aka Race Track Plot) is so simple because it is just bar plots in a circle. We can plot Circular Bar Plot monthly and then make a photo collage to compare the process along the time.

The picture below shows an example of a Circular Bar Plot we are going to create. The disadvantage of this chart is that it is hard to compare between categories. By the way, it is a good choice for getting attention with an eye-catching effect.

### 7. Starting from the center with Radial Plot

Like Circular Bar Plot, Radial Plot is based on bar charts that use polar coordinates instead of cartesian coordinates. This chart type is inconvenient when comparing categories located far away from each other, but it is an excellent choice to get attention. It can be used in Infographics.

The picture below shows an example of Radial plots showing the average PM2.5 from the 25 districts in January 2019.

### 8. Showing densities with Overlapping densities (Ridge plot)

Overlapping densities (Ridge plot) can be used with multiple time-series data by setting an axis as a timeline. Likes Circular Bar Plot and Radial Plot, the Ridge plot can get people´s attention. The code on the official Seaborn website is here.

The following picture shows an example of the Ridge plot with the densities of PM2.5 in a district in 2019.


----------



## References

[1] [Time Series Data Visualization in Python](https://pub.towardsai.net/time-series-data-visualization-in-python-2b1959726312)

[2] [Time Series Data Visualization with Python](https://machinelearningmastery.com/time-series-data-visualization-with-python/)

[3] [8 Visualizations with Python to Handle Multiple Time-Series Data](https://towardsdatascience.com/8-visualizations-with-python-to-handle-multiple-time-series-data-19b5b2e66dd0)

