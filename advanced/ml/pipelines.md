# Data Pipelines

## Overview

There are multiple stages to running machine learning algorithms since it involves a sequence of tasks including pre-processing, feature extraction, model fitting, performance, and validation.

**Pipeline** is a technique used to create a linear sequence of data preparation and modeling steps to automate machine learning workflows [5].

Pipelines help in parallelization which means different jobs can be run in parallel as well as help to inspect and debug the data flow in the model.

A data pipeline has six basic elements [7]:

1. Data Retrieval
2. Data Preparation
3. Model Training
4. Model Evaluation and Tuning
5. Model Deployment
6. Monitoring


Here we are using the `Pipeline` class from scikit-learn [5]. 


## Import Libraries

```py
    import pandas as pd
    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    from sklearn.decomposition import PCA
```

```py
    # Load dataset
    data_df=pd.read_csv("Path to CSV file")
    
    # Drop null values
    data_df.dropna()
    
    data_df.head()
    
    # Calculate the value counts for each category
    data_df['R'].value_counts()
    
    # Find the unique values
    data_df['R'].unique()
    
    # Split the data into training and testing set
    x = data_df.drop(['R'], axis=1)
    y = data_df['R']
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
```

## Create Simple Pipeline

```py
    # Create a pipeline
    pipeline_lr = Pipeline([
        ('mms', MinMaxScaler()),
        ('lr', LogisticRegression())
    ])
    
    # Fit pipeline
    pipeline_lr.fit(trainX, trainY)
    
    # Evaluate pipeline
    y_predict = pipeline_lr.predict(testX)
    print('Test Accuracy Score: {:.4f}'.format(accuracy_score(testY, y_predict)))
```

## Best Scaler

```py
    # Create a pipeline
    pipeline_lr_mm = Pipeline([
        ('mms', MinMaxScaler()),
        ('lr', LogisticRegression())
        ])
    pipeline_lr_r = Pipeline([
        ('rs', RobustScaler()),
        ('lr', LogisticRegression())
        ])
    pipeline_lr_w = Pipeline([
        ('lr', LogisticRegression())
        ])
    pipeline_lr_s = Pipeline([
        ('ss', StandardScaler()),
        ('lr', LogisticRegression())
        ])

    # Create a pipeline dictionary
    pipeline_dict = {
    0: 'Logistic Regression without scaler',
        1: 'Logistic Regression with MinMaxScaler',
        2: 'Logistic Regression with RobustScaler',
        3: 'Logistic Regression with StandardScaler',
    }

    # Create a pipeline list
    pipelines = [pipeline_lr_w, pipeline_lr_mm, 
        pipeline_lr_r, 
        pipeline_lr_s]

    # Fit the pipeline
    for p in pipelines:
        p.fit(trainX, trainY)

    # Evaluate the pipeline
    for i, val in enumerate(pipelines):
    print('%s pipeline Test Accuracy Score: %.4f' % (pipeline_dict[i], accuracy_score(testY, val.predict(testX))))
```

Convert pipeline to DataFrame and show the best model:

```py
    l = []
    for i, val in enumerate(pipelines):
        l.append(accuracy_score(testY, val.predict(testX)))
    result_df = pd.DataFrame(list(pipeline_dict.items()),columns = ['Index','Estimator'])

    result_df['Test_Accuracy'] = l

    best_model_df = result_df.sort_values(by='Test_Accuracy', ascending=False)
    print(best_model_df)
```

## Best Estimator

```py
    # Create a pipeline
    pipeline_knn = Pipeline([
        ('ss1', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=4))
        ])
    pipeline_dt = Pipeline([
        ('ss2', StandardScaler()),
        ('dt', DecisionTreeClassifier())
        ])
    pipeline_rf = Pipeline([
        ('ss3', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=80))
        ])
    pipeline_lr = Pipeline([
        ('ss4', StandardScaler()),
        ('lr', LogisticRegression())
        ])
    pipeline_svm_lin = Pipeline([
        ('ss5', StandardScaler()),
        ('svm_lin', SVC(kernel='linear'))
        ])
    pipeline_svm_sig = Pipeline([
        ('ss6', StandardScaler()),
        ('svm_sig', SVC(kernel='sigmoid'))
        ])

    # Create a pipeline dictionary
    pipeline_dict = {
        0: 'knn',
        1: 'dt',
        2: 'rf',
        3: 'lr',
        4: 'svm_lin',
        5: 'svm_sig',

        }

    # Create a List
    pipelines = [pipeline_lr, pipeline_svm_lin, pipeline_svm_sig, pipeline_knn, pipeline_dt, pipeline_rf]

    # Fit the pipeline
    for p in pipelines:
        pipe.fit(trainX, trainY)

    # Evaluate the pipeline
    l = []
    for i, val in enumerate(pipelines):
        l.append(accuracy_score(testY, val.predict(testX)))

    result_df = pd.DataFrame(list(pipeline_dict.items()),columns = ['Idx','Estimator'])

    result_df['Test_Accuracy'] = l

    b_model = result_df.sort_values(by='Test_Accuracy', ascending=False)

    print(b_model)
```

## Pipeline with PCA

Pipeline example with Principal Component Analysis (PCA) [5].


## Joblib

**Joblib** is an open-source Python library that helps to save pipelines to a file that can be used later [10].


## Pandas pipe

The pandas `pipe` function offers a structured and organized way for combining several functions into a single operation [10]. 

As the number of steps increase, the syntax becomes cleaner with the pipe function compared to executing functions.

We can also apply Python design pattern to create scalable data-wrangling pipelines [10].

The tutorial [13] shows how to create a pandas pipe and add multiple chainable functions to perform data processing and visualization. 


## Kedro

[Kedro — A Python Framework for Reproducible Data Science Project](https://towardsdatascience.com/kedro-a-python-framework-for-reproducible-data-science-project-4d44977d4f04)

[How to perform anomaly detection with the Isolation Forest algorithm](https://towardsdatascience.com/how-to-perform-anomaly-detection-with-the-isolation-forest-algorithm-e8c8372520bc?gi=9b318130c70a)

[Level Up Your MLOps Journey with Kedro](https://towardsdatascience.com/level-up-your-mlops-journey-with-kedro-5f000e5d0aa0)



## More on scikit-learn pipeline

A scikit-learn pipeline is a component provided by scikit-learn package that allow us to merge different components within the scikit-learn API and run them sequentially. 

Thus, pipelines are very helpful because the component can perform all the data preprocessing and model fitting and also help to minimize human error during the data transformation and fitting process [7] [8]. 

## Data Prep

```py
    # convert question mark '?' to NaN
    df.replace('?', np.nan, inplace=True)
        
    # convert target column from string to number
    le = LabelEncoder()
    df.income = le.fit_transform(df.income)
```

## Create Pipeline

```py
    # create column transformer component
    # We will select and handle categorical and numerical features in a differently
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipe, make_column_selector(dtype_include=['int', 'float'])),
        ('categorical', categorical_pipe, make_column_selector(dtype_include=['object'])),
        ])

    # create pipeline for numerical features
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # create pipeline for categorical features
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore'))
        ])

        
    # create main pipeline
    pipe = Pipeline([
        ('column_transformer', preprocessor),
        ('model', KNeighborsClassifier())
        ])
```

## Train and Evaluate Pipeline

```py
    # create X and y variables
    X = df.drop('income', axis=1)
    y = df.income

    # split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # fit pipeline with train data and predicting test data
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    # check pipeline accuracy
    accuracy_score(y_test, predictions)
```

The main advantage of using this pipelines is that we can save them just like any other model in scikit-learn. 

The scikit-learn pipelines are estimators, so we can save them with all the preprocessing and modelling steps into a binary file using joblib and load them from the binary file later. 

```py
    import joblib

    # Save the pipeline into a binary file
    joblib.dump(pipe, 'wine_pipeline.bin')

    # Load the saved pipeline from a binary file
    pipe = joblib.load('wine_pipeline.bin')
```


----------



## Modeling Pipeline

A _pipeline_ is a linear sequence of data preparation options, modeling operations, and prediction transform operations [10].

A pipeline allows the sequence of steps to be specified, evaluated, and used as an atomic unit.

**Pipeline:** A linear sequence of data preparation and modeling steps that can be treated as an atomic unit.


The first example uses data normalization for the input variables and fits a logistic regression model:

[Input], [Normalization], [Logistic Regression], [Predictions]


The second example standardizes the input variables, applies RFE feature selection, and fits a support vector machine.

[Input], [Standardization], [RFE], [SVM], [Predictions]


A pipeline may use a data transform that configures itself automatically, such as the RFECV technique for feature selection.

- When evaluating a pipeline that uses an automatically-configured data transform, what configuration does it choose? 

- When fitting this pipeline as a final model for making predictions, what configuration did it choose?

**The answer is, it does not matter.**

We are not concerned about the specific internal structure or coefficients of the chosen model.

We can inspect and discover the coefficients used by the model as an exercise in analysis, but it does not impact the selection and use of the model.

This same answer generalizes when considering a modeling pipeline.

We are not concerned about which features may have been automatically selected by a data transform in the pipeline. 

We are also not concerned about which hyperparameters were chosen for the model when using a grid search as the final step in the modeling pipeline.

The pipeline allows us as machine learning practitioners to move up one level of abstraction and be less concerned with the specific outcomes of the algorithms and more concerned with the capability of a sequence of procedures.

It is a shift in thinking that may take some time to get used to.




## References

[5] [Build Machine Learning Pipelines](https://medium.datadriveninvestor.com/build-machine-learning-pipelines-with-code-part-1-bd3ed7152124?gi=c419327a3c8c)

[6] [A Better Way for Data Preprocessing: Pandas Pipe](https://towardsdatascience.com/a-better-way-for-data-preprocessing-pandas-pipe-a08336a012bc)

[7] [Introduction to Scikit-learn’s Pipelines](https://towardsdatascience.com/introduction-to-scikit-learns-pipelines-565cc549754a)

[8] [Unleash the Power of Scikit-learn’s Pipelines](https://towardsdatascience.com/unleash-the-power-of-scikit-learns-pipelines-b5f03f9196de)

[9] [Lightweight Pipelining In Python using Joblib](https://towardsdatascience.com/lightweight-pipelining-in-python-1c7a874794f4)


[10] [A Gentle Introduction to Machine Learning Modeling Pipelines](https://machinelearningmastery.com/machine-learning-modeling-pipelines/)

[11] [Automate Machine Learning Workflows with Pipelines in Python and scikit-learn](https://machinelearningmastery.com/automate-machine-learning-workflows-pipelines-python-scikit-learn/)

[12] [How To Use Scikit-Learn Pipelines To Simplify Machine Learning Workflow](https://medium.com/geekculture/how-to-use-sklearn-pipelines-to-simplify-machine-learning-workflow-bde1cebb9fa2)

[13] [Simplify Data Processing with Pandas Pipeline](https://www.kdnuggets.com/2022/08/simplify-data-processing-pandas-pipeline.html)

[Recursive Feature Elimination (RFE) for Feature Selection in Python](https://machinelearningmastery.com/rfe-feature-selection-in-python/)