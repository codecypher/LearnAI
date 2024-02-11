# DagsHub

## TODO

- DagsHub and dvc
- Kedro
- Mlflow

- Poetry
- pipreqs
- pyenv


## Command Quick Reference

```bash
  dvc remote add origin s3://dagshub-hello

  dvc remote list
  dvc remote default main

  dvc push
  dvc pull

  git status
  dvc status
```

```bash
  # start tracking a file or directory
  dvc add data/data.xml

  dvc list https://dagshub.com/codecypher/hello-world
  dvc list https://dagshub.com/codecypher/hello-world data


  # import file or directory
  dvc import https://github.com/iterative/dataset-registry \
             get-started/data.xml -o data/data.xml
```

The metadata file is a placeholder for the original data that can be easily versioned like source code with Git:

```bash
    git add data/data.dvc data/.gitignore
    git commit -m "Add raw data"
```

```bash
  # show which branch is selected
  git branch
  
  # create a branch and check it out in one step
  git checkout -b metrics main
  git checkout -b new_feature_branch
  
  # switch between branches
  git checkout branchname
```

----------



## Create a Project on DagsHub

This part of the Get Started section focuses on the configuration process when creating a project on DagsHub [1]. 

We will cover how to create a DagsHub repository, connect it to your local computer, configure DVC, and set DagsHub storage as remote storage.

```bash
  dvc init
  git status

  # Configure DagsHub as DVC Remote Storage
  dvc remote add origin https://dagshub.com/codecypher/hello-world.dvc
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user codecypher
  dvc remote modify origin --local password 8414ba848e08cbc1c3943a7c0dc0122b9afd7774

  # checkpoint
  cat .dvc/config.local
```

For more information about DagsHub storage, visit the reference page.

If you still want to set up your own cloud remote storage, please refer to our [setup external remote storage](https://dagshub.com/docs/integration_guide/set_up_remote_storage_for_data_and_models/) page.

```bash
  # Version and push DVC Configurations
  git status -s
  git add .dvc .dvcignore .gitignore
  git commit -m "Initialize DVC"
  git push
```

## Version Code and Data

In the previous part of the Get Started section, we created and configured a DagsHub repository. 

In this part, we will download and add a project to our local directory, track the files using DVC and Git, and push the files to the remotes.

```bash
  # Fork and clone the hello-world repository.
  git clone -b start-version-project https://dagshub.com/codecypher/hello-world.git

  # Configure DVC locally and set DagsHub as remote storage
  dvc remote add origin https://dagshub.com/codecypher/hello-world.dvc
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user codecypher
  dvc remote modify origin --local password 8414ba848e08cbc1c3943a7c0dc0122b9afd7774

  # checkpoint
  cat .dvc/config.local
```


```bash
  # Add a Project¶
  dvc get https://dagshub.com/nirbarazida/hello-world-files requirements.txt
  dvc get https://dagshub.com/nirbarazida/hello-world-files src
  dvc get https://dagshub.com/nirbarazida/hello-world-files data/

  # Install Requirements
  pip3 install -r requirements.txt

  # checkpoint
  git status -s
```

### Track Files Using Git and DVC

At this point, we need to decide which files will be tracked by Git and which will be tracked by DVC. 

We will start with files tracked by DVC since this action will generate new files tracked by Git.

```bash
  # Track Files with DVC
  dvc add data

  # Track the changes with Git
  git add data.dvc .gitignore
  git commit -m "Add the data directory to DVC tracking"
```

```bash
  # Track Files with Git
  git status -s

  git add requirements.txt src/
  git commit -m "Add requirements and src to Git tracking"
```

### Push the Files to the Remotes

```bash
  # Push DVC tracked files
  dvc push -r origin

  # Push Git tracked files
  git push
```

### Process and Track Data Changes

Now, we would like to preprocess our data and track the results using DVC.

We will run the `data_preprocessing.py` file from our CLI.

```bash
  python3 src/data_preprocessing.py
  tree data

  # checkpoint
  git status
  dvc status
```

```bash
  # version the new status of the data directory with DVC
  dvc add data
  git add data.dvc
  git commit -m "Process raw-data and save it to data directory"

  # Push changes to remote
  dvc push -r origin
  git push
```

In this section, we covered the basic workflow of DVC and Git:

- We added the project files to the repository and tracked them using Git and DVC.
- We generated preprocessed data files and learned how to add these changes to DVC. 



## Track Experiments

In the previous part of the Get Started section, we learned how to track and push files to DagsHub using Git and DVC. 

This part covers how to track your Data Science Experiments and save their parameters and metrics. 

We assume you have a project that you want to add experiment tracking to.

We will be showing an example based on the result of the last section, but you can adapt it to your project in a straightforward way.

```bash
  # create branch
  git branch start-track-experiments

  # switch branches
  git checkout start-track-experiments
```

### Add DagsHub Logger

DagsHub logger is a plain Python Logger for your metrics and parameters. 

- The logger saves the information as human-readable files – CSV for metrics files and YAML for parameters. 

- Once you push these files to your DagsHub repository, they will be automatically parsed and visualized in the Experiments Tab. 

NOTE: Since DagsHub Experiments uses generic formats, you don't have to use DagsHub Logger. Instead, you can write your metrics and parameters into `metrics.csv` and `params.yml` files however you want and push them to your DagsHub repository where they will automatically be scanned and added to the experiment tab.

```bash
  # install the python package
  pip3 install dagshub
```

Now import dagshub to `modeling.py` module and track the Random Forest Classifier Hyperparameters and ROC AUC Score.

```bash
  # checkpoint
  git status -s

  # Track and commit the changes with Git
  git add src/modeling.py
  git commit -m "Add DagsHub Logger to the modeling module"  
```

### Create New Experiment

To create a new experiment, we need to update at least one of the two `metrics.csv` or `params.yml` files, track them using Git, and push them to the DagsHub repository. 

After editing the `modeling.py` module, once we run its script it will generate those two files.

```bash
  # Run the script
  python3 src/modeling.py

  git status -s
```

As we can see for the above output, two new files were created containing the current experiment's information.

The `metrics.csv` file has four fields:

- Name: the name of the Metric.
- Value: the value of the Metric.
- Timestamp: the time that the log was written.
- Step: the step number when logging multi-step metrics like loss.

The `params.yml` file holds the hyperparameters of the Random Forest Classifier.

```bash
  # Commit and push the files to the DagsHub repository using Git
  git add metrics.csv params.yml
  git commit -m "New Experiment - Random Forest Classifier with basic processing"
  git push
```

The two files were added to the repository and one experiment was created.

The information about the experiment is displayed under the Experiment Tab.

This part covers the Experiment Tracking workflow. We highly recommend reading the experiment tab documentation to explore the various features that it has to offer. 


## Explore a New Hypothesis

In the previous part, we learned how to track the project's files using Git and DVC, and track the experiments using DagsHub. 

This part covers the most common practice of Exploring a New Hypothesis. 

We will learn how to examine a new approach to process the data, compare the results, and save the project's best result.

```bash
  # switch branches
  git checkout master
```

### Basic Theory

The Data Science field is research-driven and exploring different solutions to a problem is a core principle. When a project evolves or grows in complexity, we need to compare results and see what approaches are more promising than others. In this process, we need to make sure we don't lose track of the project's components or miss any information. Therefore, it is useful to have a well-defined workflow.

The common workflow of exploring a new approach is to create a new branch for it. 

In the branch, we will change the code, modify the data and models, and track them using Git and DVC. We compare the new model's performances with the current model. This comparison can be a hassle when not using the proper tools to track and visualize the result. 

We can use DagsHub to overcome these challenges:

- We will log the models' performances to readable formats and commit them to DagsHub. 

- Using the Experiment Tab, we will easily compare the results and determine if the new approach was effective or not. 

- We can either merge the code, data, and models to the main branch or return to the main branch and retrieve the data and models from the remote storage to continue to the next experiment.

### Create a New Branch

We are using the Enron data set that contains emails. 

The emails are stored in a CSV file and labeled as 'Ham' or 'Spam'.

The current data processing method for the emails is to lower-case the characters and removes the string's punctuations. 

We will try to reduce the processing time by not removing punctuations and see how it will affect the model's performance.

```bash
  # Create new branch
  git checkout -b data-with-punctuations
```

### Update the Processing Method

Change the code in the `data_preprocessing.py` module.

```bash
  # Change the code in data_preprocessing.py
  # Track and Commit the changes with Git
  git add src/data_preprocessing.py
  git commit -m "Change data processing method - will not remove the string's punctuations"

  # Run the script
  python3 src/data_preprocessing.py

  # checkpoint
  git status
  dvc status

  # Track and commit the changes using DVC and Git.
  dvc add data
  git add data.dvc
  git commit -m "Processed the data and tracked it using DVC and Git"

  # Push the code and data changes to the remotes
  git push origin data-with-punctuations
  git push -f origin data-with-punctuations
  dvc push -r origin
```

### Run a new Experiment and Compare the Results

We have everything set to run our second Data Science Experiment! We will train a new model and log its performance using the DagsHub logger. Then, we will push the updated metrics.csv file to DagsHub and easily compare the results.

```bash
  # Runs script
  python3 src/modeling.py

  # checkpoint
  git status -s
  dvc status

  # Track the changes using Git and push the to the DagsHub repository
  git add metrics.csv
  git commit -m "Punctuations experiment results - update metrics.csv file"
  git push origin data-with-punctuations
```

With DagsHub, we can easily compare the model's performance between the two experiments. 

We can open the Experiment Tab in the DagsHub repository and compare the model's ROC AUC scores.

As we can see in the image above, the new data processing method did not provide better results so we will not use it.

### Retrieve Files

Our experiment resulted in worse performance and we want to retrieve the previous version. Now, we can reap the benefits of our workflow. 

The best version of the project is always stored on the main branch. 

hen concluding an experiment with insufficient impprovements, we simply need to check out the version we want (the master branch) and pull the remote storage files based on the .dvc pointers.

```bash
  # Checkout to branch master using Git and pull the data files 
  # from the remote storage using DVC
  git checkout master
  dvc checkout
```

Congratulations - Now we are finished!

In the Get Started section, we covered the fundamentals of DagsHub usage:

- We started by creating a repository and configuring Git and DVC. 

- We added project files to the repository using Git (for code and configuration files) and DVC (for data). 

- We created our very first data science experiment using DagsHub logger to log metrics and parameters. 

- We learned how to explore new approaches and retrieve another version's files.



## References

[1] [Get Started](https://dagshub.com/docs/getting_started/overview/)

[2] [Get Started: Data Versioning](https://dvc.org/doc/start/data-management)

[3] [Setup Remote Storage for Data and Models](https://dagshub.com/docs/integration_guide/set_up_remote_storage_for_data_and_models/)



