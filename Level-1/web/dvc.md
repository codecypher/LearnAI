# DVC Get Started

## Get Started

Assuming DVC is already installed, we can initialize it by running dvc init inside a Git project:

We are be building an NLP project from scratch together. 

The final result is published on GitHub, so we can fork and clone the repo.

```bash
  # Setup DVC
  # pip install dvc
  mamba install dvc
  dvc init
  git status
```

Edit `.dvc/config`.

```bash
    dvc remote add -d storj s3://dvc-get-started/dvcstore
    dvc remote modify storj endpointurl https://gateway.storjshare.io/dvc-get-started
    dvc remote modify storj access_key_id jughy4gkwhncxtikac36l5oqac7q
    dvc remote modify storj secret_access_key j2sb5lbyqyo5giabatq4jogee2aegwipaaftsiqelp33vozgae23i
    dvc remote modify storj listobjects true

    dvc remote modify myremote grant_full_control \
      id=aws-canonical-user-id,id=another-aws-canonical-user-id

    dvc pull

    git status
    dvc status
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

## Data Versioning

We can make Git handle arbitrarily large files and directories with the same performance it has with small code files.

We can a repository and see data files and machine learning models in the workspace. 

The foundation of DVC consists of a few commands that we can run along with `git` to track large files, directories, or ML model files.

Having initialized a project in the previous section, we can get the data file we will be using:

```bash
    dvc get https://github.com/iterative/dataset-registry \
          get-started/data.xml -o data/data.xml
```

To start tracking a file or directory:

```bash
    dvc add data/data.xml
```

DVC stores information about the added file in a special .dvc file named `data/data.xml.dvc` which is a small text file with a human-readable format. 

The metadata file is a placeholder for the original data that can be easily versioned like source code with Git:

```bash
    git add data/data.xml.dvc data/.gitignore
    git commit -m "Add raw data"
```

`dvc add` moved the data to the project's cache, and linked it back to the workspace. 

Now, the `.dvc/cache` should look like this:

```
    .dvc/cache
    └── 22
        └── a1a2931c8370d3aeedd7183606fd7f
```

The hash value of the data.xml file we just added (22a1a29...) determines the cache path shown above. And if you check `data/data.xml.dvc`, you will find it there too:

```
    outs:
      + md5: 22a1a2931c8370d3aeedd7183606fd7f
        path: data.xml
```

### Storing and Sharing

We can upload DVC-tracked data or model files with `dvc push`, so they are safely stored [remotely](https://dvc.org/doc/command-reference/remote) which means they can also be retrieved on other environments later with `dvc pull`. 

First, we need to set up a remote storage location:

```bash
    dvc remote add -d storage s3://mybucket/dvcstore
    dvc remote add -d storage s3://dvc-get-started

    dvc remote default main

    git add .dvc/config
    git commit -m "Configure remote storage"

    git push origin data-with-punctuations
    dvc push

    dvc remote list
```

```ini
# .dvc/config
[core]
    remote = storj
['remote "storj"']
    url = s3://dvc-get-started
    endpointurl = https://gateway.storjshare.io
    access_key_id = jughy4gkwhncxtikac36l5oqac7q
    secret_access_key = j2sb5lbyqyo5giabatq4jogee2aegwipaaftsiqelp33vozgae23i
    listobjects = true
```

DVC supports many remote storage types including Amazon S3, SSH, Google Drive, Azure Blob Storage, and HDFS. See `dvc remote add` for more details and examples.

We usually also want to `git commit` and `git push` the corresponding `.dvc` files.

DVC remotes let you store a copy of the data tracked by DVC outside of the local cache (usually a cloud storage service). For simplicity, we will setup a local remote in a temporary `dvcstore` directory (create the dir first if needed):

`dvc push` copied the data cached locally to the remote storage we set up earlier. 

The remote storage directory should look like this:

```
    .../dvcstore
    └── 22
        └── a1a2931c8370d3aeedd7183606fd7f
```

### Retrieve

Having DVC-tracked data and models stored remotely, it can be downloaded when needed in other copies of this project with `dvc pull` which we usually run after `git clone` and `git pull`.

If you have run `dvc push`, you can delete the cache (.dvc/cache) and data/data.xml to experiment with `dvc pull`:

```bash
    rm -rf .dvc/cache
    rm -f data/data.xml

    dvc pull
```

### Make changes

When you make a change to a file or directory, run `dvc add` again to track the latest version:

```bash
    dvc add data/data.xml
```

Usually we also run `git commit` and `dvc push` to save the changes:

```bash
    git commit data/data.xml.dvc -m "Dataset updates"
    dvc push
```

### Switch between versions

The regular workflow is to use `git checkout` first (to switch a branch or checkout a .dvc file version) and then run `dvc checkout` to sync data:

```bash
    # Go back to the original version of the data
    git checkout HEAD~1 data/data.xml.dvc
    dvc checkout
```

 We can commit it (no need to do dvc push this time since this original version of the dataset was already saved):

 ```bash
     git commit data/data.xml.dvc -m "Revert dataset updates"
 ```


DVC is technically not even a version control system. 

The `.dvc `file contents define data file versions and Git provides the version control. 

DVC in turn creates these `.dvc` files, updates them, and synchronizes DVC-tracked data in the workspace efficiently to match them.

### Large Datasets

In cases where we process very large datasets, we need an efficient mechanism (in terms of space and performance) to share a lot of data including different versions. Do you use network attached storage (NAS)? Or a large external volume? You can learn more about advanced workflows using these links:

- A [shared cache](https://dvc.org/doc/user-guide/how-to/share-a-dvc-cache) can be set up to store, version, and access a lot of data on a large shared volume efficiently.

- An advanced scenario is to track and version data directly on the remote storage (such as S3). See [Managing External Data](https://dvc.org/doc/user-guide/managing-external-data) to learn more.



## Data and Model Access

We learned how to track data and models with DVC and how to commit their versions to Git. 

- How can we use these artifacts outside of the project? 
- How do we download a model to deploy it? 
- How to download a specific version of a model? 
- How to reuse datasets across different projects?

Remember those `.dvc` files dvc add generates? Those files (and `dvc.lock` discussed later) have their history in Git. 

DVC's remote storage config is also saved in Git and contains all the information needed to access and download any version of datasets, files, and models.

 This means a Git repository with DVC files becomes an entry point and can be used instead of accessing files directly.

### Find a file or directory

We can use `dvc list` to explore a DVC repository hosted on any Git server. 

We can see what is in the `get-started` directory of the `dataset-registry` repo:

```bash
    dvc list https://github.com/iterative/dataset-registry get-started
    dvc list https://github.com/codecypher/dvc-get-started data
```

The benefit of the `dvc list` command versus browsing a Git hosting website is that the list includes files and directories tracked by both Git and DVC (`data.xml` is not visible if you check GitHub).

### Download

One way is to simply download the data with `dvc get` which is useful when working outside of a DVC project environment such as an automated ML model deployment task:

```bash
    dvc get https://github.com/iterative/dataset-registry \
          use-cases/cats-dogs
```

When working inside another DVC project though, this is not the best strategy because the connection between the projects is lost since other developers will not know where the data came from or if new versions are available.

### Import file or directory

`dvc import` downloads any file or directory while also creating a `.dvc` file (which can be saved in the project):

```bash
    dvc import https://github.com/iterative/dataset-registry \
             get-started/data.xml -o data/data.xml
```

`dvc import`  is similar to `dvc get` + `dvc add`, but the resulting `.dvc` files include metadata to track changes in the source repository which allows us to bring in changes from the data source later using `dvc update`.

Note that the dataset registry repository does not actually contain a `get-started/data.xml` file. Like `dvc get`, `dvc import` downloads from remote storage.

The `.dvc` files created by `dvc import` have special fields such as the data source repo and path (under deps):

```
    +deps:
    +- path: get-started/data.xml
    +  repo:
    +    url: https://github.com/iterative/dataset-registry
    +    rev_lock: 96fdd8f12c14fa58a1b7354f15c7adb50e4e8542
     outs:
     - md5: 22a1a2931c8370d3aeedd7183606fd7f
       path: data.xml
```

The `url` and `rev_lock` subfields under `repo` are used to save the origin and version of the dependency, respectively.

### Python API

It is also possible to integrate your data or models directly in source code with DVC's Python API. This lets you access the data contents directly from within an application at runtime. For example:

```py
    import dvc.api

    with dvc.api.open(
        'get-started/data.xml',
        repo='https://github.com/iterative/dataset-registry'
    ) as fd:
        # fd is a file descriptor which can be processed normally
```


## Data Pipelines

Versioning large data files and directories for data science is great, but not enough. 

How is data filtered, transformed, or used to train ML models? DVC introduces a mechanism to capture _data pipelines_ which are a series of data processes that produce a final result.

DVC pipelines and their data can also be easily versioned (using Git) which allows us to better organize projects, and reproduce the workflow and results later — exactly as they were built originally! 

We could capture a simple ETL workflow, organize a data science project, or build a detailed machine learning pipeline.

### Pipeline stages

Use `dvc stage add` to create _stages_ which represent processes (source code tracked with Git) which form the steps of a _pipeline_. 

Stages also connect code to its corresponding data _input_ and _output_. 

```bash
    # Get the sample code
    wget https://code.dvc.org/get-started/code.zip
    unzip code.zip
    rm -f code.zip
    tree

    pip install -r src/requirements.txt

    # commit the source code directory using Git
    git add src/
    git commit -m "download example code"
```

We can transform a Python script into a stage:

```bash
    dvc stage add -n prepare \
                    -p prepare.seed,prepare.split \
                    -d src/prepare.py -d data/data.xml \
                    -o data/prepared \
                    python src/prepare.py data/data.xml
```

A `dvc.yaml` file is generated which includes information about the command we want to run (python src/prepare.py data/data.xml), its dependencies, and outputs.

DVC uses these metafiles to track the data used and produced by the stage, so there is no need to use `dvc add` on `data/prepared` manually.

After we have added a stage, we can run the pipeline with `dvc repro`. Then, we can use `dvc push` if you want to save all the data to remote storage (usually along with `git commit `to version the DVC metafiles).

The resulting prepare stage in `dvc.yaml` contains all of the information above:

### Dependency graphs (DAGs)

By using `dvc stage add` multiple times and specifying outputs of a stage as dependencies of another one, we can describe a sequence of commands which gets to a desired result. 

This is what we call a _data pipeline_ or _dependency graph_.

We can create a second stage chained to the outputs of prepare to perform feature extraction:

```bash
    dvc stage add -n featurize \
                -p featurize.max_features,featurize.ngrams \
                -d src/featurization.py -d data/prepared \
                -o data/features \
                python src/featurization.py data/prepared data/features
```

The `dvc.yaml` file is updated automatically and should include two stages now.

We can add the training itself. Nothing new this time, just the same `dvc run` command with the same set of options:

```bash
    dvc stage add -n train \
                -p train.seed,train.n_est,train.min_split \
                -d src/train.py -d data/features \
                -o model.pkl \
                python src/train.py data/features model.pkl
```

This should be a good time to commit the changes with Git which include `.gitignore`, `dvc.lock`, and `dvc.yaml` that describe our pipeline.

### Reproduce

The whole point of creating the `dvc.yaml` file is the ability to easily reproduce a pipeline:

```bash
    dvc repro
    dvc status
```

`dvc repro` relies on the DAG definition from `dvc.yaml` and uses `dvc.lock` to determine what  needs to be run.

DVC pipelines (`dvc.yaml` file, `dvc stage add`, and `dvc repro`) solve a few important problems:

- **Automation:** run a sequence of steps in a "smart" way which makes iterating on your project faster. DVC automatically determines which parts of a project need to be run and it caches "runs" and their results to avoid unnecessary reruns.

- **Reproducibility:** the dvc.yaml and dvc.lock files describe what data to use and which commands will generate the pipeline results (such as an ML model). Storing these files in Git makes it easy to version and share.

- **Continuous Delivery and Continuous Integration (CI/CD) for ML:** describing projects in way that can be reproduced (built) is the first necessary step before introducing CI/CD systems. See the [CML](https://cml.dev/) project for some examples.

### Visualize

Having built our pipeline, we need a good way to understand its structure. Seeing a graph of connected stages would help. 

```bash
    dvc dag
```


## Metrics, Parameters, and Plots

DVC makes it easy to track metrics, update parameters, and visualize performance with plots. 

All of the above can be combined into experiments to run and compare many iterations of your ML project.

```bash
  # create a branch and check it out in one step
  git checkout -b metrics main
```

### Collecting metrics

First, let's see what is the mechanism to capture values for these ML attributes. Let's add a final evaluation stage to our pipeline from before:

```bash
    dvc run -n evaluate \
          -d src/evaluate.py -d model.pkl -d data/features \
          -M evaluation.json \
          --plots-no-cache evaluation/plots/precision_recall.json \
          --plots-no-cache evaluation/plots/roc.json \
          --plots-no-cache evaluation/plots/confusion_matrix.json \
          --plots evaluation/importance.png \
          python src/evaluate.py model.pkl data/features
```

The `-M` option here specifies a metrics file and `--plots-no-cache` specifies a plots file (produced by this stage) which will not be cached by DVC. 

The `dvc run` command generates a new stage in the `dvc.yaml` file:

The biggest difference to previous stages in our pipeline is in two new sections: `metrics` and `plots` which are used to mark certain files containing ML "telemetry". 

Metrics files contain scalar values (such as AUC) and plots files contain matrices, data series (such as ROC curves or model loss plots), or images to be visualized and compared.

NOTE: With `cache: false`, DVC skips caching the output since we want evaluation.json, precision_recall.json, confusion_matrix.json, and roc.json to be versioned by Git.

`evaluate.py` writes the model's ROC-AUC and average precision to e`valuation.json` which is marked as a metrics file with `-M`.

```json
    { "avg_prec": 0.5204838673030754, "roc_auc": 0.9032012604172255 }
```

`evaluate.py` also writes precision, recall, and thresholds arrays (obtained using precision_recall_curve) into `precision_recall.json`:

```json
    {
      "prc": [
        { "precision": 0.021473008227975116, "recall": 1.0, "threshold": 0.0 },
        ...,
        { "precision": 1.0, "recall": 0.009345794392523364, "threshold": 0.6 }
      ]
    }
```

Similarly, it writes arrays for the roc_curve into `roc.json`, confusion matrix into `confusion_matrix.json`, and an image `importance.png` with a feature importance bar chart for additional plots.

NOTE: DVC does not force us to use any specific file names, format ,or structure of a metrics or plots file. It iss completely user/case-defined. Refer to [dvc metrics](https://dvc.org/doc/command-reference/metrics) and [dvc plots](https://dvc.org/doc/command-reference/plots) for more details.

We can view tracked metrics and plots with DVC. 

```bash
    dvc metrics show
```

To view plots, first specify which arrays to use as the plot axes. We only need to do this once, and DVC will save our plot configurations.

```bash
    dvc plots modify evaluation/plots/precision_recall.json \
                   -x recall -y precision

    dvc plots modify evaluation/plots/roc.json -x fpr -y tpr

    dvc plots modify evaluation/plots/confusion_matrix.json \
                   -x actual -y predicted -t confusion
```

Now we can view the plots. 

We can run `dvc plots show` on you terminal which generates an HTML file that we can open in a browser, or we you can load the project in VS Code and use the "Plots Dashboard" of the "DVC Extension" to visualize the plots.

```bash
    dvc plots show
```

We save this iteration, so we can compare it later:

```bash
    git add .
    git commit -a -m "Create evaluation stage"
```

Later we will see how to compare and visualize different pipeline iterations. For now, we see how  we can capture another important piece of information which will be useful for comparison: parameters.

### Defining stage parameters

It is common for data science pipelines to include configuration files that define adjustable parameters to train a model, do pre-processing, etc. 

DVC provides a mechanism for stages to depend on the values of specific sections of such a config file (YAML, JSON, TOML, and Python formats are supported).

We should already have a stage with parameters in `dvc.yaml` since the featurize stage was created with this `dvc run` command.

```yaml
    featurize:
      cmd: python src/featurization.py data/prepared data/features
      deps:
        - data/prepared
        - src/featurization.py
      params:
        - featurize.max_features
        - featurize.ngrams
      outs:
        - data/features
```

The `params` section defines the parameter dependencies of the featurize stage. By default, DVC reads those values (featurize.max_features and featurize.ngrams) from a `params.yaml` file, but parameter file names and structure can also be user- and case-defined.

Here is the contents of the `params.yaml` file:

```yaml
    prepare:
      split: 0.20
      seed: 20170428

    featurize:
      max_features: 100
      ngrams: 1

    train:
      seed: 20170428
      n_est: 50
      min_split: 2
```

### Updating params and iterating

We are definitely not happy with the AUC value we got so far! Let's edit the params.yaml file to use bigrams and increase the number of features:

```
 featurize:
-  max_features: 100
-  ngrams: 1
+  max_features: 200
+  ngrams: 2
```

The beauty of `dvc.yaml` is that all we need to do now is run:

```bash
    dvc repro
```

It will analyze the changes, use existing results from the run-cache, and execute only the commands needed to produce new results (model, metrics, plots). The same logic applies to other possible adjustments such as edit source code or update datasets.

### Comparing iterations

We can see how the updates improved performance. 

DVC has a few commands to see changes in and visualize metrics, parameters, and plots which can work for one or across multiple pipeline iteration(s). 

We can compare the current "bigrams" run with the last committed "baseline" iteration:

```bash
    # show how params in the workspace differ vs the last commit
    dvc params diff
    # Path         Param                   HEAD  workspace
    # params.yaml  featurize.max_features  100   200
    # params.yaml  featurize.ngrams        1     2
```

```bash
    # show how metrics in the workspace differ vs the last commit
    dvc metrics diff
    # Path             Metric    HEAD      workspace    Change
    # evaluation.json  avg_prec  0.89668   0.9202       0.02353
    # evaluation.json  roc_auc   0.92729   0.94096      0.01368
```

```bash
    # compare all plots
    dvc plots diff
    # file:///Users/dvc/example-get-started/plots.html
```




## Experiments

In machine learning projects, the number of experiments grows rapidly. 

DVC can track these experiments, list and compare their most relevant metrics, parameters, and dependencies, navigate among them, and commit only the ones that we need to Git.

In this section, we explore the basic features of DVC experiment management with the `example-dvc-experiments` project.

### Initializing a project with DVC experiments

If you already have a DVC project, that is great. You can start to use `dvc exp` commands right away to run experiments in your project. 

Here, we briefly discuss how to structure an ML project with DVC experiments using `dvc exp init`.

A typical machine learning project has data, a set of scripts that train a model, a bunch of hyperparameters that tune training and models, and outputs metrics and plots to evaluate the models. 

The `dvc exp init` command has sane defaults for the names of these elements to initialize a project:

```bash
    dvc exp init python src/train.py
```

Here, python `src/train.py` specifies how you run experiments, but it could be any other command.

If the project uses different names for them, we can set directories for source code (default: src/), data (data/), models (models/), plots (plots/), and files for hyperparameters (params.yaml), metrics (metrics.json) with the options supplied to `dvc exp init`.

We can also set these options in a dialog format with `dvc exp init --interactive`.

Running the experiment with the default project settings requires only the command:

```bash
    dvc exp run
```

`dvc exp run` runs the command specified in `dvc.yaml` (python train.py), and creates models, plots, and metrics in the respective directories. 

The experiment is then associated with the values found in the parameters file (params.yaml) and other dependencies, as well as the metrics produced.

### More information about (Hyper)parameters

It is pretty common for data science projects to include configuration files that define adjustable parameters to train a model, adjust model architecture, do pre-processing, etc. 

DVC provides a mechanism for experiments to depend on the specific variables from a file.

By default, DVC assumes that a parameters file named `params.yaml` is available in your project. DVC parses this file and creates dependencies to the variables found in the file: model.conv_units and train.epochs.

We can review the experiment results with `dvc exp show` and see these metrics and results in a nicely formatted table:

```bash
    dvc exp show
```

The `workspace` row in the table shows the results of the most recent experiment that is available in the workspace. 

The table also shows each experiment in a separate row, along with the Git commit IDs they are attached to. 

We can see that the experiment we run has a name exp-6dccf and was run from the commit ID 7317bc6.

Now, we can do some more experimentation.

The option `dvc exp run --set-param` allows us to update experimental parameters without modifying the files manually. 

We can use this feature to set the convolutional units in `train.py`.

```bash
    dvc exp run --set-param featurize.max_features=100 --set-param featurize.ngrams=1
```

### Run multiple experiments in parallel

Instead of running the experiments one-by-one, we can define them to run in a batch which is really handy when you have long running experiments.

- We add experiments to the queue using the `--queue` option of `dvc exp run`. 
- We also use `-S` (--set-param) to set a value for the parameter.

```bash
    dvc exp run --queue -S model.conv_units=32
    dvc exp run --queue -S model.conv_units=64
    dvc exp run --queue -S model.conv_units=128
    dvc exp run --queue -S model.conv_units=256
```

Now, run all (--run-all) queued experiments in parallel. You can specify the number of parallel processes using --jobs:

```bash
    dvc exp run --run-all --jobs 2
```

### Comparing and persisting experiments

The experiments are run several times with different parameters. 

We can use `dvc exp show` to compare all of these experiments.

```bash
    dvc exp show
```

By default, it shows all the metrics, parameters and dependencies with the timestamp. 

If we have a large number of metrics, parameters, dependencies or experiments, this may lead to a cluttered view. 

We can limit the table to specific columns using the `--drop` option.

```bash
    dvc exp show --drop 'Created|train|loss'
```

### More information about metrics

Metrics are what we use to evaluate your models. 

DVC associates metrics to experiments for later comparison. Any scalar value can be used as a metric. 

We can specify text files to contain metrics using `dvc exp init --metrics`, and write them in the experimentation code.

An alternative to manual metrics generation is to use DVCLive to generate these files. 

`dvc exp show `and `dvc metrics show` are used to tabulate the experiments and Git commits with their associated metrics. 

In the above tables, loss and acc values are metrics found in the `metrics.json` file.

Metrics files are interpreted specially also in Iterative Studio.


After selecting an experiment from the table, we can create a Git branch that contains the experiment with all its related files.

```bash
    # display results of all experiments
    dvc exp show

    # create a Git branch that contains a specific experiment
    dvc exp branch exp-17dd9 "cnn-256"
    # Git branch 'cnn-256' has been created from experiment 'exp-17dd9'.

    # switch to the new branch
    git checkout cnn-256
```

We can checkout and continue working from this branch or merge the branch into the main branch with the usual Git commands.


----------



## Use Cases

We provide short articles on common data science scenarios that DVC can help with or improve. 

Our use cases are not written to be run end-to-end like tutorials. 

For more general, hands-on experience with DVC, please see "Get Started" instead.



## References

[Get Started](https://dvc.org/doc/start)

