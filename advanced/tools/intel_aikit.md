# Intel oneAPI AI Analytics Toolkit

There are different hardware architectures such as CPU, GPU, FPGAs, AI accelerators, etc. The code written for one architecture cannot easily run on another architecture. 

> Intel oneAPI is mainly for Linux64

Therefore, Intel created a unified programming model called oneAPI to solve this very same problem. With oneAPI, it does not matter which hardware architectures (CPU, GPU, FGPA, or accelerators) or libraries or languages, or frameworks you use, the same code runs on all hardware architectures without any changes and additionally provides performance benefits.

```bash
  conda install -c intel intel-aikit
```


curl -O "https://packagecontrol.io/prerelease/Package Control.sublime-package"

## Intel optimized Modin

With Intel optimized Modin, you can expect further speed improvement on Pandas operations. 

```bash
  conda create -n aikit-modin -c intel intel-aikit-modin  # install in new environment
  conda install -c intel intel-aikit-modin
```

> The intel-aikit-modin package includes Intel Distribution of Modin, Intel Extension for Scikit-learn, and Intel optimizations for XGboost. 


## Intel optimized Scikit-learn

The Intel optimized Scikit-learn helps to speed the model building and inference on single and multi-node systems.

```bash
  conda install -c conda-forge scikit-learn-intelex
```

```py
  from sklearnex import patch_sklearn
  patch_sklearn()
  
  # undo the patching
  sklearnex.unpatch_sklearn()
```

## Intel optimized XGBoost

The XGBoost is one of the most widely used boosting algorithms in data science. In collaboration with the XGBoost community, Intel has optimized the XGBoost algorithm to provide high-performance w.r.t. model training and faster inference on Intel architectures.

```py
  import xgboost as xgb
```

## Intel optimized TensorFlow and Pytorch

In collaboration with Google and Meta (Facebook), Intel has optimized the two popular deep learning libraries TensorFlow and Pytorch for Intel architectures. 

By using Intel-optimized TensorFlow and Pytorch, you will benefit from faster training time and inference.

To use Intel-optimized TensorFlow and Pytorch, you do not have to modify anything. We just need to install `intel-aikit-tensorflow` or `intel-aikit-pytorch`. 

```bash
  conda create -n aikit-tf -c intel intel-aikit-tensorflow  # default is tensorflow 2.5
  conda create -n aikit-pt -c intel intel-aikit-pytorch

  # ==========

  # cannot install on macos
  # workaround unable to resolve environment for intel-aikit-tensorflow
  # and mkl-service package failed to import.
  conda create -n aikit-tf

  conda activate aikit-tf
  mamba update --all -y

  # install tensorflow v 2.6
  # incompatible with intel-aikit-modin=2021.4.1
  conda install -c intel intel-aikit-tensorflow=2023.1.1  # tensorflow v 2.6

  # may be needed after install intel-aikit-tensorflow
  mamba install python-flatbuffers
```

## Intel optimized Python

The AI Analytics Toolkit also comes with Intel-optimized Python. 

When you install any of the above-mentioned tools (Modin, TensorFlow, or Pytorch), Intel optimized Python is also installed by default.

This Intel distribution of Python includes commonly used libraries such as Numpy, SciPy, Numba, Pandas, and Data Parallel Python. 

All these libraries are optimized to provide high performance which is achieved with the efficient use of multi-threading, vectorization, and more importantly memory management.


## Model Zoo for Intel Architecture

[Intel Model Zoo](https://github.com/IntelAI/models) contains links to pre-trained models (such as ResNet, UNet, BERT, etc.), sample scripts, best practices, and step-by-step tutorials to run popular machine learning models on Intel architecture.


## Intel Neural Compressor

Intel Neural Compressor [3] is an open-source Python library that helps developers deploy low-precision inference solutions on popular deep learning frameworks (TensorFlow, Pytorch, and ONNX).

The tool automatically optimizes low-precision recipes by applying different compression techniques such as quantization, pruning, mix-precision, etc. and thereby increasing inference performance without losing accuracy.


```bash
  # Install intel conda packages with Continuum's Python (version conflicts on Linux)
  conda install mkl intel::mkl --no-update-deps
  conda install numpy intel::numpy --no-update-deps


  # Install intel optimization for tensorflow from anaconda channel 
  # cannot install tensorflow-mkl on macOS (version conflicts)
  conda install -c anaconda tensorflow
  conda install -c anaconda tensorflow-mkl

  # Install intel optimization for tensorflow from intel channel
  # conda install tensorflow -c intel
```

```bash
  # Intel Extension for Scikit-learn
  conda install -c conda-forge scikit-learn-intelex

  from sklearnex import patch_sklearn
  patch_sklearn()
```



----------


## Optimize your CPU for Deep Learning

Anaconda has now made it convenient for the AI community to enable high-performance-computing in TensorFlow.

Starting from TensorFlow v1.9, Anaconda has and will continue to build TensorFlow using oneDNN primitives to deliver maximum performance in your CPU.

```bash
    # turn off anaconda default SSL verification
    conda config --set ssl_verify false

    # Setup Intel python in a new virtual environment
    conda create -n intel -c intel intelpython3_full=3.9
    conda activate intel
    
    brew install python@3.9
    ln -s /home/linuxbrew/.linuxbrew/bin/python3.9 /home/linuxbrew/.linuxbrew/bin/python3

    conda create -n aikit-tf -c intel -c conda-forge intel-aikit-tensorflow

    # Intel provides an optimized math kernel library (mkl)
    # that optimizes the mathematical operations and improves speed.
    # conda install -c anaconda tensorflow-mkl
    # conda install -c intel keras

    python3 -V
```


## Setting up a Ubuntu 18.04 LTS system for Scientific Computing

```bash
    # find info about package
    sudo apt-cache search 'package-name'

    sudo apt update
    sudo apt install -y cmake flex patch zlib1g-dev libbz2-dev libboost-all-dev libcairo2 libcairo2-dev \
                     libeigen3-dev lsb-core lsb-base net-tools network-manager \
                     git-core git-gui git-doc xclip gdebi-core
    git clone git@github.com:codecypher/Mac.git
```



## References

[1] [Introduction to Intel oneAPI AI Analytics Toolkit](https://pub.towardsai.net/introduction-to-intels-oneapi-ai-analytics-toolkit-8dd873925b96?gi=25547ad4241c)

[2] [Install Intel® AI Analytics Toolkit via Conda](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/install-intel-ai-analytics-toolkit-via-conda.html)

[3] [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) 

[4] [Optimize your CPU for Deep Learning](https://towardsdatascience.com/optimize-your-cpu-for-deep-learning-424a199d7a87)


[How to Speed up Scikit-Learn Model Training](https://medium.com/distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1)


[Installing Intel Distribution for Python and Intel Performance Libraries with Anaconda](https://www.intel.com/content/www/us/en/developer/articles/technical/using-intel-distribution-for-python-with-anaconda.html)

[Intel Optimization for TensorFlow Installation Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html)

