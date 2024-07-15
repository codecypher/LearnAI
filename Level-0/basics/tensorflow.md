# Tensorflow

Here are some useful tutorials on Keras and Tensorflow. 



## Keras Tutorials

[Introduction to Keras for Engineers](https://keras.io/getting_started/intro_to_keras_for_engineers/)


[How to Load a Dataset in TensorFlow](https://medium.com/geekculture/how-to-load-a-dataset-in-tensorflow-263b53d69ffa)

[3 ways to create a Machine Learning model with Keras and TensorFlow 2.0](https://towardsdatascience.com/3-ways-to-create-a-machine-learning-model-with-keras-and-tensorflow-2-0-de09323af4d3)

[Getting started with TensorFlow 2.0](https://medium.com/@himanshurawlani/getting-started-with-tensorflow-2-0-faf5428febae)

[Introducing TensorFlow Datasets](https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3)

[How to (quickly) Build a Tensorflow Training Pipeline](https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0?gi=f2df1cc3455f)


[Deep Learning (Keras)](https://machinelearningmastery.com/start-here/#deeplearning)

[Convolutional Layers vs Fully Connected Layers](https://towardsdatascience.com/convolutional-layers-vs-fully-connected-layers-364f05ab460b)


## More Keras Tutorials

[Deep Learning with Python: Neural Networks (complete tutorial)](https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0)


[Training and evaluation with the built-in methods](https://keras.io/guides/training_with_built_in_methods/)

[Working with preprocessing layers](https://keras.io/guides/preprocessing_layers/)

[Transfer learning and fine-tuning](https://keras.io/guides/transfer_learning/)


----------



## Speedup TensorFlow Training

### Use @tf.function decorator

In TensorFlow 2, there are two execution modes: eager execution and graph execution. 

  1. In eager execution mode, the user interface is more intuitive but it suffers from performance issues because every function is called in a Python-native way. 

  2. In graph execution mode, the user interface is less intuitive but offers better performance due to using Python-independent dataflow graphs.

The `@tf.function` decorator allows you to execute functions in graph execution mode. 

```py
@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b) 
 
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)
```

When the @tf.function-decorated function is first called, a “tracing” takes place and a tf.Graph is generated. Once the graph is generated, the function will be executed in graph execution mode with no more tracing.

If you call the function with Python-native values or tensors that continuously change shapes, a graph will be generated every time which causes a lack of memory and slows down training.

### Optimize dataset loading

1. Use TFRecord format to save and load your data

Using the TFRecord format, we can save and load data in binary form that makes encoding and decoding the data much faster. 

2. Utilize tf.data.Dataset methods to efficiently load your data

TensorFlow2 supports various options for data loading.

2.1 num_parallel_calls option in interleaves and map method

The `num_parallel_calls` option represents a number of elements processed in parallel. 

We can set this option to `tf.data.AUTOTUNE` to dynamically adjust the value based on available hardware.

```py
dataset = Dataset.range(1, 6)
dataset = dataset.map(lambda x: x + 1, num_parallel_calls=tf.data.AUTOTUNE)
```

2.2 use prefetch method

The `prefetch` method creates a Dataset which prepares the next element while the current one is being processed. 

Most dataset pipelines are recommended to be ended with the `prefetch` method.

```py
dataset = tf.data.Dataset.range(3)
dataset = dataset.prefetch(2)
```

### Use mixed precision

Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory. 

In fact, forward/backward-propagation is computed in float16 and gradients are scaled to a proper range afterward. 

Since NVIDIA GPU generally runs faster in float16 rather than in float32, this reduces time and memory costs. 

Here are the steps to enable mixed precision for training. 

1. Set the data type policy

Add a line at the very beginning of the training code to set a global data type policy.

```py
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

2. Fix the data type of the output layer to float32

Add a line at the end of your model code or pass dtype='float32' to the output layer of your model.

```py
outputs = layers.Activation('linear', dtype='float32')(outputs)
```

3. Wrap your optimizer with LossScaleOptimizer

Wrap your optimizer with `LossScaleOptimizer` for loss scaling if you create a custom training loop instead of using `keras.model.fit`.

```py
optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
```

### Accelerated Linear Algebra (XLA)

Accelerated Linear Algebra (XLA) is a domain-specific compiler for linear algebra that can accelerate TensorFlow models without almost no source code changes.

XLA compiles the TensorFlow graph into a sequence of computation kernels generated specifically for the given model. 

Without XLA, TensorFlow graph executes three kernels: one for addition, one for multiplication, and one for reduction. However, XLA compiles these three kernels into one kernel so that intermediate results no longer have to be saved during the computation.

Using XLA, we can use less memory and also speed up training.

Enabling XLA is as simple as using the `@tf.function` decorator. 

```py
  # enable XlA globally
  tf.config.optimizer.set_jit(True)
  
  @tf.function(jit_compile=True)
  def dense_layer(x, w, b):
      return add(tf.matmul(x, w), b)
```


----------



## Improve Tensorflow Performance

Here are some performance tips [2] [3]. 

In TensorFlow 2, eager execution is turned on by default. The user interface is intuitive and flexible (running one-off operations is much easier and faster), but this can come at the expense of performance and deployability [4].

We can use `tf.function` which is a transformation tool that creates Python-independent dataflow graphs out of your Python code. 

This will help create performant and portable models and it is required to use `SavedModel`.

The main recommendations are [4]:

- Debug in eager mode then decorate with `@tf.function`.

- Don not rely on Python side-effects such as object mutation or list appends.

- `tf.function` works best with TensorFlow ops; NumPy and Python calls are converted to constants.


### Mixed Precision on NVIDIA GPUs

Mixed precision (MP) training offers significant computational speedup by performing operations in the half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network.

MP uses both 16bit and 32bit floating point values to represent variables to reduce the memory requirements and to speed up training. MP relies on the fact that modern hardware accelerators such as GPUs and TPUs can run computations faster in 16bit.

There are numerous benefits to using numerical formats with lower precision than 32-bit floating-point: require less memory; require less memory bandwidth. 

- Speeds up math-intensive operations such as linear and convolution layers by using Tensor Cores.

- Speeds up memory-limited operations by accessing half the bytes compared to single-precision.

- Reduces memory requirements for training models, enabling larger models or larger mini-batches.

Among NVIDIA GPUs, those with compute capability 7.0 or higher will see the greatest performance benefit from mixed-precision because they have special hardware units called Tensor Cores to accelerate float16 matrix multiplications and convolutions.


### Mix Precision in Tensorflow

The mixed precision API is available in TensorFlow 2.1 with Keras interface. 

To use mixed precision in Keras, we have to create a _dtype policy_ which specify the dtypes layers will run in. 

Then, layers created will use mixed precision with a mix of float16 and float32.

```py
  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_policy(policy)
  
  # Now design your model and train it
```

```py
import tensorflow as tf

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
  tpu = None

if tpu:
  policyConfig = 'mixed_bfloat16'
else: 
  policyConfig = 'mixed_float16'
policy = tf.keras.mixed_precision.Policy(policyConfig)
tf.keras.mixed_precision.set_global_policy(policy)
view raw mixed
```

> NOTE: Tensor Cores provide mix precision which requires certain dimensions of tensors such as dimensions of your dense layer, number of filters in Conv layers, number of units in RNN layer to be a multiple of 8.

To compare the performance of mixed-precision with float32, change the policy from `mixed_float16` to float32 which can improve performance up to 3x.

For numerical stability it is recommended that the model’s output layers use float32. This is achieved by either setting dtype=tf.float32 in the last layer or activation, or by adding a linear activation layer tf.keras.layers.Activation("linear", dtype=tf.float32) right at the model’s output. 

In addition data must be in float32 when plotting model predictions with matplotlib since plotting float16 data is not supported.

If you train your model with `tf.keras.model.fit` then you are done! If you implement a custom training loop with mixed_float16 a further step is required called _loss scaling_.

Mixed precission can speed up training on certain GPUs and TPUs. 

When using `tf.keras.model.fit`  to train your model, the only step required is building the model with mixed precission by using a global policy. 

If a custom training loop is implemented, the optimizer wrapper `tf.keras.mixed_precission.LossScaleOptimizer` should be implemented to prevent overflow and underflow.

### Fusing multiple ops into one

Usually when you run a TensorFlow graph, all operations are executed individually by the TensorFlow graph executor which means each op has a pre-compiled GPU kernel implementation. 

Fused Ops combine operations into a single kernel for improved performance.

Without fusion, without XLA, the graph launches three kernels: one for the multiplication, one for the addition and one for the reduction.

```py
  def model_fn(x, y, z): 
      return tf.reduce_sum(x + y * z)
```

Using op fusion, we can compute the result in a single kernel launch by fusing the addition, multiplication, and reduction into a single GPU kernel.

### Fusion with Tensorflow 2.x

Newer Tensorflow versions come with XLA which does fusion along with other optimizations for us.

Fusing ops together provides several performance advantages:

- Completely eliminates Op scheduling overhead (big win for cheap ops)

- Increases opportunities for ILP, vectorization etc.

- Improves temporal and spatial locality of data access


----------



## Tensorflow Install

[Setting up a Ubuntu 18.04 LTS system for deep learning and scientific computing](https://medium.com/@IsaacJK/setting-up-a-ubuntu-18-04-1-lts-system-for-deep-learning-and-scientific-computing-fab19f7ca39d)

[Optimize your CPU for Deep Learning](https://towardsdatascience.com/optimize-your-cpu-for-deep-learning-424a199d7a87)

[Intel Optimization for TensorFlow Installation Guide](https://software.intel.com/content/www/us/en/develop/articles/intel-optimization-for-tensorflow-installation-guide.html)



## Tensorflow Performance

[Optimizing a TensorFlow Input Pipeline: Best Practices in 2022](https://medium.com/@virtualmartire/optimizing-a-tensorflow-input-pipeline-best-practices-in-2022-4ade92ef8736)

[Leverage the Intel TensorFlow Optimizations for Windows to Boost AI Inference Performance](https://medium.com/intel-tech/leverage-the-intel-tensorflow-optimizations-for-windows-to-boost-ai-inference-performance-ba56ba60bcc4)



## Tensorflow GPU

[Using GPUs With Keras and wandb: A Tutorial With Code](https://wandb.ai/authors/ayusht/reports/Using-GPUs-With-Keras-A-Tutorial-With-Code--VmlldzoxNjEyNjE)

[Use a GPU with Tensorflow](https://www.tensorflow.org/guide/gpu)

[Using an AMD GPU in Keras with PlaidML](https://www.petelawson.com/post/using-an-amd-gpu-in-keras/)


[The Ultimate TensorFlow-GPU Installation Guide For 2022 And Beyond](https://towardsdatascience.com/the-ultimate-tensorflow-gpu-installation-guide-for-2022-and-beyond-27a88f5e6c6e)

[Time to Choose TensorFlow Data over ImageDataGenerator](https://towardsdatascience.com/time-to-choose-tensorflow-data-over-imagedatagenerator-215e594f2435)



## Tensorflow on macOs

[Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/)

[GPU-Accelerated Machine Learning on MacOS](https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545)

[Installing TensorFlow on the M1 Mac](https://towardsdatascience.com/installing-tensorflow-on-the-m1-mac-410bb36b776)


[apple/tensorflow_macos](https://github.com/apple/tensorflow_macos/issues/153)

[Tensorflow Mac OS GPU Support](https://stackoverflow.com/questions/44744737/tensorflow-mac-os-gpu-support)

[Install Tensorflow 2 and PyTorch for AMD GPUs](https://medium.com/analytics-vidhya/install-tensorflow-2-for-amd-gpus-87e8d7aeb812)




## References

[1]: [A simple guide to speed up your training in TensorFlow](https://blog.seeso.io/a-simple-guide-to-speed-up-your-training-in-tensorflow-2-8386e6411be4?gi=55c564475d16)

[2]: [Accelerate your training and inference running on Tensorflow](https://towardsdatascience.com/accelerate-your-training-and-inference-running-on-tensorflow-896aa963aa70)

[3]: [Speed up your TensorFlow Training with Mixed Precision on GPUs and TPUs](https://towardsdatascience.com/speed-up-your-tensorflow-training-with-mixed-precision-on-gpu-tpu-acf4c8c0931c)

[4]: [Better performance with tf.function](https://www.tensorflow.org/guide/function)
