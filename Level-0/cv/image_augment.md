# Image Augmentation

## Image Data Preparation

When training vision models, it is common to resize images to a lower dimension ((224 x 224), (299 x 299), etc.) to allow mini-batch learning and also to keep up the compute limitations [1].

We generally make use of image resizing methods like _bilinear interpolation_ for this step and the resized images do not lose much of their perceptual character to the human eyes.

In "Learning to Resize Images for Computer Vision Tasks," Talebi et al. show that if we try to optimize the perceptual quality of the images for the vision models rather than the human eyes, their performance can be improved even further [1].

**For a given image resolution and a model, how to best resize the given images?**

As shown in the paper, this idea helps to consistently improve the performance of the common vision models (pre-trained on ImageNet-1k) such as DenseNet-121, ResNet-50, MobileNetV2, and EfficientNets.

The example implements the learnable image resizing module as proposed in the paper and demonstrates the process on the Cats and Dogs dataset using the DenseNet-121 architecture.


## Feature Engineering for Images

Here are some important concepts for feature engineering when dealing with images.

### Curse of Dimensionality

The _curse of dimensionality_ is used to describe the problems that arise when working with datasets that have a large number of dimensions [4].

One reason is that the number of data points needed to accurately learn the underlying distribution of the data increases exponentially with the number of dimensions.

Thus, if we have a data set with 100 features, we may need 1,000 or even 10,000 data points to train a well-performing model.

### Reduce Picture Dimension

A simple way to reduce the dimension of our feature vector is to decrease the size of the image with decimation (downsampling) by reducing the resolution of the image [4].

If the color component is not relevant, we can also convert pictures to grayscale to divide the number dimension by three. But there are other ways to reduce the dimension of the picture and potentially extract features. For example, we can use wavelet decomposition.

Wavelet decomposition is a way of breaking down a signal in both space and frequency. In the case of pictures, this means breaking down the image into its horizontal, vertical, and diagonal components.

### Histogram of Oriented Gradient

The HOG feature descriptor is a popular technique used in computer vision and image processing for detecting objects in digital images [4].

The HOG descriptor became popular after Dalal and Triggs showed the efficiency of this descriptor in 2005 that focused on pedestrian detection in static images.

The HOG descriptor is a type of feature descriptor that encodes the shape and appearance of an object by computing the distribution of intensity gradients in an image.

The most important parameter in our case is the number of pixels per cell as it will give us a way to find the best trade-off between the number of dimensions and the number of details captured in the picture.

Example of the histogram of oriented gradient with 8 by 8-pixel cells.

Example of the histogram of oriented gradient with 16 by 16-pixel cells.

For the example above, the input image has 20,000 dimensions (100 by 200 pixels) and the HOG feature has 2,400 dimensions with the 8 by 8-pixel cell and 576 for the 16 by 16-pixel cell. That’s an 88% and 97% reduction, respectively.

### Principal Component Analysis

We can also use Principal Component Analysis (PCA) to reduce the dimension of our feature vector [4].

PCA is a statistical technique that can be used to find the directions (components) that maximize the variance and minimizes the projection error in a dataset.

Axis with the largest variance (in green) and lower projection error (in red) (image by author)
In other words, PCA can be used to find the directions that represent the most information in the data.

There are a few things to keep in mind when using PCA [4]:

- PCA is best used as a tool for dimensionality reduction, not for feature selection.

- When using PCA for dimensionality reduction, it is important to normalize the data first.

- PCA is a linear transformation, so it will not be able to capture non-linear relationships in the data.

- To reduce to N dimensions, you need at least N-1 observations

### Manifold Learning

Manifold Learning is in some ways an extension of linear methods like PCA to reduce dimensionality but for non-linear structures in data [4].

A _manifold_ is a topological space that is locally Euclidean which means that near each point it resembles the Euclidean space.

Manifolds appear naturally in many areas of mathematics and physics and the study of manifolds is a central topic in differential geometry.

There are a few things to keep in mind when working with manifold learning [4]:

- Manifold learning is a powerful tool for dimensionality reduction.

- Manifold learning can be used to find hidden patterns in data.

- Manifold learning is usually a computationally intensive task, so it is important to have a good understanding of the algorithms before using them.

It is rare that a real-life process uses all of its dimensions to describe its underlying structure.

For example, in the pictures below only a few dimensions should be necessary to describe the cup's position and rotation.

In this case, once projected with a manifold learning algorithm such as t-distributed Stochastic Neighbor Embedding (t-SNE), only two dimensions are able to encode the cup position and rotation.

There are many ways to reduce the dimension of a picture. The approach to take will depend on the type of data and the problem being solved.

Feature engineering is an iterative process, so it helps to have an overview of different possibilities and available approaches.

---------

## Image Augmentation using imgaug

The term _image augmentation_ refers to techniques used to increase the amount of data by adding slightly modified copies of existing data or creating  synthetic data from existing data.

In ML, data augmentations are used to reduce overfitting data during  training.

Image augmentation generates modified training data using existing image instances which boosts model performance, especially on small datasets and in cases of class imbalance.

In short, we make new versions of our images that are based on the originals but include intentional flaws.

Creating complex augmentation functions from scratch can be a difficult undertaking for programmers which is where the library `imgaug` can help.

Here we will create three augmentations that are commonly used in model training.

### Brightness

HSV (Hue, Saturation, Value) is a colour space developed by A. R. Smith in 1978 based on intuitive colour properties, often known as the Hexcone Model. This model’s colour parameters are hue (H), saturation (S), and lightness (V).

We can adjust the V value of the image:

```py
aug = iaa.imgcorruptlike.Brightness(severity=3)
fig = plt.figure(figsize=(17, 17))
for n, images in enumerate(image[0:3]):
    fig.add_subplot(2, 3, n+1)
    blur_image = aug(image=cv2.imread(images))
    plt.imshow(blur_image[:, :, ::-1])
    plt.title('Dog: {}'.format(n))
    plt.show()
```

### Blurness

Blurriness is obtained by calculating and analysing the Fast Fourier Transform.

The Fourier transform identifies the frequencies present in the image.

If there are not many high frequencies, the image will be fuzzy.

It is up to you to define the terms ‘low’ and ‘high.’

We can apply different blur methods of the image:

```py
aug = iaa.imgcorruptlike.MotionBlur(severity=5)
fig = plt.figure(figsize=(20, 20))
for n, images in enumerate(image[0:3]):
    fig.add_subplot(2, 3, n+1)
    blur_image = aug(image=cv2.resize(cv2.imread(images), (750, 1000), interpolation = cv2.INTER_AREA))
    plt.imshow(blur_image[:, :, ::-1])
    plt.title('Dog: {}'.format(n))
    plt.show()
```

### Gaussian Noise

Gaussian noise is a sort of noise with a Gaussian distribution such as the white noise typically observed which has a random value and is in impulses.

We can add different noise methods to the image:

```py
aug = iaa.imgcorruptlike.GaussianNoise(severity=5)
fig = plt.figure(figsize=(20, 20))
for n, images in enumerate(image[0:3]):
    fig.add_subplot(2, 3, n+1)
    blur_image = aug(image=cv2.resize(cv2.imread(images), (750, 1000), interpolation = cv2.INTER_AREA))
    plt.imshow(blur_image[:, :, ::-1])
    plt.title('Dog: {}'.format(n))
    plt.show()
```

### Saturation Augmentation

Saturation augmentation is similar to hue augmentation in that it adjusts the image’s vibrancy.

A grayscale image is entirely desaturated, a partially desaturated image has muted colours, and a positive saturation pushes hues closer to the primary colours.

When colors in an image differ, adjusting the saturation of an image can help the model perform better.

We can control the saturation level on the image:

```py
aug = iaa.imgcorruptlike.Saturate(severity=3)
fig = plt.figure(figsize=(20, 20))
for n, images in enumerate(image[0:3]):
    fig.add_subplot(2, 3, n+1)
    blur_image = aug(image=cv2.resize(cv2.imread(images), (750, 1000), interpolation = cv2.INTER_AREA))
    plt.imshow(blur_image[:, :, ::-1])
    plt.title('Dog: {}'.format(n))
    plt.show()
```

### Rotation

A random rotation of the source picture clockwise or counterclockwise by a specified amount of degrees alters the item's location in the frame.

Random rotation can help you enhance your model without collecting and labelling additional data.

We can control the rotation level on the image:

```py
aug = iaa.Affine(rotate=(-45, 45))
fig = plt.figure(figsize=(20, 20))
for n, images in enumerate(image[0:3]):
    fig.add_subplot(2, 3, n+1)
    blur_image = aug(image=cv2.resize(cv2.imread(images), (750, 1000), interpolation = cv2.INTER_AREA))
    plt.imshow(blur_image[:, :, ::-1])
    plt.title('Dog: {}'.format(n))
    plt.show()
```


### Multiple Augmentations

We can perform multiple augmentations on a single batch of images.

Here we do the following:

- We apply the Crop augmentation to crop the single image from each side anywhere between 0 to 16px which is randomly chosen.

- We apply Gaussian Noise and Motion Blur with a severity value of 5.

```py
aug = iaa.Sequential([
    iaa.Crop(px=(0, 16)),
    iaa.imgcorruptlike.GaussianNoise(severity=5),
    iaa.imgcorruptlike.MotionBlur(severity=5)
])

fig = plt.figure(figsize=(20, 20))
for n, images in enumerate(image[0:3]):
    fig.add_subplot(2, 3, n+1)
    blur_image = aug(image=cv2.resize(cv2.imread(images), (750, 1000), interpolation = cv2.INTER_AREA))
    plt.imshow(blur_image[:, :, ::-1])
    plt.title('Dog: {}'.format(n))
    plt.show()
```

----------


## Tutorials

### Image Data Preparation

[How to Manually Scale Image Pixel Data for Deep Learning](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)

[How to Normalize, Center, and Standardize Images in Keras](https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/)

[How to Evaluate Pixel Scaling Methods for Image Classification](https://machinelearningmastery.com/how-to-evaluate-pixel-scaling-methods-for-image-classification/)


### Image Data Augmentation

[Image Processing and Data Augmentation Techniques for Computer Vision](https://towardsdatascience.com/image-processing-techniques-for-computer-vision-11f92f511e21)

[Data Augmentation Compilation with Python and OpenCV](https://towardsdatascience.com/data-paugmentation-compilation-with-python-and-opencv-b76b1cd500e0)


[5 Image Augmentation Techniques Using imgAug](https://betterprogramming.pub/5-common-image-augmentations-for-machine-learning-c6b5a03ebf38)

[5 Useful Image Manipulation Techniques Using Python OpenCV](https://betterprogramming.pub/5-useful-image-manipulation-techniques-using-python-opencv-505492d077ef)


[How to Configure and Use Image Data Augmentation using Keras ImageDataGenerator](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)


[Introduction to Test-Time Data Augmentation](https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/)


### Image Data Pipeline

We can build better and faster image pipelines using `tf.data` [5].

While training a neural network, it is quite common to use `ImageDataGenerator` class to generate batches of tensor image data with real-time data augmentation, but the `tf.data` API can be used to build a faster input data pipeline with reusable pieces.


### Keras Examples

[3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification/)




## Data Augmentation for MNIST

[Improving Classification accuracy on MNIST using Data Augmentation](https://towardsdatascience.com/improving-accuracy-on-mnist-using-data-augmentation-b5c38eb5a903?gi=916228e35c66)

We can write a method to shift the images in all four directions by the given order.

We will shift the images to each of the four directions by one pixel and generate four more images from a single image.

----------


## MNIST Image Augmentation Using Tensorflow

The tutorial in [7] uses the `ImageDataGenerator` class in the `tensorflow.keras` python library.

The article in [8] provides a better alternative using tensorflow Dataset.


### Step 1: Import the MNIST dataset

In step 1, we will import the MNIST dataset using the tensorflow library. The imported dataset will be divided into train/test and input/output arrays.

```py
    from tensorflow.keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

### Step 2: Identify and Plot Baseline Digits Using Matplotlib

We plot a subset of the MNIST images to help us understand the augmentation effects on the MNIST dataset.

To plot a subset of MNIST images, use the following code:

### Step 3:  Understand Image Augmentation and Techniques Relevant To MNIST

The original MNIST dataset contains centered, upright, and size normalized digits.

Realistically, hand-written digits will seldom meet these criteria in real-world applications. Some digits will be larger, smaller, rotated, or skewed more than others.

To create a robust digit recognition model, it is in your interest to augment the MNIST dataset and capture these types of behavior.

We discuss the various types of augmentation techniques we can use to enhance the MNIST digit dataset using the Keras `ImageDataGenerator` class.

- Rotate
- Shift
- Shear
- Zoom

- Crop (center and random)
- Resize
- Flip (horiz/vert)
- ColorJitter
- Blur
- Greyscale

- Adding Noise
- Saturation
- Cutout
- Filter

_Cutout_ is a simple regularization technique of randomly masking out square regions of input during training which can be used to improve the robustness and overall performance of convolutional neural networks.

This method can also be used in conjunction with existing forms of data augmentation and other regularizers to further improve model performance.


_ColorJitter_ is another simple type of image data augmentation where we randomly change the brightness, contrast, and saturation of the image.

#### Overlay Images

Sometimes, we need to add a background to an existing image for formatting purposes. For instance, by padding a solid color as margins, we can make many images of different sizes become the same shape. Several techniques are relevant here.

### Step 4: Augment The MNIST Dataset

Finally, we can combine all of the previously mentioned transformations to obtain unique digit representations that can now be used to improve digit recognition model performance.


## References

[1]: [Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer/)

[2]: [5 Image Augmentation Techniques Using imgAug](https://betterprogramming.pub/5-common-image-augmentations-for-machine-learning-c6b5a03ebf38)

[3]: [How to Load Large Datasets From Directories](https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/)

[4]: [Feature Engineering for Images](https://towardsdatascience.com/feature-engineering-for-machine-learning-with-picture-data-d7ff8554920)

[5]: [Time to Choose TensorFlow Data over ImageDataGenerator](https://towardsdatascience.com/time-to-choose-tensorflow-data-over-imagedatagenerator-215e594f2435)

[6]: [An Intuitive Guide to PCA](https://towardsdatascience.com/an-intuitive-guide-to-pca-1174055fc800)

[7]: [How To Augment the MNIST Dataset Using Tensorflow](https://medium.com/the-data-science-publication/how-to-augment-the-mnist-dataset-using-tensorflow-4fbf113e99a0)

[8]: [Time to Choose TensorFlow Data over ImageDataGenerator](https://towardsdatascience.com/time-to-choose-tensorflow-data-over-imagedatagenerator-215e594f2435)

[9]:  [How to Explore a Dataset of Images with Graph Theory](https://towardsdatascience.com/how-to-explore-a-dataset-of-images-with-graph-theory-fd339c696d99)
