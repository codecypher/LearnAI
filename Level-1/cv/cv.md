# Computer Vision

## What is Computer Vision

_Computer Vision (CV)_ is defined as a field of study that seeks to develop techniques to help computers “see” and understand the content of digital images such as photographs and videos [1].

Many popular computer vision applications involve trying to recognize things in photographs [1]:

- **Object Classification:** What broad category of object is in this photograph?

- **Object Identification:** Which type of a given object is in this photograph?

- **Object Verification:** Is the object in the photograph?

- **Object Detection:** Where are the objects in the photograph?

- **Object Landmark Detection:** What are the key points for the object in the photograph?

- **Object Segmentation:** What pixels belong to the object in the image?

- **Object Recognition:** What objects are in this photograph and where are they?

- Facial Recognition

## Key Computer Vision Concepts

The following sections discuss several important CV concepts [2]:

- Image Data Handling
- Image Data Preparation
- Image Data Augmentation

- Image Classification
- Object Recognition
- Object Classification
- Object Detection

## Image segmentation

**Semantic segmentation** is the process to allocate a semantic label to every pixel present in an image [5].

For example, traffic sign detection aims to classify each pixel as a car, a pedestrian, or a road sign.

Semantic segmentation (SS) is different from standard segmentation tasks which labels the pixels according to their physical properties (their color).

SS aims at annotating each pixel with its object label. The algorithm to identify objects works under the learning architecture of neural networks.

Image segmentation is the process of partitioning a digital image (still images and video images) into multiple segments. These segments are homogeneous regions in terms of content and characteristics. Segmentation is used in computer vision and image processing tasks, such as object recognition and scene understanding.

There are two common types of Image Segmentation [5]:

- Semantic Segmentation
- Instant Segmentation

## Image Data Handling

For CV projects, it can be confusing to determine which library to choose to read and manipulate image datasets [6] [7].

The `sklearn.datasets` package embeds some small toy datasets which can be accessed using the dataset loading utilities [8].

The Keras deep learning library provides access to four standard computer vision datasets [10].

The Keras deep learning library provides a sophisticated API for loading, preparing, and augmenting image data [9].

### NHWC vs HCHW

You will often encounter obscure image format issues going between Linux and macOS (NHWC vs HCHW) [4].

## Exploratory Data Analysis for CV

There are four steps to perform before starting image processing tasks [11]:

1. Check the size of the images in the dataset.
2. Check the blurriness of the images in the dataset.
3. Check the color distribution of the images in the dataset.
4. Check for there is class imbalance.

### Check the size of the images

The first thing you could probably do right after extracting the zip folder is check the size of the images.

If your image dataset is huge, you could probably consider writing several lines of code to do it for you.

```py
import cv2

# Read the image file
image = cv2.imread("image.png")

# Read the height and width of the image
height, width, shape = image.shape
```

```py
import cv2

dataset_folder = "/path/to/image/dataset/folder"

# Assuming this is a classification task and
# your dataset has multiple classes
folders = os.listdir(dataset_folder)
images = []
image_shapes = []
for folder in folders:
  folder_path = os.path.join(dataset_folder, folder)
  for path in os.listdir(folder_path):
    image_path = os.path.join(os.path.join(dataset_folder, folder), path)
    image = cv2.imread(image_path)

    # Convert to RGB as OpenCV uses BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

    # Obtain shape of each image and store them in an array
    (width, height) = image.shape[1], image.shape[0]
    image_shapes.append((width, height))

# Return only unique shapes
unique_shapes = list(set(image_shapes))
```

### Check the blurriness of the images

There could be times when a blurred image could sneak into a well-prepared image dataset and ruin your experience of exploratory data analysis.

A simple Laplacian function from OpenCV can do the trick for us.

```py
import cv2

def variance_of_laplacian(image):
  return cv2.Laplacian(image, cv2.CV_64F).var()
```

The variance of the Laplacian returns a numeric focus measure. With a threshold, we are able to tell if an image is considered blurry.

We can pick a few clear images and compute the variance of their Laplacian to get an average of thresholds to determine what is considered non-blurry. Then, any images with a score above that value will be removed.

There are ways to fix the blurriness of images such as smoothing and sharpening but removing blurry images will save you a lot of time, assuming you have a large dataset.

### Check the color distribution

Color distribution can be important if your project is related to the color features such as differentiating between green plants of different species.

Since the colors tend to overlap, understanding which channel in each color space stands out from the rest or gives a particular weight is essential.

We can try getting the mean of the R, G, and B channel of the RGB color space using the code below.

```py
# Obtain the mean of the R channel of the RGB space of the image
mean_r_values = []
for i in range(len(images)):
  mean_r_value = np.mean(images[i][:, :, 0])
  mean_r_values.append(mean_r_value)

# Obtain the mean of the G channel of the RGB space of the image
mean_g_values = []
for i in range(len(images)):
  mean_g_value = np.mean(images[i][:, :, 1])
  mean_g_values.append(mean_g_value)

# Obtain the mean of the B channel of the RGB space of the image
mean_b_values = []
for i in range(len(images)):
  mean_b_value = np.mean(images[i][:, :, 2])
  mean_b_values.append(mean_b_value)

# Obtain the mean of all channels of the RGB space of the image
mean_values = []
for i in range(len(images)):
  mean_value = np.mean(images[i])
  mean_values.append(mean_value)
```

Then, a simple plot of each mean value array will allow us to explore the color distribution in more detail which should give you an overview of the color features that you are able to extract later on.

### Check for class imbalance

Class imbalance is a crucial step in computer vision projects.

In real-world projects, balanced datasets are hard to come by which will result in the accuracy of a model on a certain class being higher than that on the other class.

The problem comes if there is a class imbalance.

Suppose you have two classes, class dog and class cat and we find that there are significantly more images in the dog class.

There are two things we can do:

1. We can try to reduce the number of images in the dog category until both classes are balanced.

2.  We can try to increase the number of images in the cat category until both classes are balanced.

If the number of images of both classes is sufficient, the first solution is hassle-free.

However, if we have no choice but to increase the number of images, data augmentation is the way to go.

We could try any of the following methods:

- Rotation
- Flip
- Translation
- Convolution

## Object Recognition

_Object recognition_ is a general term to describe a collection of related computer vision tasks that involve identifying objects in digital images.

_Image classification_ involves predicting the class of an object in an image.

_Object localization_ refers to identifying the location of one or more objects in an image and drawing a bounding box around them.

_Object detection_ combines these two tasks and localizes and classifies one or more objects in an image.

We will be using the term _object recognition_ broadly to encompass both image classification (what object classes are present in the image) and object detection (localize all objects present in the image).

We can distinguish between these three computer vision tasks:

- **Image Classification:** Predict the type or class of an object in an image.

Input: An image with a single object such as a photograph.
Output: A class label (one or more integers that are mapped to class labels).

- **Object Localization:** Locate the presence of objects in an image and indicate their location with a bounding box.

Input: An image with one or more objects, such as a photograph.
Output: One or more bounding boxes (defined by a point, width, and height).

- **Object Detection:** Locate the presence of objects with a bounding box and types or classes of the located objects in an image.

Input: An image with one or more objects, such as a photograph.
Output: One or more bounding boxes and a class label for each bounding box.

- **Object segmentation:** Instances of recognized objects are indicated by highlighting the specific pixels of the object instead of a coarse bounding box which is sometimes called _object instance segmentation_ or _semantic segmentation_.

<img width="600" alt="Overview of Object Recognition Computer Vision Tasks" src="https://machinelearningmastery.com/wp-content/uploads/2019/05/Object-Recognition.png" />

Figure: Overview of Object Recognition Computer Vision Tasks

Most of the recent innovations in image recognition problems have come as part of participation in the ILSVRC tasks which is an annual academic competition with a separate challenge for each of three problem types: image classification, object localization, and object detection.

- Image classification: Algorithms produce a list of object categories present in the image.

- Single-object localization: Algorithms produce a list of object categories present in the image and an axis-aligned bounding box indicating the position and scale of one instance of each object category.

- Object detection: Algorithms produce a list of object categories present in the image along with an axis-aligned bounding box indicating the position and scale of every instance of each object category.

## Object Detection

The approach will depend on the goal but most common object detection projects are classification apps which try to determine the counts of types of objects (pedestrian/vehicle).

Object detection involves identifying the presence, location, and type of one or more objects in a given photograph.

**Object detection** involves building upon the methods for _object classification_ (what are they), _object localization_ (what are their extent), and _object recognition_ (where are they).

Object detection combines all these tasks and draws a bounding box around each object of interest in the image and assigns them a class label.

Yolo is the latest approach to object detection and there is a version that can run on raspi.

Yolo is supposed to be easier to setup/use and have faster performance.

## Object Detection vs Object Recognition

Object detection and object recognition are both essential tasks in computer vision, but they serve different purposes [13] and [14].

### What is Object Detection

**Definition:** Object detection identifies and locates multiple objects within an image or video.

Object detection provides both the class label and the bounding box coordinates for each detected object.

Functionality: It answers the question, "What objects are where?"

For example, in a street scene, it can identify cars, pedestrians, and traffic signs, along with their locations.

Techniques: Common methods include:

- Convolutional Neural Networks (CNNs)
- You Only Look Once (YOLO)
- Region-based CNN (R-CNN)

### What is Object Recognition

**Definition:** Object recognition focuses on identifying and classifying objects in images or videos.

Object recognition usually assigns a single label to the entire image rather than locating multiple objects.

Functionality: It answers the question, "What objects are present?"

For example, it might label an image as containing a "dog" without specifying its location.

Techniques: Common methods include:

- Histogram of Oriented Gradients (HOG)
- Support Vector Machines (SVM)
- Deep learning approaches using CNNs

## Image Classification Examples

[How to Develop a CNN for CIFAR-10 Photo Classification](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)

[How to Develop a CNN to Classify Photos of Dogs and Cats](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/)

[Image Classification with Python: CNN vs Transformers](https://pub.towardsai.net/image-classification-with-python-cnn-vs-transformers-fe509cbbc2d0)

[TensorFlow for Image Classification – Transfer Learning Made Easy](https://www.kdnuggets.com/2022/01/tensorflow-computer-vision-transfer-learning-made-easy.html)

[Fixed Feature Extractor as the Transfer Learning Method for Image Classification Using MobileNet](https://audhiaprilliant.medium.com/fixed-feature-extractor-as-the-transfer-learning-method-for-image-classification-using-mobilenet-b26376e25d49)

[How to Develop a CNN to Classify Satellite Photos](https:/low /machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)

[Binary Image Classification in PyTorch](https://towardsdatascience.com/binary-image-classification-in-pytorch-5adf64f8c781)

## Object Detection Examples

[How to Train an Object Detection Model with Keras](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)

[How to Perform Object Detection With YOLOv3 in Keras](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)

[Object Detection with Convolutional Neural Networks](https://towardsdatascience.com/object-detection-with-convolutional-neural-networks-c9d729eedc18)

[Detect an object with OpenCV-Python](https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/)

[Object detection with Tensorflow model and OpenCV](https://towardsdatascience.com/object-detection-with-tensorflow-model-and-opencv-d839f3e42849)

[How to Detect Objects in Real-Time Using OpenCV and Python](https://towardsdatascience.com/how-to-detect-objects-in-real-time-using-opencv-and-python-c1ba0c2c69c0)

[Making Road Traffic Counting App based on Computer Vision and OpenCV](https://medium.com/machine-learning-world/tutorial-making-road-traffic-counting-app-based-on-computer-vision-and-opencv-166937911660)

[Traffic Sign Detection using Convolutional Neural Network and Keras](https://towardsdatascience.com/traffic-sign-detection-using-convolutional-neural-network-660fb32fe90e)

[Traffic sign recognition using deep neural networks using Keras](https://towardsdatascience.com/traffic-sign-recognition-using-deep-neural-networks-6abdb51d8b70)

[YOLO Object Detection on the Raspberry Pi](https://towardsdatascience.com/yolo-object-detection-on-the-raspberry-pi-6de3629256fa)

------

[Training a model for custom object detection (TF 2.x) on Google] Colab](https://techzizou007.medium.com/training-a-model-for-custom-object-detection-tf-2-x-on-google-colab-4507f2cc6b80)

[Training YOLOv5 custom dataset with ease](https://medium.com/mlearning-ai/training-yolov5-custom-dataset-with-ease-e4f6272148ad)

[Object tracking using YOLOv4 and TensorFlow](https://pythonawesome.com/object-tracking-implemented-with-yolov4-and-tensorflow/)

------

[Vehicle Detection and Tracking using Machine Learning and HOG](https://towardsdatascience.com/vehicle-detection-and-tracking-using-machine-learning-and-hog-f4a8995fc30a?gi=b793ee27f135)

[How to Perform Object Detection with Mask R-CNN](https://machinelearningmastery.com/how-to-perform-object-detection-in-photographs-with-mask-r-cnn-in-keras/)

## Keras Examples

[Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer/)

[Working with preprocessing layers](https://keras.io/guides/preprocessing_layers/)

[Transfer learning and fine-tuning](https://keras.io/guides/transfer_learning/)

[3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification/)

## Pretrained CV Models

Here are some notes on pretrained models for computer vision.

### VGG

Given a photograph of an object, determine which of 1,000 specific objects the photograph shows.

A competition-winning model for this task is the VGG model by researchers at Oxford.

What is important about this model (besides capability of classifying objects in photographs) is that the model weights are freely available and can be loaded and used in your own models and applications.

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) is an annual computer vision competition. Each year, teams compete on two tasks.

The first is to detect objects within an image coming from 200 classes which is called _object localization_.

The second is to classify images, each labeled with one of 1000 categories which is called _image classification_.

[VGG-16 CNN model](https://www.geeksforgeeks.org/vgg-16-cnn-model/)

[How to Develop VGG, Inception and ResNet Modules from Scratch in Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)

[How to Develop VGG, Inception, and ResNet Modules from Scratch in Keras](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)

### Residual Networks (ResNet)

After the first CNN-based architecture (AlexNet) that win the ImageNet 2012 competition, every subsequent winning architecture uses more layers in a deep neural network to reduce the error rate which works for less number of layers.

[Residual Networks (ResNet)](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/)

When we increase the number of layers, there is a common problem in deep learning called Vanishing/Exploding gradient which causes the gradient to become 0 or too large. Thus, when we increase the number of layers, the training and test error rate also increases.

### Residual Block

In order to solve the problem of the vanishing/exploding gradient, ResNet introduced the concept called Residual Network which uses a technique called _skip connections_.

The skip connection skips training from a few layers and connects directly to the output.

## CV and Text

The article [12] discusses how to detect and extract text, figures, tables from any type of document with Computer Vision.

## CV Libraries

Here are some notes on useful Python libraries for computer vision.

- ageitgey/face_recognition
- qubvel/segmentation_models

### OpenCV

[OpenCV](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

OpenCV is a huge open-source library for computer vision, machine learning, and image processing.

OpenCV supports a wide variety of programming languages like Python, C++, Java, etc.

OpenCV can process images and videos to identify objects, faces, or eve handwriting.

### openpilot

The openpilot repo is an open-source driver assistance system that performs the functions of Adaptive Cruise Control (ACC), Automated Lane Centering (ALC), Forward Collision Warning (FCW) and Lane Departure Warning (LDW) for a growing variety of supported car makes, models, and model years.

However, you will need to buy their product and install it on your car and it is not completely DIY but reduces the effort.

### Tutorials

[Introduction to OpenCV](https://www.geeksforgeeks.org/introduction-to-opencv/)

[OpenCV Python Tutorial](https://www.geeksforgeeks.org/opencv-python-tutorial/)

## References

[1]: [A Gentle Introduction to Computer Vision](https://machinelearningmastery.com/what-is-computer-vision/)

[2]: [Guide to Deep Learning for Computer Vision](https://machinelearningmastery.com/start-here/#dlfcv)

[3]: [A Gentle Introduction to Object Recognition](https://machinelearningmastery.com/object-recognition-with-deep-learning/)

[4]: [A Gentle Introduction to Channels-First and Channels-Last Image Formats](https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/)

[5]: [Semantic Segmentation](https://www.technologyhq.org/semantic-segmentation-what-is-it-and-how-does-it-help/)

[6]: [Image-Processing Libraries to Load Image Data in Machine Learning and Deep Learning](https://medium.com/geekculture/image-processing-libraries-to-load-image-data-in-machine-learning-and-deep-learning-4c6ca538cc95)

[7]: [How to Load and Manipulate Images for Deep Learning in Python With PIL/Pillow](https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/)

[8]: [scikit-learn Dataset loading utilities](https://scikit-learn.org/stable/datasets.html))

[9]: [How to Load, Convert, and Save Images With the Keras API](https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/)

[10]: [How to Load and Visualize Standard Computer Vision Datasets With Keras](https://machinelearningmastery.com/how-to-load-and-visualize-standard-computer-vision-datasets-with-keras/

[11]: [Computer Vision: Exploring your image datasets the RIGHT way](https://medium.com/mlearning-ai/computer-vision-exploring-your-image-datasets-the-right-way-538c5ae8ca5d)

[12]: [Document Parsing with Python and OCR](https://towardsdatascience.com/document-parsing-with-python-ocr-75543448e581)

[13]: [What is the difference between Object Localization, Object Recognition, and Object Detection?](https://www.geeksforgeeks.org/computer-vision/what-is-the-difference-between-object-localization-object-recognition-and-object-detection/)

[14]: [Object Recognition](https://www.mathworks.com/solutions/image-processing-computer-vision/object-recognition.html)

[Build a Semantic Segmentation Model With One Line Of Code](https://pub.towardsai.net/build-a-semantic-segmentation-model-with-one-line-of-code-32b6eab0cb81)

[9 Applications of Deep Learning for Computer Vision](https://machinelearningmastery.com/applications-of-deep-learning-for-computer-vision/)
