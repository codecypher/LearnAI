#!/usr/bin/env python3
"""
  cv_img_aug.py
  SVC classifier example using dogs_cats sample dataset
  
  Code samples performing image augmentation using OpenCV

  References:
  [Recognizing hand-written digits](https://scikit-learn.org/)
  [Image Processing in Python](https://www.geeksforgeeks.org/)
  [Image Processing and Data Augmentation Techniques for Computer Vision](https://towardsdatascience.com)
  [Image Augmentation with skimage](https://towardsdatascience.com)
  [Random Numbers in Python](https://www.geeksforgeeks.org/)

  Jeff Holmes
  02/11/2022
"""
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_datareader as pdr
import pandas_datareader.data as web
import matplotlib.pyplot as plt

import datetime as dt
import json
import math
import os
import random
import sys

from PIL import Image
from scipy import ndimage
from skimage import color

from sys import exit
from time import time


from sklearn import datasets, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def read_image(path):
    """
    Read an image
    """
    try:
        # Read image from dis
        img = cv2.imread(path)

        # Shape of image in terms of pixels
        (height, width) = img.shape[:2]

        # Write image back to disk.
        # cv2.imwrite('result.jpg', res)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except IOError:
        print ('Error while reading file!')


def center_crop(img, crop_pixels = 50):
    # Shape of image in terms of pixels
    (height, width) = img.shape[:2]
    cropped = img[crop_pixels:im.shape[0] - crop_pixels, crop_pixels:img.shape[1] - crop_pixels]
    return cropped


def resize(img, width, height):
    # Shape of image in terms of pixels
    (height, width) = img.shape[:2]

    # Specify the size of image along with interploation methods.
    # cv2.INTER_AREA is used for shrinking, whereas cv2.INTER_CUBIC is used for zooming
    resized = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC)

    return resized


def flip_vertical(img):
    # Shape of image in terms of pixels
    (height, width) = img.shape[:2]
    flip_v = np.flip(img, 0)
    return flip_v


def flip_horizontal(img):
    # Shape of image in terms of pixels
    (height, width) = img.shape[:2]
    flip_h = np.flip(img, 1)
    return flip_h


def rotate(img, angle):
    # Shape of image in terms of pixels
    (rows, cols) = img.shape[:2]

    # Calculates an affine matrix of 2D rotation
    # getRotationMatrix2D creates a matrix needed for transformation.
    # We want matrix for rotation w.r.t center to angle degrees without scaling.
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))

    return rotated


def translate(img, x, y):
    # Create translation matrix.
    # If the shift is (x, y) then matrix would be
    # T = [1 0 x]
    #     [0 1 y]
    # Shift by (100, 50)
    T = np.float32([[1, 0, x], [0, 1, y]])

    (rows, cols) = img.shape[:2]

    # warpAffine does appropriate shifting given the translation matrix.
    translated = cv2.warpAffine(img, T, (cols, rows))

    return translated


def blur(img, ksize):
    blurred = cv2.blur(img, (ksize, ksize))
    return blurred


def grey_scale(img):
    """
    Convert color images to greyscale
    """
    grey = color.rgb2gray(img)
    return grey


def bright_contrast(img, alpha=1, beta=0):
    """
    Change brightness and contrast
    Args:
        alpha (int): gain which controls contrast
        beta  (int): bias which controls brightness
    """
    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_image


def edge_detection(img):
    """
    The process of image detection involves detecting sharp edges in the image.
    This edge detection is essential in context of image recognition
    or object localization/detection. There are several algorithms for detecting edges.
    Here we use Canny Edge Detection.
    """
    # Create translation matrix.
    # If the shift is (x, y) then matrix would be
    # M = [1 0 x]
    #     [0 1 y]
    # Let's shift by (100, 50).
    M = np.float32([[1, 0, 100], [0, 1, 50]])

    (rows, cols) = img.shape[:2]

    # warpAffine does appropriate shifting given the translation matrix
    res = cv2.warpAffine(img, M, (cols, rows))


def generate_images(folder):
    """
    Generate augmented images using the images in the given folder

    Image augmentation can be applied mainly on two domains:

      1. Position Augmentation
      2. Color Augmentation

    The most popular data augmentation used in deep learning:

      1. Random rotation
      2. Flip (horizontal and vertical)
      3. Zoom
      4. Random shift
      5. Brightness
    """
    for root, dirs, files in os.walk(folder):
        for filename in files:
            basename, extension = os.path.splitext(filename)
            if extension == '.jpg':
                file_path = os.path.join(root, filename)

                # Read image object
                img = read_image(file_path)

                # Resize image
                img = resize(img, 500, 500)
                plt.imshow(img)

                # Rotate image
                angles = [15, 30, 45, 60, 75, 90]
                for theta in angles:
                    new_file = basename + "_rotate_" + theta + extension
                    rotated = rotate(img, theta)
                    cv2.imwrite(new_file, rotated)


def load_images(folder):
    """
    Load images from folders
    """
    img_list = []
    target_list = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            basename, extension = os.path.splitext(filename)
            if extension == '.jpg':
                file_path = os.path.join(root, filename)

                # read image object
                img = Image.open(file_path)

                # display some details about the image
                # print(img.format)
                # print(img.size)
                # print(img.mode)

                newsize = (500, 500)
                img = img.resize(newsize)

                # asarray() is used to convert PIL image to numpy array
                np_img = np.asarray(img)
                img_list.append(np_img)

                idx = root.rindex("/")  # find last occurence
                folder_name = root[idx+1:]

                if folder_name == 'cats':
                  target = 0
                else:
                  target = 1
                target_list.append(target)

    np_data = np.array(img_list)
    np_target = np.array(target_list)
    print(np_data.shape)
    print(np_target.shape)

    return np_data, np_target


def show_image(img, grey=False):
    if grey:
        plt.imshow(img,cmap = plt.cm.gray)
    else:
        plt.imshow(img)
    plt.show()


def random_snippets():
    """
    Code snippets using random numbers
    """
    # using seed() to seed a random number so that
    # same random numbers on each execution
    random.seed(5)

    # Generate a real (float) random number between 0 and 1
    print("A random number between 0 and 1 is : ", end="")
    print(random.random())

    # Generate a random number between a given positive range
    r1 = random.randint(5, 15)
    print("Random number between 5 and 15 is % s" % (r1))

    # Generate a random number between two given negative range
    r2 = random.randint(-10, -2)
    print("Random number between -10 and -2 is % d" % (r2))

    # Generate a random number from a given list of numbers.
    print("A random number from list is : ", end="")
    print(random.choice([1, 4, 8, 10, 3]))

    # Generate in range from 20 to 50.
    # The last parameter 3 is step size to skip three numbers when selecting.
    print("A random number from range is : ", end="")
    print(random.randrange(20, 50, 3))


def apply_transforms():
    """
    Perform multiple image augmentations
    
    See generate_images() for saving images
    """
    file_path = "./data/dogs_cats/dogs/dog.0.jpg"

    # Read image object
    img = read_image(file_path)

    resized = resize(img, 500, 500)
    rotated = rotate(img, 30)
    flip_v = flip_vertical(img)
    flip_h = flip_horizontal(img)
    shifted = translate(img, 50, 50)
    blurred = blur(img, 5)
    grey = grey_scale(img)

    res = bright_contrast(img, alpha=1.5)
    res = bright_contrast(img, beta=50)



def main():
    images, target = load_images("./data/dogs_cats")

    # flatten the images
    n_samples = len(images)
    data = images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)

    # Display sample of y_train
    # for i, y in enumerate(y_train):
    #     if i < 10:
    #         print(i, y)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the image on the test dataset
    predicted = clf.predict(X_test)

    acc = accuracy_score(y_test, predicted)
    print("Accuracy: ", acc)

    # Visualize the first 8 (ncols) test samples and show predicted value in the title
    # _, axes = plt.subplots(nrows=1, ncols=8, figsize=(12, 4))
    # print(axes.shape, X_test.shape, predicted.shape)
    # for ax, image, prediction in zip(axes, X_test, predicted):
    #     ax.set_axis_off()
    #     image = image.reshape(500, 500, 3)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     ax.set_title(f'Prediction: {prediction}')
    # plt.show()


# Check that code is under main function
if __name__ == "__main__":
    apply_transforms()
    # generate_images()
    # main()
    print('Done')
