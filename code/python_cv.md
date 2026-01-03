# Python CV Code Snippets

Here are some useful Python code snippets for common computer vision (CV) tasks.

## Show samples from each class

```py
    import numpy as np
    import matplotlib.pyplot as plt

    def show_images(num_classes):
        """
        Show image samples from each class
        """
        fig = plt.figure(figsize=(8,3))

        for i in range(num_classes):
            ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
            idx = np.where(y_train[:]==i)[0]
            x_idx = X_train[idx,::]
            img_num = np.random.randint(x_idx.shape[0])
            im = np.transpose(x_idx[img_num,::], (1, 2, 0))
            ax.set_title(class_names[i])
            plt.imshow(im)

        plt.show()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_train, img_channels, img_rows, img_cols =  X_train.shape
    num_test, _, _, _ =  X_train.shape
    num_classes = len(np.unique(y_train))

    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

    show_images(num_classes)
```

## Display multiple images in one figure

```py
    # import libraries
    import cv2
    from matplotlib import pyplot as plt

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    num_rows = 2
    num_cols = 2

    # Read the images into list
    images = []
    img = cv2.imread('Image1.jpg')
    images.append(img)

    img = cv2.imread('Image2.jpg')
    images.append(img)

    img = cv2.imread('Image3.jpg')
    images.append(img)

    img = cv2.imread('Image4.jpg')
    images.append(img)


    # Adds a subplot at the 1st position
    fig.add_subplot(num_rows, num_cols, 1)

    # showing image
    plt.imshow(Image1)
    plt.axis('off')
    plt.title("First")

    # Adds a subplot at the 2nd position
    fig.add_subplot(num_rows, num_cols, 2)

    # showing image
    plt.imshow(Image2)
    plt.axis('off')
    plt.title("Second")

    # Adds a subplot at the 3rd position
    fig.add_subplot(num_rows, num_cols, 3)

    # showing image
    plt.imshow(Image3)
    plt.axis('off')
    plt.title("Third")

    # Adds a subplot at the 4th position
    fig.add_subplot(num_rows, num_cols, 4)

    # showing image
    plt.imshow(Image4)
    plt.axis('off')
    plt.title("Fourth")
```

## Plot images side by side

```py
    _, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()
```

## Visualize a batch of image data

TODO: Add code sample

## Know your dataset instances

Display the number of instances of each class.

```py
import os
#Give path of folder in which you stored images and annotations
path = r"Your dataset *folder* location"
# Change the directory to path
os.chdir(path)
x=[]
# Spinning through all files
for file in os.listdir():
# Checking for text annotation file
    if file.endswith(".txt"):
        file_path = f"{path}\{file}"
        with open(file_path, 'r') as f:
            for line in f:
                a=line[0]
                x.append(a)
print(x)
#to count instances
from collections import Counter
Counter(x)
```

## Preprocessing of images

Sometimes we have more than class instances, we have may have other objects/things in our image dataset.

Removing these noises and sending them to the model for training improves the performance of the model.

If you run the above code, then you will have your training image in front of you, and your mouse will act as a mask maker.

After clicking and hovering the mouse on an unnecessary object will direct create a mask on that object. I took white color for use case purposes, but you can take any according to your problem.

You can train a separate object detection model for noise, and below that, you can attach this code. At first, the model will detect noise and then the code will mask that bounding box with your desired color.

## Data Augmentation

In every computer vision project, we will want to augment the dataset.

There is a library by TensorFlow known as `ImageDataGenerator` that can help.

## Dataset Creation

We may need images from a webcam but it is hard to click  and save it in the labeled folder for classification or object detection.

This code will automate clicking of images for particular labels and it store them at the proper location.

## Extract areas from an image

We can use many techniques such as pixel measurement and others but we have to do calibration before extracting areas to match the original dimensions and their representations in the image and their ratios.

Many programmers use inbuilt ratios and reference object schemes, but here we try a new way of calculating the calibration factor: We draw a line to do the calibration.

## Code Snippets

Here are some useful code snippets for images.

### Reading an image

```py
  def read_image(path):
      im = cv2.imread(str(path))
      return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
```

### Showing multiple images in a grid

```py
  image_paths = list(Path('./dogs').iterdir())
  images = [read_image(p) for p in image_paths]

  fig = plt.figure(figsize=(20, 20))
  columns = 4
  rows = 4

  pics = []
  for i in range(columns*rows):
      pics.append(fig.add_subplot(rows, columns, i+1,title=image_paths[i].parts[-1].split('.')[0]))
      plt.imshow(images[i])

  plt.show()
```

## Kaggle Download

We can download datasets from Kaggle using an API token.

```bash
  %%capture
  !pip install kaggle

  # upload your tokn
  from google.colab import files
  import time

  uploaded = files.upload()
  time.sleep(3)

  # download directly from kaggle
  !cp kaggle.json ~/.kaggle/
  !chmod 600 ~/.kaggle/kaggle.json
  !kaggle competitions download "dogs-vs-cats"
```

```py
  import zipfile

  # unzip the downloaded folder into a new folder
  data_zip = "/content/dogs-vs-cats.zip"
  data_dir = "./data"
  data_zip_ref = zipfile.ZipFile(data_zip,"r")
  data_zip_ref.extractall(data_dir)

  # unzip the test subfolder
  test_zip = "/content/data/test1.zip"
  test_dir = "./data"
  test_zip_ref = zipfile.ZipFile(test_zip,"r")
  test_zip_ref.extractall(test_

  # unzip the train subfolder
  train_zip = "/content/data/train.zip"
  train_dir = "./data"
  train_zip_ref = zipfile.ZipFile(train_zip,"r")
  train_zip_ref.extractall(train_dir)
```

### Structure and Populate Subfolders

To facilitate the management of the dataset, we create an easy-to-manage folder structure.

The goal is to have a folder called `train` that will contain the subfolders dog and cat which will obviously contain all the images of the respective pets.

The same thing should be done for the validation folder.

```py
  import os
  import glob

  dat_dir = "/content/data"

  # create training dir
  training_dir = os.path.join(data_dir,"training")
  if not os.path.isdir(training_dir):
    os.mkdir(training_dir)

  # create dog in training
  dog_training_dir = os.path.join(training_dir,"dog")
  if not os.path.isdir(dog_training_dir):
    os.mkdir(dog_training_dir)

  # create cat in training
  cat_training_dir = os.path.join(training_dir,"cat")
  if not os.path.isdir(cat_training_dir):
    os.mkdir(cat_training_dir)

  # create validation dir
  validation_dir = os.path.join(data_dir,"validation")
  if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

  # create dog in validation
  dog_validation_dir = os.path.join(validation_dir,"dog")
  if not os.path.isdir(dog_validation_dir):
    os.mkdir(dog_validation_dir)

  # create cat in validation
  cat_validation_dir = os.path.join(validation_dir,"cat")
  if not os.path.isdir(cat_validation_dir):
  os.mkdir(cat_validation_dir)
```

Now we shuffle the data and populate the new  subfolders.

```py
  import shutil

  split_size = 0.80
  cat_imgs_size = len(glob.glob("/content/data/train/cat*"))
  dog_imgs_size = len(glob.glob("/content/data/train/dog*"))

  for i,img in enumerate(glob.glob("/content/data/train/cat*")):
    if i < (cat_imgs_size * split_size):
      shutil.move(img,cat_training_dir)
    else:
      shutil.move(img,cat_validation_dir)

  for i,img in enumerate(glob.glob("/content/data/train/dog*")):
    if i < (dog_imgs_size * split_size):
      shutil.move(img,dog_training_dir)
    else:
      shutil.move(img,dog_validation_dir)
```

### Plot some image examples

```py
  import matplotlib.pyplot as plt
  import numpy as np
  import cv2

  from IPython.core.pylabtools import figsize

  samples_dog = [os.path.join(dog_training_dir,np.random.choice(os.listdir(dog_training_dir),1)[0]) for _ in range(8)]
  samples_cat = [os.path.join(cat_training_dir,np.random.choice(os.listdir(cat_training_dir),1)[0]) for _ in range(8)]

  nrows = 4
  ncols = 4

  fig, ax = plt.subplots(nrows,ncols,figsize = (10,10))
  ax = ax.flatten()

  for i in range(nrows*ncols):
    if i < 8:
      pic = plt.imread(samples_dog[i%8])
      ax[i].imshow(pic)
      ax[i].set_axis_off()
    else:
      pic = plt.imread(samples_cat[i%8])
      ax[i].imshow(pic)
      ax[i].set_axis_off()
      plt.show()
```

## References

[1]: [Working on a Computer Vision Project? These Code Chunks Will Help You](https://pub.towardsai.net/working-on-a-computer-vision-project-these-code-chunks-will-help-you-45756bbe7e65)
