# Python Computer Vision Snippets


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



## References

[1] [Working on a Computer Vision Project? These Code Chunks Will Help You](https://pub.towardsai.net/working-on-a-computer-vision-project-these-code-chunks-will-help-you-45756bbe7e65)

