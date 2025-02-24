# Computer Vision Tips


## Command-line Snippets

```py
# Check dimensions of images
import cv2
from pathlib import Path

images = Path('path').rglob('*.png')
for img_path in images:
  img = cv2.imread(img_path)
  print(img_path, img.shape)
```

```py
  # check that count of images and labels are equal

  # count all files
  ls path/to/images | wc -l

  # count of PNG files
  find -name *.png -type | wc -l
```


----------



## Video/Image manipulation using ffmpeg

`ffmpeg` is a very useful utility for CV. 

### Check video duration

```bash
  ffmpeg -i file.mp4 2>&1 | grep “Duration”
```

### Convert video format

```bash
  ffmpeg -i video.mp4 video.avi

  # extract audio only
  ffmpeg -i input.mp4 -vn output.mp3
```

### Generate dataset from videos

```bash
  ffmpeg -ss 00:10:00 -i input_video.mp4 -to 00:02:00 -c copy output.mp4

  # video without audio
  ffmpeg -i input_video.mp4 -an -c:v copy output.mp4
```

where

- -ss starting time
- -i  input video
- -to time interval like 2 minutes.
- -c  output codec
- -an make output without audio.

### Generate a sequence of frames

```py
  # generate images from videos for 20 seconds
  ffmpeg -ss 00:32:15 -t 20 -i videos.ts ~/frames/frame%06d.png
  
  # rescale the images
  ffmpeg -ss 00:10:00 -t 20 -i video.ts -vf scale=iw/2:ih output_path/frame%06d.png
  
  ffmpeg -ss 00:10:00 -t 20 -i video.ts -vf scale=960x540 output_path/frame%06d.png
```

### Crop a bounding box of video

```py
  ffmpeg -i input.mp4 -filter:v "crop=w:h:x:y" output.mp4
```

## Stack Videos

- Horizontally
- Vertically
- 2x2 grid stacking with xstack



## Improve Model Performance

[How to Accelerate Computer Vision Model Inference](https://wallarooai.medium.com/how-to-accelerate-computer-vision-model-inference-98ba449c0f53)



## References

[Achieving 95.42% Accuracy on Fashion-Mnist Dataset](https://secantzhang.github.io/blog/deep-learning-fashion-mnist)

