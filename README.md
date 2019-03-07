# Swap-Cat-and-Human-Faces
![swap sample](https://raw.githubusercontent.com/BruceMacD/Swap-Cat-and-Human-Faces/master/data/results/side_by_side.png)

This is a basic face-swap implementation between a cat and a human using OpenCV. This works as a basic proof-of-concept for face-swapping a human with any mask. The application works by simply matching points from cat facial landmarks to corresponding points in the human face mask. 

Originally I developed this with the intention of creating an app to swap cat and human faces, but I found that it isn't accurate enough for a great app experience. Feel free to implement or extend this code in your own application.

## Usage
./face_swap.py -i <data/human_img.jpg> -i <data/cat_img.jpg>

```
./face_swap.py -i data/headshot.jpg -i data/mimi.jpg
```

## Requirements
* OpenCV v3.0+
* numpy
* dlib (for the landmark detection)
* Python 3

## Sources
Headshot

https://www.pexels.com/photo/adult-attractive-beautiful-beauty-415829/
