# groupImg

A python script to organize your images by similarity.

It uses a [k-means](https://en.wikipedia.org/wiki/K-means_clustering) algorithm to separatem them in clusters.

Watch it working below.

[![groupImg](http://img.youtube.com/vi/LgzsJU-b34o/0.jpg)](http://www.youtube.com/watch?v=LgzsJU-b34o)

## How to use

Install the requiriments

```
pip install -r requirements.txt
```

Add the *groupimg* to your scripts folder and give it execute permission.

Call the script passing the image folder you want to organize.

```
groupimg -f /home/user/Pictures/
```

## Parameters

\-f folder where your images are (use absolute path). ```groupimg -f /home/user/Pictures```
\-k number of folders you want to separate your images. ```groupimg -f /home/user/Pictures -k 5```
\-m if you want to move your images instead of just copy them.
\-s if you want the algorithm to consider the size of the images as a feature.

## To Do

Add multiprocessing
Rewrite how the script handles the OS calls (find images, navigate to folders...)
Add new ways to break down the image as features
