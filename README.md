# robotdatapy

`robotdatapy` is a Python package for interfacing with robot data.

## Install

`robotdatapy` is available via PyPi! To install:

```
pip install robotdatapy
```

Or to install the most recent version:

```
git clone git@github.com:mbpeterson70/robotdatapy.git
cd robotdatapy
pip install .
```

## Data Interfaces

The primary use of this package is for interfacing with robot data. 
When developing offline robot applications, it can be difficult to deal with all of the different ways that data can be saved in (ROS bags, csv files, individual images, etc.). 
The goal of this package is to provide classes for loading data from a variety of sources, enabling a downstream task to use these data interfaces without needing to account for where that data is coming from. 

Additionally, when dealing with offline data, a user may want to get a camera image at a certain time as well as a robot pose estimate at that same time. 
However, pose estimates are often discrete and may not be synced with the camera image. 
This package provides a way of dealing with time synchronization between data via interpolation or finding the nearest datapoint to a requested timestamp.  

This README briefly describes three robot data classes: `PoseData`, `ImgData`, and `ArrayData`. See the [examples folder](./examples/) for example Python notebooks of interacting with different robot data.

### PoseData

PoseData can load pose information from ROS1/2 bags, csv files, KITTI, or directly from a set of times and poses. 
Interpolation between poses is enabled by default, making it easy to get positions and orientations of a robot body at any time.
Additionally, a transformation can be specified to be pre-multiplied or post-multiplied (via the `T_premultiply` or `T_postmultiply` keyword argument) changing the reference frame given by the PoseData object.

### ImgData

ImgData can be loaded from ROS1/2 bags, a zipped file of images, a numpy `npz` files of times and images, or directly from a list of times and cv images.
Depth images are supported as well.

### ArrayData

This class can be used for storing generic data. For example, discrete samples of position, velocity, and acceleration.
Linear interpolation can be turned on to enable accessing this data at any time.