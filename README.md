# UncaliPose

This is the official implementation of **"Multi-View Multi-Person 3D Pose Estimation with Unknown Camera Poses"**.

## System overview:

Given multi-view images, (a) we first detect 2D poses and solve cross-view human matching as constrained optimization.  Then, (b) we estimate camera poses and perform self-validation.  Finally, (c) we solve 3D human poses, aggregate multi-view information, and further optimize through (d) Bundle Adjustment.

<p align="left">
    <img src="./figure/system_overview.png" alt="system overview"  width="800">
</p>

## Usage of code and data

### Environment configuration

To configure the environment, simply run the following command:
```
  pip install -r requirements.txt
```


### Downloading pre-processed data

We have pre-processed the open Campus and Shelf datasets to the following required format.  The processed datasets can be downloaded at: 

* [Processed Campus Data](https://drive.google.com/file/d/1YCh4GHY3vkwKpSZsnj6sx84cmwFN7XaP/view?usp=sharing)

* [Processed Shelf Data](https://drive.google.com/file/d/1_Y9x0L7PF8ll92CySbpSsKXaXpEurnLx/view?usp=sharing).


### Running code

We use IPython as the entrance of the code for the consideration of readability.  Simply run the Ipython files would work.  If Ipython is not your preference, copying the contents of the Ipython files into self-defined python files should also work.

## Result demo

### Multi-view 3D human pose pstimation without knowing the camera poses

<p align="left">
    <img src="./figure/human_pose_estimation.png" alt="3d human pose estimation figure"  width="550">
</p>

### Camera pose estimation

<p align="left">
    <img src="./figure/camera_pose_estimation.png" alt="3d human pose estimation table"  width="500">
</p>

<!-- ### Multi-view 3D human pose pstimation with one moving camera mounted on a flying drone

<p align="left">
    <img src="./figure/drone_pose_estimation.png" alt="3d human pose estimation table"  width="500">
</p> -->


## Citation
If you think our work is helpful, please consider citing:

```
@inproceedings{Xu_2022_BMVC,
author    = {Yan Xu and Kris Kitani},
title     = {Multi-View Multi-Person 3D Pose Estimation with Uncalibrated Camera Networks},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0132.pdf}
}
```
