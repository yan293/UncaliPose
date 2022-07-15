# UncalibratedMVMPWithTracking

This is the official implementation of our paper **"Multi-View Multi-Person 3D Pose Estimation with Unknown Camera Poses"**.

If our work is helpful to your research, please consider citing:

```
  TO COME...
```

<p align="left">
    <img src="./figure/system_overview.png" alt="system overview"  width="800">
</p>



## Configure Environment

We used anaconda to configure the Python environment.  To configure the environment, run the following command:
```
  pip install -r requirements.txt
```


## Download Data

We have pre-processed the open Campus and Shelf datasets to the required format.  Please download the datasets at: 

* [Processed Campus Data](https://drive.google.com/file/d/1YCh4GHY3vkwKpSZsnj6sx84cmwFN7XaP/view?usp=sharing)

* [Processed Shelf Data](https://drive.google.com/file/d/1_Y9x0L7PF8ll92CySbpSsKXaXpEurnLx/view?usp=sharing).


## Use Code

We use IPython as the entrance of the code for the consideration of readability.  Simply run the Ipython files would work.  If Ipython is not the preference, copying the contents of the Ipython files into self-defined python files would also work.

## Result

### 3D Pose Estimation and 2D Pose Reprojection Visualization

<p align="left">
    <img src="./figure/3d_pose_est_figure.png" alt="3d human pose estimation figure"  width="550">
</p>

### Quantitative Evaluation on Open Datasets

<p align="left">
    <img src="./figure/3d_pose_est_table.png" alt="3d human pose estimation table"  width="500">
</p>
