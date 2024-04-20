# GTSAMLoopClosure_ROS2
## Overview
This repository is a ROS2 implementation of Loop Closure for LiDAR SLAM, modified from [SimpleLoopClosure](https://github.com/kamibukuro5656/SimpleLoopClosure).  
It is a simple implementation using pose graph optimization with GTSAM and radius search with nanoflann.  
It was created to obtain Loop Closed point cloud maps from an algorithm that provides only LiDAR odometry.  

## Build
Please install [GTSAM4.x](https://gtsam.org/get_started/).
~~~
  colcon build
~~~


## Parameters
Please refer to [Parameters.md](Parameters.md).

## Requirements for LiDAR Odometry
- LiDAR odometry must publish "nav_msgs::Odometry" format odometry and a paired LiDAR scan.
- The timestamps of the two topics must be almost the same time.

