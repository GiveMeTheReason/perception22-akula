# AkulaPackageROS2

### Description

The package launches two separate nodes for Velodyne lidar and Basler camera and implements node for IMU data (disabled for now) and node for encoder values that
are obtained from the main server through UDP socket. To select which nodes to run, modify **Main.launch.py** inside **launch** directory. To setup parameters for
robot state publisher change **akula.urdf** and **akula_general.xacro** files from **model** directory. In case of using IMU extend **imu.yaml** inside the 
**config** directory, if needed, it is read by IMU node. The nodes themself are located in the **src** directory, so, if you want to add new nodes, place them 
there, changing **CMakeLists.txt** file accordingly.

### Running

There are three ways to run the package:
1. The intended way to use Android app but pressing buttons
2. If you want to avoid using app, call services, that are controlled by the main server, manually:

```
systemctl --user start akula_sensors_launcher.service
systemctl --user stop akula_sensors_launcher.service
```
and

```
systemctl --user start akula_sensors_record.service
systemctl --user stop akula_sensors_record.service
```
which can be helpful, if you need to control sensors and recording from the outside of the Akula control system (web server, for example)

3. To avoid all that abstraction completely source ROS environment and launch the package:

```
source path/to/env/install/setup.bash
ros2 launch akula_package Main.launch.py
```
therefore specifying which topics you want to record:

```
ros2 bag record -o bags/save/path topic1 topic2 topic3
```
