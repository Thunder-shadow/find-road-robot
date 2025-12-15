#!/bin/bash

# 清理ROS和Gazebo进程脚本

echo "清理ROS和Gazebo进程..."

# 杀死相关进程
pkill -f roscore
pkill -f roslaunch
pkill -f gazebo
pkill -f rviz
pkill -f python.*ros
pkill -f rosnode
pkill -f rosmaster

# 确保进程被杀死
sleep 2

# 检查是否还有残留
if pgrep -f "ros\|gazebo" > /dev/null; then
    echo "强制杀死残留进程..."
    pkill -9 -f roscore
    pkill -9 -f roslaunch
    pkill -9 -f gazebo
    sleep 1
fi

echo "清理完成！"
