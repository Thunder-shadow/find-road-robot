#!/bin/bash

# 苹果配送机器人仿真启动脚本 - Web版本

echo "========================================="
echo "苹果配送机器人仿真系统启动 - Web版本"
echo "========================================="

# 设置环境变量
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=127.0.0.1

# 检查Python依赖
echo "检查Python依赖..."
python3 -c "import torch; import cv2; import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Python依赖检查失败"
    echo "请安装必要的Python包:"
    echo "  pip install torch torchvision opencv-python numpy ultralytics aiohttp websockets"
    exit 1
fi

# 检查YOLO模型
if [ ! -f "models/yolov11_apple_best.pt" ]; then
    echo "⚠️ 未找到YOLOv11模型文件: models/yolov11_apple_best.pt"
    echo "将使用模拟模式进行测试"
fi

# 函数：清理进程
cleanup() {
    echo "清理残留进程..."
    pkill -f roscore
    pkill -f roslaunch
    pkill -f gazebo
    pkill -f rviz
    pkill -f python.*backend
    sleep 2
}

# 函数：检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "警告: $1 未安装，某些功能可能受限"
        return 1
    fi
    return 0
}

# 检查必要命令
check_command roscore
check_command roslaunch
check_command gazebo

# 清理环境
cleanup

echo "步骤 1: 启动 roscore"
cd ~/find-road-robot/catkin_ws
source devel_isolated/setup.bash
gnome-terminal --tab --title="roscore" -- bash -c "roscore; exec bash"
sleep 3

echo "步骤 2: 启动 Gazebo 世界"
cd ~/find-road-robot/catkin_ws
conda activate pytorch
source devel_isolated/setup.bash
gnome-terminal --tab --title="Gazebo" -- bash -c "roslaunch multi_robot_scenario TD3_world.launch; exec bash"
sleep 5

echo "步骤 3: 加载机器人 + RVIZ"
cd ~/find-road-robot/catkin_ws
conda activate pytorch
source devel_isolated/setup.bash
gnome-terminal --tab --title="Robot+RVIZ" -- bash -c "roslaunch multi_robot_scenario multi_robot_scenario.launch; exec bash"
sleep 3

echo "步骤 4: 运行测试脚本"
cd ~/find-road-robot/TD3
conda activate pytorch
source ~/find-road-robot/catkin_ws/devel_isolated/setup.bash
gnome-terminal --tab --title="Test Script" -- bash -c "python test_velodyne_td3.py; exec bash"

echo "步骤 5: 启动Web服务器"
cd $(dirname "$0")/..
conda activate pytorch
gnome-terminal --tab --title="Web Server" -- bash -c "python backend.py; exec bash"

echo "========================================="
echo "Web版本系统启动完成！"
echo "访问地址: http://localhost:8080"
echo ""
echo "YOLOv11版本特性:"
echo "  - 使用YOLOv11进行苹果检测"
echo "  - 支持多目标同时识别"
echo "  - Web界面实时控制"
echo "  - 可视化结果显示"
echo "========================================="

# 打开浏览器
if command -v xdg-open &> /dev/null; then
    echo "正在打开浏览器..."
    xdg-open http://localhost:8080 2>/dev/null &
elif command -v open &> /dev/null; then
    echo "正在打开浏览器..."
    open http://localhost:8080 2>/dev/null &
fi

# 等待用户输入
read -p "按回车键停止仿真并清理进程..."

echo "停止仿真系统..."
cleanup
echo "仿真系统已停止"
