"""
苹果配送机器人系统配置文件 - YOLOv11版本 + Web前端
"""

import torch
import os

class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    
    # ROS配置
    ros_master_uri = "http://localhost:11311"
    ros_ip = "127.0.0.1"
    
    # Web服务器配置
    web_host = "0.0.0.0"
    web_port = 8080
    websocket_port = 8081
    
    # ROS topics
    odom_topic = "/p3dx/odom"
    cmd_vel_topic = "/p3dx/cmd_vel"
    laser_topic = "/p3dx/front_laser/scan"
    speech_topic = "/speech_commands"
    status_topic = "/robot_status"
    camera_topic = "/camera/rgb/image_raw"
    
    # 速度参数
    max_linear_speed = 0.8
    max_angular_speed = 1.5
    
    # 苹果种类与货架映射
    apple_to_shelf = {
        "红富士": "1号货架",
        "黄元帅": "1号货架", 
        "蛇果": "2号货架",
        "国光": "3号货架",
        "青苹果": "4号货架",
        "嘎啦": "5号货架"
    }
    
    # 货架位置
    shelf_locations = {
        "1号货架": (1.0, 3.0),
        "2号货架": (1.0, 1.0),
        "3号货架": (1.0, -1.0),
        "4号货架": (5.0, 4.0),
        "5号货架": (5.0, 1.0),
        "充电站": (-9.0, 7.0),
        "起点": (-5.0, -5.0)
    }
    
    # 视觉识别配置 - YOLOv11版本
    vision_model_path = "models/yolov11_apple_best.pt"
    vision_input_size = (640, 640)  # YOLOv11标准输入尺寸
    apple_classes = ["红富士", "黄元帅", "蛇果", "国光", "青苹果", "嘎啦"]
    confidence_threshold = 0.5
    iou_threshold = 0.45
    
    # 语音识别配置
    speech_timeout = 3
    energy_threshold = 1000
    command_mapping = {
        "红富士": "red_fuji",
        "黄元帅": "yellow_general",
        "蛇果": "snake_fruit",
        "国光": "national_light",
        "青苹果": "green_apple",
        "嘎啦": "gala",
        "开始导航": "start_navigation",
        "返回起点": "return_home",
        "充电": "charge",
        "停止": "stop",
        "识别苹果": "recognize_apple"
    }
    
    # 导航参数
    goal_reached_threshold = 0.5
    obstacle_safety_distance = 0.3
    navigation_timeout = 300
    initial_position = (-8.0, 8.0, 0.0)
    
    # 训练参数
    dueling_dqn_model_path = "models/dueling_dqn_heuristic.pth"
    training_episodes = 1000
    max_episode_steps = 150
    batch_size = 64
    learning_rate = 1e-4
    gamma = 0.99
    
    # 地图参数
    map_resolution = 0.2
    map_size = 100
    map_origin = (-10.0, -10.0)
    
    # Web前端配置
    websocket_reconnect_timeout = 3000
    status_update_interval = 1000  # ms
    max_log_lines = 1000


class SimulationConfig:
    """仿真系统配置"""
    world_path = "~/find-road-robot/catkin_ws/src/multi_robot_scenario/worlds/TD3.world"
    launch_files = {
        "roscore": "cd ~/find-road-robot/catkin_ws && source devel_isolated/setup.bash && roscore",
        "gazebo": "cd ~/find-road-robot/catkin_ws && conda activate pytorch && source devel_isolated/setup.bash && roslaunch multi_robot_scenario TD3_world.launch",
        "robot_rviz": "cd ~/find-road-robot/catkin_ws && conda activate pytorch && source devel_isolated/setup.bash && roslaunch multi_robot_scenario multi_robot_scenario.launch",
        "test_script": "cd ~/find-road-robot/TD3 && conda activate pytorch && source ~/find-road-robot/catkin_ws/devel_isolated/setup.bash && python test_velodyne_td3.py"
    }
    
    # Gazebo模型配置
    robot_model = "p3dx"
    camera_model = "camera"
    laser_model = "front_laser"
    
    # 测试配置
    test_waypoints = [
        (-5.0, -5.0),
        (1.0, 3.0),
        (1.0, 1.0),
        (5.0, 4.0),
        (-9.0, 7.0)
    ]
