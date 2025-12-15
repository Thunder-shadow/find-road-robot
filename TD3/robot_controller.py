"""
机器人主控制器 - YOLOv11版本
整合YOLOv11视觉识别、语音控制和导航训练
"""

import os
import sys
import time
import math
import threading
import queue
import json
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
from dataclasses import dataclass

import rospy
import numpy as np
import torch
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from squaternion import Quaternion as Squaternion

from config import Config
from vision_recognizer import AppleVisionRecognizer
from navigation_trainer import DuelingDQNAStarTrainer, HeuristicLearner, OccupancyGridMap


class OperatingMode(Enum):
    """操作模式"""
    VOICE_CONTROL = "语音控制"
    VISION_CONTROL = "视觉控制"
    TRAINING = "训练模式"
    NAVIGATION = "导航模式"
    IDLE = "空闲"


@dataclass
class RobotStatus:
    """机器人状态"""
    position: Tuple[float, float, float]  # x, y, theta
    velocity: Tuple[float, float]  # linear, angular
    mode: OperatingMode
    battery_level: float
    current_task: str
    vision_recognition: Optional[Dict] = None
    navigation_target: Optional[Tuple[float, float]] = None
    navigation_progress: float = 0.0
    obstacles_detected: List[float] = None
    last_update: float = 0.0


class LocalMicrophoneVoiceControl:
    """本地麦克风语音控制"""
    def __init__(self):
        self.command_queue = queue.Queue()
        self.listening = False
        self.recognizer = None
        self.microphone = None
        
        self._initialize_recognizer()
        
    def _initialize_recognizer(self):
        """初始化语音识别器"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            
            # 尝试获取麦克风
            mic_list = sr.Microphone.list_microphone_names()
            if mic_list:
                self.microphone = sr.Microphone()
                print(f"找到麦克风设备: {len(mic_list)}个")
            else:
                print("⚠️ 未找到麦克风设备")
                self.microphone = None
                return
            
            # 调整环境噪声
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("✅ 语音识别器初始化成功")
        except Exception as e:
            print(f"⚠️ 语音识别器初始化失败: {e}")
            self.recognizer = None
    
    def start_listening(self):
        """开始监听"""
        if self.listening or not self.recognizer or not self.microphone:
            return False
        
        self.listening = True
        self.listener_thread = threading.Thread(
            target=self._listening_loop, daemon=True)
        self.listener_thread.start()
        return True
    
    def _listening_loop(self):
        """监听循环"""
        import speech_recognition as sr
        
        while self.listening:
            try:
                with self.microphone as source:
                    print("🎤 正在聆听...")
                    audio = self.recognizer.listen(
                        source, 
                        timeout=Config.speech_timeout,
                        phrase_time_limit=5
                    )
                    
                    text = self.recognizer.recognize_google(audio, language='zh-CN')
                    if text:
                        print(f"🗣️ 识别到语音: {text}")
                        self._process_speech(text)
                        
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("⚠️ 未能识别语音内容")
            except sr.RequestError as e:
                print(f"⚠️ 语音识别服务错误: {e}")
            except Exception as e:
                print(f"语音识别错误: {e}")
    
    def _process_speech(self, text: str):
        """处理语音"""
        # 简单关键词匹配
        for keyword, command in Config.command_mapping.items():
            if keyword in text:
                print(f"✅ 匹配命令: {keyword} -> {command}")
                self.command_queue.put((keyword, command))
                return
        
        # 模糊匹配
        matched = self._fuzzy_match(text)
        if matched:
            print(f"✅ 模糊匹配: {matched}")
            self.command_queue.put((matched, Config.command_mapping.get(matched, "")))
        else:
            print(f"⚠️ 未识别命令: {text}")
    
    def _fuzzy_match(self, text: str) -> Optional[str]:
        """模糊匹配命令"""
        import difflib
        
        commands = list(Config.command_mapping.keys())
        matches = difflib.get_close_matches(text, commands, n=1, cutoff=0.6)
        return matches[0] if matches else None
    
    def get_command(self, timeout=0.1):
        """获取命令"""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_listening(self):
        """停止监听"""
        self.listening = False


class NavigationController:
    """导航控制器"""
    def __init__(self, cmd_vel_publisher):
        self.cmd_vel_pub = cmd_vel_publisher
        
        # 导航状态
        self.navigating = False
        self.current_target = None
        self.current_target_name = None
        self.waypoints = []
        self.current_waypoint = 0
        
        # 控制参数
        self.kp_linear = 1.2
        self.kp_angular = 2.0
        self.safe_distance = Config.obstacle_safety_distance
        
        # 路径规划器
        self.occupancy_map = OccupancyGridMap(
            resolution=Config.map_resolution,
            size=Config.map_size,
            origin=Config.map_origin
        )
        self.heuristic_learner = None
        
        # 加载预训练模型
        if os.path.exists(Config.dueling_dqn_model_path):
            self.heuristic_learner = HeuristicLearner()
            self.heuristic_learner.load(Config.dueling_dqn_model_path)
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple]:
        """规划路径"""
        if self.heuristic_learner:
            # 使用学习启发式的A*算法
            return self.a_star_with_learned_heuristic(start, goal)
        else:
            # 简单直线路径
            return [goal]
    
    def a_star_with_learned_heuristic(self, start_world, goal_world):
        """使用学习启发式的A*算法"""
        start_cell = self.occupancy_map.world_to_grid(*start_world)
        goal_cell = self.occupancy_map.world_to_grid(*goal_world)
        
        # 简单A*实现
        open_set = [(0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self.heuristic_cost(start_cell, goal_cell)}
        
        while open_set:
            open_set.sort()
            current_f, current = open_set.pop(0)
            
            if current == goal_cell:
                # 重建路径
                path = self.reconstruct_path(came_from, current, start_cell)
                world_path = [self.occupancy_map.grid_to_world(i, j) for i, j in path]
                return world_path
            
            # 探索邻居
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_cell(*neighbor):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_cost(neighbor, goal_cell)
                    
                    if neighbor not in [cell for _, cell in open_set]:
                        open_set.append((f_score[neighbor], neighbor))
        
        return []
    
    def heuristic_cost(self, cell, goal):
        """启发式代价"""
        if self.heuristic_learner:
            # 使用学习启发式
            state = self.heuristic_learner.create_state(cell, goal, self.occupancy_map)
            return self.heuristic_learner.predict_heuristic(state)
        else:
            # 欧几里得距离
            return math.sqrt((cell[0]-goal[0])**2 + (cell[1]-goal[1])**2)
    
    def is_valid_cell(self, grid_x, grid_y):
        """检查单元格是否有效"""
        if not (0 <= grid_x < self.occupancy_map.size and 
                0 <= grid_y < self.occupancy_map.size):
            return False
        return not self.occupancy_map.is_occupied(grid_x, grid_y)
    
    def reconstruct_path(self, came_from, current, start):
        """重建路径"""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path
    
    def start_navigation(self, target_position: Tuple[float, float], target_name: str):
        """开始导航"""
        if self.navigating:
            self.stop_navigation()
        
        self.current_target = target_position
        self.current_target_name = target_name
        self.navigating = True
        
        # 规划路径
        start_pos = self.get_current_position()  # 需要从外部获取
        self.waypoints = self.plan_path(start_pos, target_position)
        self.current_waypoint = 0
        
        print(f"🗺️ 开始导航到 {target_name}")
        print(f"路径点: {len(self.waypoints)} 个")
        
        return True
    
    def get_current_position(self):
        """获取当前位置（简化，实际应从外部获取）"""
        # 这里应该返回从ROS获取的实际位置
        return (0.0, 0.0)
    
    def stop_navigation(self):
        """停止导航"""
        self.navigating = False
        self.current_target = None
        self.current_target_name = None
        self.waypoints = []
        self.current_waypoint = 0
        
        # 停止机器人
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        
        print("🛑 导航已停止")
    
    def navigate_step(self, robot_position: Tuple[float, float, float], 
                     laser_data: Optional[LaserScan] = None) -> bool:
        """执行一步导航"""
        if not self.navigating or not self.current_target:
            return False
        
        robot_x, robot_y, robot_theta = robot_position
        
        # 检查是否到达目标
        distance_to_target = math.hypot(
            robot_x - self.current_target[0],
            robot_y - self.current_target[1]
        )
        
        if distance_to_target < Config.goal_reached_threshold:
            print(f"🎉 到达目标 {self.current_target_name}!")
            self.stop_navigation()
            return True
        
        # 获取当前路径点
        if self.current_waypoint < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint]
        else:
            waypoint = self.current_target
        
        # 计算控制命令
        dx = waypoint[0] - robot_x
        dy = waypoint[1] - robot_y
        target_angle = math.atan2(dy, dx)
        
        # 角度误差
        angle_error = target_angle - robot_theta
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        
        # 距离误差
        distance_error = math.hypot(dx, dy)
        
        # 避障检查
        obstacle_detected = False
        if laser_data:
            ranges = np.array(laser_data.ranges)
            ranges = np.nan_to_num(ranges, nan=laser_data.range_max)
            if np.any(ranges < self.safe_distance):
                obstacle_detected = True
        
        # 计算控制量
        if obstacle_detected:
            # 避障行为
            linear_speed = -0.1
            angular_speed = self.kp_angular * 0.5
            if angle_error > 0:
                angular_speed = angular_speed
            else:
                angular_speed = -angular_speed
        else:
            # 正常导航
            linear_speed = min(Config.max_linear_speed, 
                             distance_error * self.kp_linear)
            angular_speed = max(-Config.max_angular_speed,
                              min(Config.max_angular_speed,
                                  angle_error * self.kp_angular))
            
            # 接近路径点时减速
            if distance_error < 0.5:
                linear_speed = linear_speed * (distance_error / 0.5)
        
        # 发布控制命令
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_vel_pub.publish(twist)
        
        # 检查是否到达当前路径点
        if distance_error < 0.3 and self.current_waypoint < len(self.waypoints) - 1:
            self.current_waypoint += 1
            print(f"前往下一个路径点 {self.current_waypoint}/{len(self.waypoints)}")
        
        return False


class AppleDeliveryRobotController:
    """苹果配送机器人主控制器 - YOLOv11版本"""
    
    def __init__(self):
        print("=" * 70)
        print("🍎 苹果配送机器人系统 - YOLOv11集成版")
        print("=" * 70)
        
        # 初始化ROS
        try:
            rospy.init_node('apple_delivery_controller', anonymous=True)
        except:
            pass
        
        # 传感器数据
        self.odom_data = None
        self.laser_data = None
        self.robot_pose = Config.initial_position
        
        # ROS发布器/订阅器
        self.cmd_vel_pub = rospy.Publisher(Config.cmd_vel_topic, Twist, queue_size=10)
        self.status_pub = rospy.Publisher(Config.status_topic, String, queue_size=10)
        
        # 订阅器
        self.odom_sub = rospy.Subscriber(Config.odom_topic, Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber(Config.laser_topic, LaserScan, self.laser_callback)
        
        # 初始化组件 - YOLOv11版本
        self.vision_recognizer = AppleVisionRecognizer()
        self.voice_control = LocalMicrophoneVoiceControl()
        self.navigation = NavigationController(self.cmd_vel_pub)
        self.navigation_trainer = DuelingDQNAStarTrainer()
        
        # 机器人状态
        self.status = RobotStatus(
            position=Config.initial_position,
            velocity=(0.0, 0.0),
            mode=OperatingMode.IDLE,
            battery_level=100.0,
            current_task="初始化",
            obstacles_detected=[]
        )
        
        # 控制标志
        self.running = True
        self.paused = False
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.task_thread = threading.Thread(target=self._task_processor, daemon=True)
        self.task_thread.start()
        
        # 状态更新线程
        self.status_thread = threading.Thread(target=self._status_updater, daemon=True)
        self.status_thread.start()
        
        # 等待传感器数据
        self._wait_for_sensors()
        
        print("✅ 机器人控制器初始化完成")
        self._print_instructions()
    
    def odom_callback(self, msg):
        """里程计回调"""
        self.odom_data = msg
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            heading = Squaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)[2]
            
            # 计算速度
            linear_x = msg.twist.twist.linear.x
            angular_z = msg.twist.twist.angular.z
            
            self.robot_pose = (x, y, heading)
            
            # 更新状态
            self.status.position = (x, y, heading)
            self.status.velocity = (linear_x, angular_z)
            
        except Exception as e:
            print(f"里程计解析错误: {e}")
    
    def laser_callback(self, msg):
        """激光雷达回调"""
        self.laser_data = msg
        
        # 更新障碍物检测
        if msg:
            ranges = np.array(msg.ranges)
            ranges = np.nan_to_num(ranges, nan=msg.range_max)
            close_obstacles = ranges[ranges < Config.obstacle_safety_distance]
            self.status.obstacles_detected = close_obstacles.tolist()
    
    def get_robot_pose(self):
        """获取机器人位姿"""
        return self.robot_pose
    
    def get_laser_data(self):
        """获取激光雷达数据"""
        return self.laser_data
    
    def _wait_for_sensors(self):
        """等待传感器数据"""
        print("等待传感器数据...")
        timeout = 30.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.odom_data is not None and self.laser_data is not None:
                print("✅ 传感器数据接收成功")
                return True
            
            print(f"等待中... (已等待{time.time() - start_time:.1f}秒)")
            time.sleep(1)
        
        print("⚠️ 传感器数据等待超时，使用模拟数据继续")
        return False
    
    def _print_instructions(self):
        """打印使用说明"""
        print("\n" + "=" * 70)
        print("YOLOv11版本使用说明:")
        print("=" * 70)
        print("视觉识别功能:")
        print("  - 使用YOLOv11模型检测苹果")
        print("  - 支持多目标同时检测")
        print("  - 自动识别苹果种类并导航到对应货架")
        print()
        print("语音命令:")
        for apple in Config.apple_to_shelf.keys():
            shelf = Config.apple_to_shelf[apple]
            print(f"  '{apple}' -> 导航到{shelf}")
        print()
        print("系统命令:")
        print("  '开始导航' - 开始配送任务")
        print("  '返回起点' - 返回起始点")
        print("  '充电' - 前往充电站")
        print("  '停止' - 停止当前导航")
        print("  '识别苹果' - 进行苹果识别")
        print("=" * 70)
    
    def start_voice_control(self):
        """开始语音控制"""
        if self.voice_control.start_listening():
            self.status.mode = OperatingMode.VOICE_CONTROL
            self.status.current_task = "语音控制模式"
            print("✅ 语音控制已启动")
            return True
        return False
    
    def stop_voice_control(self):
        """停止语音控制"""
        self.voice_control.stop_listening()
        if self.status.mode == OperatingMode.VOICE_CONTROL:
            self.status.mode = OperatingMode.IDLE
        print("🛑 语音控制已停止")
    
    def recognize_apple_from_image(self, image_path: str):
        """从图像识别苹果 - YOLOv11版本"""
        print(f"识别图像: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return None
        
        try:
            # 识别苹果 - YOLOv11版本
            apple_class, confidence, details = self.vision_recognizer.recognize(image_path)
            
            # 获取检测数量
            num_detections = details.get('num_detections', 0)
            
            # 更新状态
            self.status.vision_recognition = {
                'class': apple_class,
                'confidence': confidence,
                'image_path': image_path,
                'detections': details.get('detections', []),
                'num_detections': num_detections,
                'class_distribution': details.get('class_distribution', {}),
                'timestamp': time.time()
            }
            
            print(f"识别结果: {apple_class} (置信度: {confidence:.2%})")
            print(f"检测到 {num_detections} 个苹果")
            
            # 显示各类别统计
            if 'class_distribution' in details:
                for cls_name, count in details['class_distribution'].items():
                    print(f"  {cls_name}: {count}个")
            
            # 如果是苹果，获取对应的货架
            if (apple_class in Config.apple_to_shelf and 
                apple_class not in ["未检测到", "识别错误", "背景"]):
                shelf = Config.apple_to_shelf[apple_class]
                print(f"对应货架: {shelf}")
                
                # 添加到任务队列
                self.add_task({
                    'type': 'navigate_to_shelf',
                    'shelf': shelf,
                    'apple': apple_class,
                    'confidence': confidence,
                    'num_detections': num_detections
                })
            
            return apple_class, confidence, details
            
        except Exception as e:
            print(f"❌ 识别失败: {e}")
            return None
    
    def navigate_to_shelf(self, shelf_name: str, apple_type: str = None):
        """导航到指定货架"""
        if shelf_name not in Config.shelf_locations:
            print(f"❌ 未知货架: {shelf_name}")
            return False
        
        target_position = Config.shelf_locations[shelf_name]
        
        print(f"🚀 导航到 {shelf_name}")
        if apple_type:
            print(f"苹果类型: {apple_type}")
        
        # 开始导航
        success = self.navigation.start_navigation(target_position, shelf_name)
        
        if success:
            self.status.mode = OperatingMode.NAVIGATION
            self.status.current_task = f"导航到{shelf_name}"
            self.status.navigation_target = target_position
            return True
        
        return False
    
    def return_to_start(self):
        """返回起点"""
        return self.navigate_to_shelf("起点")
    
    def go_to_charging(self):
        """前往充电站"""
        return self.navigate_to_shelf("充电站")
    
    def stop_navigation(self):
        """停止导航"""
        self.navigation.stop_navigation()
        if self.status.mode == OperatingMode.NAVIGATION:
            self.status.mode = OperatingMode.IDLE
        self.status.current_task = "空闲"
        print("🛑 导航已停止")
    
    def start_training(self, episodes: int = 10):
        """开始训练"""
        print(f"开始训练 {episodes} 回合")
        
        self.status.mode = OperatingMode.TRAINING
        self.status.current_task = f"训练模式 ({episodes}回合)"
        
        # 在单独的线程中运行训练
        def training_thread():
            self.navigation_trainer.train(episodes=episodes)
            self.status.mode = OperatingMode.IDLE
            self.status.current_task = "训练完成"
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
        
        return True
    
    def add_task(self, task: Dict):
        """添加任务到队列"""
        self.task_queue.put(task)
    
    def _task_processor(self):
        """任务处理器"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
                
                if task['type'] == 'navigate_to_shelf':
                    self.navigate_to_shelf(
                        task['shelf'],
                        task.get('apple')
                    )
                
                elif task['type'] == 'return_home':
                    self.return_to_start()
                
                elif task['type'] == 'charge':
                    self.go_to_charging()
                
                elif task['type'] == 'stop':
                    self.stop_navigation()
                
                elif task['type'] == 'recognize_apple':
                    image_path = task.get('image_path')
                    if image_path:
                        self.recognize_apple_from_image(image_path)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"任务处理错误: {e}")
    
    def _status_updater(self):
        """状态更新器"""
        while self.running:
            try:
                # 更新导航进度
                if self.status.mode == OperatingMode.NAVIGATION:
                    if self.status.navigation_target:
                        robot_x, robot_y, _ = self.status.position
                        target_x, target_y = self.status.navigation_target
                        
                        total_distance = math.hypot(
                            Config.initial_position[0] - target_x,
                            Config.initial_position[1] - target_y
                        )
                        current_distance = math.hypot(
                            robot_x - target_x,
                            robot_y - target_y
                        )
                        
                        if total_distance > 0:
                            progress = max(0, min(1, 1 - (current_distance / total_distance)))
                            self.status.navigation_progress = progress
                
                # 电池模拟消耗
                if self.status.mode != OperatingMode.IDLE:
                    self.status.battery_level = max(0, self.status.battery_level - 0.001)
                
                # 更新最后更新时间
                self.status.last_update = time.time()
                
                # 发布状态消息
                status_msg = String()
                status_data = {
                    'mode': self.status.mode.value,
                    'position': self.status.position,
                    'battery': self.status.battery_level,
                    'task': self.status.current_task,
                    'navigation_target': self.status.navigation_target,
                    'navigation_progress': self.status.navigation_progress,
                    'timestamp': self.status.last_update
                }
                status_msg.data = json.dumps(status_data)
                self.status_pub.publish(status_msg)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"状态更新错误: {e}")
    
    def process_voice_commands(self):
        """处理语音命令"""
        command = self.voice_control.get_command()
        if command:
            keyword, cmd_type = command
            
            if keyword in Config.apple_to_shelf:
                shelf = Config.apple_to_shelf[keyword]
                self.add_task({
                    'type': 'navigate_to_shelf',
                    'shelf': shelf,
                    'apple': keyword
                })
            
            elif cmd_type == 'start_navigation':
                # 开始配送任务
                print("开始配送任务")
                self.status.current_task = "配送任务开始"
            
            elif cmd_type == 'return_home':
                self.add_task({'type': 'return_home'})
            
            elif cmd_type == 'charge':
                self.add_task({'type': 'charge'})
            
            elif cmd_type == 'stop':
                self.add_task({'type': 'stop'})
            
            elif cmd_type == 'recognize_apple':
                print("请选择图像进行识别")
    
    def run(self):
        """运行主循环"""
        print("\n🚀 机器人系统启动！")
        
        try:
            # 启动语音控制
            self.start_voice_control()
            
            while self.running and not rospy.is_shutdown():
                if not self.paused:
                    # 处理语音命令
                    self.process_voice_commands()
                    
                    # 执行导航步骤
                    if self.status.mode == OperatingMode.NAVIGATION:
                        self.navigation.navigate_step(
                            self.get_robot_pose(),
                            self.get_laser_data()
                        )
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 系统被中断")
        except Exception as e:
            print(f"\n❌ 系统错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        self.running = False
        
        self.stop_voice_control()
        self.stop_navigation()
        
        # 停止机器人
        try:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
        except:
            pass
        
        print("✅ 清理完成")


# 测试函数
def test_robot_controller():
    """测试机器人控制器"""
    print("测试机器人控制器 - YOLOv11版本...")
    
    controller = AppleDeliveryRobotController()
    
    # 模拟测试
    print("1. 测试状态获取...")
    print(f"当前模式: {controller.status.mode}")
    print(f"当前位置: {controller.status.position}")
    print(f"当前任务: {controller.status.current_task}")
    
    print("2. 测试导航任务...")
    controller.add_task({
        'type': 'navigate_to_shelf',
        'shelf': '1号货架',
        'apple': '红富士'
    })
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_robot_controller()
