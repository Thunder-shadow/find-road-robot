#!/usr/bin/env python3
"""
苹果识别与语音控制配送机器人系统
简化版 - 仅保留语音识别和地点导航功能
修改版：模仿训练代码，不依赖完整TF树
"""

import os
import sys
import time
import math
import json
import threading
import queue
import subprocess
import signal
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import torch

import rospy
import numpy as np
import speech_recognition as sr
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, Quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from squaternion import Quaternion as Squaternion

# 配置参数
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ROS topics
    odom_topic = "/p3dx/odom"
    cmd_vel_topic = "/p3dx/cmd_vel"
    laser_topic = "/p3dx/front_laser/scan"
    speech_topic = "/speech_commands"
    
    # 速度参数
    max_linear_speed = 1.2
    max_angular_speed = 3.0
    
    # 苹果种类到货架的映射
    apple_to_shelf = {
        "红富士": "1号货架",
        "黄元帅": "1号货架", 
        "蛇果": "2号货架",
        "国光": "3号货架",
        "青苹果": "4号货架",
        "嘎啦": "5号货架"
    }
    
    # 货架位置配置
    shelf_locations = {
        "1号货架": (1.0, 3.0),
        "2号货架": (1.0, 1.0),
        "3号货架": (1.0, -1.0),
        "4号货架": (5.0, 4.0),
        "5号货架": (5.0, 1.0),
        "充电站": (-9.0, 7.0),
        "起点": (-5.0, -5.0)
    }
    
    # 语音识别参数
    speech_timeout = 3
    energy_threshold = 1000
    
    # 导航参数
    goal_reached_threshold = 0.5
    obstacle_safety_distance = 0.3
    navigation_timeout = 300
    
    # 初始位置（与launch文件一致）
    initial_position = (-8.0, 8.0, 0.0)  # x, y, theta


class OperatingMode(Enum):
    """机器人操作模式"""
    VOICE_CONTROL = "voice_control"
    IDLE = "idle"


class NavigationController:
    """导航控制器 - 模仿训练代码的导航逻辑"""
    
    def __init__(self, cmd_vel_pub, get_pose_callback, get_laser_callback):
        self.cmd_vel_pub = cmd_vel_pub
        self.get_robot_pose = get_pose_callback
        self.get_laser_data = get_laser_callback
        
        # 导航状态
        self.navigating = False
        self.current_target = None
        self.current_target_name = None
        
        # 控制参数
        self.kp_linear = 1.5  # 线性速度比例
        self.kp_angular = 2.5  # 角速度比例
        self.safe_distance = 0.5  # 安全距离
        
        # 路径跟踪
        self.path_index = 0
        self.waypoints = []
        
        print("✅ 导航控制器初始化完成")
    
    def start_navigation(self, target_position: Tuple[float, float], target_name: str):
        """开始导航到目标点"""
        if self.navigating:
            print("⚠️ 当前正在导航中，先停止当前导航")
            self.stop_navigation()
        
        self.current_target = target_position
        self.current_target_name = target_name
        self.navigating = True
        
        # 生成简单路径（直线路径）
        start_x, start_y, _ = self.get_robot_pose()
        target_x, target_y = target_position
        
        # 生成3个航点
        self.waypoints = []
        for i in range(1, 4):
            ratio = i / 4.0
            wx = start_x + (target_x - start_x) * ratio
            wy = start_y + (target_y - start_y) * ratio
            self.waypoints.append((wx, wy))
        
        self.waypoints.append((target_x, target_y))
        self.path_index = 0
        
        print(f"🗺️ 开始导航到 {target_name}")
        print(f"起点: ({start_x:.2f}, {start_y:.2f})")
        print(f"终点: ({target_x:.2f}, {target_y:.2f})")
        print(f"路径点: {len(self.waypoints)} 个")
        
        return True
    
    def stop_navigation(self):
        """停止导航"""
        self.navigating = False
        self.current_target = None
        self.current_target_name = None
        self.waypoints = []
        self.path_index = 0
        
        # 停止机器人
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        
        print("🛑 导航已停止")
    
    def compute_heading_error(self, robot_x: float, robot_y: float, robot_heading: float, 
                            waypoint_x: float, waypoint_y: float) -> float:
        """计算机器人航向与目标航向之间的误差"""
        dx = waypoint_x - robot_x
        dy = waypoint_y - robot_y
        
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        
        target_heading = math.atan2(dy, dx)
        error = robot_heading - target_heading
        error = (error + math.pi) % (2 * math.pi) - math.pi
        
        return error
    
    def check_obstacle(self) -> Tuple[bool, float]:
        """检查障碍物"""
        laser_data = self.get_laser_data()
        if laser_data is not None:
            ranges = np.array(laser_data.ranges)
            ranges = np.nan_to_num(ranges, nan=laser_data.range_max)
            min_distance = np.min(ranges) if len(ranges) > 0 else 10.0
            return min_distance < self.safe_distance, min_distance
        return False, 10.0
    
    def navigate_step(self):
        """执行一步导航"""
        if not self.navigating or not self.current_target:
            return False
        
        # 获取当前位置和目标
        robot_x, robot_y, robot_heading = self.get_robot_pose()
        target_x, target_y = self.current_target
        
        # 计算到目标的距离
        distance_to_goal = math.hypot(robot_x - target_x, robot_y - target_y)
        
        # 检查是否到达目标
        if distance_to_goal < Config.goal_reached_threshold:
            print(f"🎉 到达目标 {self.current_target_name}!")
            self.stop_navigation()
            return True
        
        # 获取当前航点
        if self.path_index < len(self.waypoints):
            waypoint_x, waypoint_y = self.waypoints[self.path_index]
            distance_to_waypoint = math.hypot(robot_x - waypoint_x, robot_y - waypoint_y)
            
            # 如果接近当前航点，切换到下一个
            if distance_to_waypoint < 0.5 and self.path_index < len(self.waypoints) - 1:
                self.path_index += 1
                waypoint_x, waypoint_y = self.waypoints[self.path_index]
                print(f"前往下一个航点 {self.path_index}/{len(self.waypoints)}")
        else:
            waypoint_x, waypoint_y = target_x, target_y
        
        # 计算航向误差
        heading_error = self.compute_heading_error(robot_x, robot_y, robot_heading, 
                                                  waypoint_x, waypoint_y)
        
        # 检查障碍物
        has_obstacle, min_distance = self.check_obstacle()
        
        # 根据情况计算控制命令
        if has_obstacle:
            print(f"⚠️ 检测到障碍物，距离: {min_distance:.2f}m")
            # 避障策略
            if heading_error > 0:
                angular = Config.max_angular_speed * 0.8
            else:
                angular = -Config.max_angular_speed * 0.8
            linear = -0.2  # 轻微后退
        else:
            # 正常导航控制
            # 线性速度：基于距离的目标速度
            linear_speed = min(Config.max_linear_speed * 0.8, 
                              distance_to_goal * self.kp_linear * 0.5)
            
            # 角速度：基于航向误差
            angular_speed = -heading_error * self.kp_angular
            angular_speed = max(-Config.max_angular_speed * 0.6,
                               min(Config.max_angular_speed * 0.6, angular_speed))
            
            # 接近目标时减速
            if distance_to_goal < 1.0:
                linear_speed = linear_speed * (distance_to_goal / 1.0)
            
            linear = linear_speed
            angular = angular_speed
        
        # 发布控制命令
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)
        
        # 显示导航状态
        if int(time.time()) % 5 == 0:  # 每5秒打印一次状态
            print(f"📍 导航状态: 距离目标{distance_to_goal:.2f}m, "
                  f"航向误差{heading_error:.3f}rad, "
                  f"速度({linear:.2f}, {angular:.2f})")
        
        return False
    
    def is_navigating(self):
        """是否正在导航"""
        return self.navigating


class LocalMicrophoneVoiceControl:
    """本地麦克风语音控制模块"""
    
    def __init__(self):
        self.command_queue = queue.Queue()
        self.listening = False
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self._initialize_microphone()
        self.command_mapping = {
            "红富士": "red_fuji",
            "黄元帅": "yellow_general",
            "蛇果": "snake_fruit",
            "国光": "national_light",
            "青苹果": "green_apple",
            "嘎啦": "gala",
            "开始导航": "start_navigation",
            "返回起点": "return_home",
            "充电": "charge",
            "停止": "stop"
        }
    
    def _initialize_microphone(self):
        """初始化麦克风"""
        print("初始化麦克风...")
        try:
            # 抑制ALSA警告
            os.environ['PYTHONWARNINGS'] = 'ignore'
            
            # 列出可用麦克风
            print("检测麦克风设备...")
            try:
                mic_list = sr.Microphone.list_microphone_names()
                if mic_list:
                    print(f"找到 {len(mic_list)} 个音频输入设备:")
                    for i, mic_name in enumerate(mic_list):
                        print(f"  [{i}] {mic_name}")
                    
                    # 尝试选择默认麦克风
                    print("\n尝试使用默认麦克风...")
                    self.microphone = sr.Microphone()
                    
                    # 测试麦克风
                    with self.microphone as source:
                        print("正在调整环境噪声...")
                        try:
                            self.recognizer.adjust_for_ambient_noise(source, duration=1)
                            print(f"✅ 麦克风初始化成功")
                            print(f"  环境噪声能量阈值: {self.recognizer.energy_threshold}")
                            return
                        except Exception as e:
                            print(f"⚠️ 默认麦克风测试失败: {e}")
                    
                    # 如果默认麦克风失败，尝试其他设备
                    for device_index in range(len(mic_list)):
                        if device_index != 0:  # 跳过已经尝试过的默认设备
                            print(f"\n尝试设备 [{device_index}]: {mic_list[device_index]}")
                            try:
                                self.microphone = sr.Microphone(device_index=device_index)
                                with self.microphone as source:
                                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                                    print(f"✅ 使用设备 [{device_index}] 成功")
                                    return
                            except Exception as e:
                                print(f"设备 [{device_index}] 失败: {e}")
                else:
                    print("❌ 未找到音频输入设备")
                    
            except Exception as e:
                print(f"❌ 检测麦克风设备失败: {e}")
            
            # 如果没有找到可用麦克风
            self.microphone = None
            print("⚠️ 使用备用音频输入方案")
                
        except Exception as e:
            print(f"❌ 麦克风初始化失败: {e}")
            self.microphone = None
    
    def start_listening(self):
        """开始监听语音命令"""
        if self.listening:
            return
        
        if not self.microphone:
            print("❌ 无法启动语音识别：无可用麦克风设备")
            print("请检查麦克风连接或音频驱动")
            return False
        
        self.listening = True
        self.listener_thread = threading.Thread(target=self._listening_loop, daemon=True)
        self.listener_thread.start()
        print("✅ 开始监听语音命令...")
        print("请说苹果名称如'红富士'或系统命令如'返回起点'")
        return True
    
    def _listening_loop(self):
        """监听循环"""
        print("语音监听线程启动...")
        consecutive_errors = 0
        
        while self.listening and consecutive_errors < 5:
            try:
                with self.microphone as source:
                    print("\n🎤 正在聆听... (说话即可)")
                    audio = self.recognizer.listen(
                        source, 
                        timeout=Config.speech_timeout,
                        phrase_time_limit=5
                    )
                    
                    # 识别语音
                    print("识别中...")
                    try:
                        # 使用Google语音识别API（需要网络连接）
                        text = self.recognizer.recognize_google(audio, language='zh-CN')
                        if text:
                            self._process_speech(text)
                            consecutive_errors = 0  # 重置错误计数
                            
                    except sr.UnknownValueError:
                        print("未能识别语音内容")
                        consecutive_errors += 1
                    except sr.RequestError as e:
                        print(f"语音识别服务错误: {e}")
                        consecutive_errors += 1
                    except Exception as e:
                        print(f"识别错误: {e}")
                        consecutive_errors += 1
                        
            except sr.WaitTimeoutError:
                # 超时正常，继续监听
                consecutive_errors = 0
                continue
            except Exception as e:
                print(f"监听错误: {e}")
                consecutive_errors += 1
                time.sleep(1)
        
        if consecutive_errors >= 5:
            print("⚠️ 连续多次识别失败，语音监听停止")
            self.listening = False
    
    def _process_speech(self, text: str):
        """处理识别到的语音"""
        print(f"🗣️ 识别到: {text}")
        
        # 简单关键字匹配
        for keyword in self.command_mapping.keys():
            if keyword in text:
                print(f"✅ 匹配命令: {keyword}")
                self.command_queue.put(keyword)
                return
        
        # 模糊匹配
        matched = self._fuzzy_match(text)
        if matched:
            print(f"✅ 模糊匹配: {matched}")
            self.command_queue.put(matched)
        else:
            print("⚠️ 未识别命令，请重试")
    
    def _fuzzy_match(self, text: str) -> Optional[str]:
        """模糊匹配命令"""
        import difflib
        
        commands = list(self.command_mapping.keys())
        matches = difflib.get_close_matches(text, commands, n=1, cutoff=0.6)
        return matches[0] if matches else None
    
    def get_command(self, timeout: float = 0.1) -> Optional[str]:
        """获取语音命令"""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def cleanup(self):
        """清理资源"""
        self.listening = False


class AppleDeliveryRobot:
    """苹果配送机器人主控制器 - 语音控制版（模仿训练代码）"""
    
    def __init__(self):
        print("=" * 70)
        print("🍎 苹果配送机器人系统 - 语音控制版 (TF树独立)")
        print("=" * 70)
        
        # 初始化ROS
        try:
            rospy.init_node('apple_delivery_robot_voice_only', anonymous=True, disable_signals=True)
            print("✅ ROS节点初始化成功")
        except:
            print("ROS节点已初始化")
        
        # 传感器数据
        self.odom_data = None
        self.laser_data = None
        self.robot_pose = Config.initial_position  # (x, y, theta)
        
        # ROS发布器/订阅器
        self.cmd_vel_pub = rospy.Publisher(Config.cmd_vel_topic, Twist, queue_size=10)
        self.status_pub = rospy.Publisher("/robot_status", String, queue_size=10)
        
        # 订阅传感器数据
        self.odom_sub = rospy.Subscriber(Config.odom_topic, Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber(Config.laser_topic, LaserScan, self.laser_callback)
        
        # 初始化语音模块
        print("\n初始化语音模块...")
        self.voice = LocalMicrophoneVoiceControl()
        
        # 初始化导航控制器
        print("\n初始化导航控制器...")
        self.navigation = NavigationController(
            self.cmd_vel_pub,
            self.get_robot_pose,
            self.get_laser_data
        )
        
        # 控制状态
        self.running = True
        self.current_mode = OperatingMode.VOICE_CONTROL
        
        # 启动语音监听
        if not self.voice.start_listening():
            print("\n⚠️ 语音监听启动失败，系统继续运行但无法接收语音命令")
        
        # 等待传感器数据
        self._wait_for_sensors()
        
        print("\n✅ 系统初始化完成！")
        self._print_instructions()
    
    def odom_callback(self, msg):
        """里程计回调 - 直接获取位姿，不依赖TF"""
        self.odom_data = msg
        
        # 直接从odometry消息计算位姿（模仿训练代码）
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            
            # 使用squaternion库转换（和训练代码一样）
            heading = Squaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)[2]
            self.robot_pose = (x, y, heading)
            
        except Exception as e:
            print(f"Error parsing odometry: {e}")
            # 使用初始位置作为后备
            self.robot_pose = Config.initial_position
    
    def laser_callback(self, msg):
        """激光雷达回调"""
        self.laser_data = msg
    
    def get_robot_pose(self) -> Tuple[float, float, float]:
        """获取机器人位姿（模仿训练代码）"""
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
        print("语音控制版使用说明:")
        print("=" * 70)
        print("可用语音命令:")
        print("  1. 苹果名称 (将导航到对应货架):")
        for apple in Config.apple_to_shelf.keys():
            shelf = Config.apple_to_shelf[apple]
            print(f"    '{apple}' -> 导航到{shelf}")
        print()
        print("  2. 系统命令:")
        print("    '开始导航' - 开始配送任务")
        print("    '返回起点' - 返回起始点")
        print("    '充电' - 前往充电站")
        print("    '停止' - 停止当前导航")
        print()
        print("当前机器人状态:")
        x, y, theta = self.get_robot_pose()
        print(f"  位置: ({x:.2f}, {y:.2f})")
        print(f"  朝向: {theta:.2f} rad ({theta*180/math.pi:.1f}°)")
        print()
        print("注意:")
        print("  - 语音识别需要网络连接使用Google语音识别API")
        print("  - ALSA警告信息是音频驱动问题，通常不影响使用")
        print("  - 请靠近麦克风清晰发音")
        print("=" * 70)
    
    def _handle_voice_command(self, command: str):
        """处理语音命令"""
        print(f"\n🎤 执行命令: {command}")
        
        # 苹果配送命令
        if command in Config.apple_to_shelf:
            shelf = Config.apple_to_shelf[command]
            if shelf in Config.shelf_locations:
                target = Config.shelf_locations[shelf]
                print(f"🚀 导航到{shelf}: 坐标 {target}")
                self.navigation.start_navigation(target, shelf)
                self._publish_status(f"开始导航到{shelf}")
        
        # 系统命令
        elif command == "开始导航":
            print("📍 开始配送任务")
            self._start_delivery_task()
        
        elif command == "返回起点":
            print("🏠 返回起点")
            target = Config.shelf_locations["起点"]
            self.navigation.start_navigation(target, "起点")
            self._publish_status("返回起点")
        
        elif command == "充电":
            print("🔋 前往充电站")
            target = Config.shelf_locations["充电站"]
            self.navigation.start_navigation(target, "充电站")
            self._publish_status("前往充电站")
        
        elif command == "停止":
            print("🛑 停止导航")
            self.navigation.stop_navigation()
            self._publish_status("停止导航")
        
        else:
            print(f"⚠️ 未知命令: {command}")
    
    def _start_delivery_task(self):
        """开始配送任务"""
        print("📦 开始苹果配送任务")
        self._publish_status("开始苹果配送任务")
        
        # 这里可以添加配送任务的逻辑
        # 例如：遍历所有苹果类型，依次导航到对应货架
    
    def _publish_status(self, status: str):
        """发布状态信息"""
        try:
            msg = String()
            msg.data = status
            self.status_pub.publish(msg)
        except Exception as e:
            print(f"发布状态失败: {e}")
    
    def run(self):
        """运行主循环"""
        print("\n🚀 系统启动！等待语音命令...")
        print("机器人初始位置:", self.get_robot_pose())
        
        # 最后位置更新时间
        last_pose_update = time.time()
        last_status_print = time.time()
        
        try:
            while self.running and not rospy.is_shutdown():
                current_time = time.time()
                
                # 定期更新位置显示
                if current_time - last_pose_update > 2.0:
                    x, y, theta = self.get_robot_pose()
                    if current_time - last_status_print > 10.0:
                        print(f"🤖 机器人位置: ({x:.2f}, {y:.2f}), 朝向: {theta:.2f} rad")
                        last_status_print = current_time
                    last_pose_update = current_time
                
                # 处理语音命令
                voice_cmd = self.voice.get_command(timeout=0.1)
                if voice_cmd:
                    self._handle_voice_command(voice_cmd)
                
                # 执行导航步骤
                if self.navigation.is_navigating():
                    self.navigation.navigate_step()
                
                # 显示状态提示
                if int(current_time) % 15 == 0 and int(current_time) > 0:
                    if not self.navigation.is_navigating():
                        print("\n💡 提示: 请说出苹果名称或系统命令")
                        print("   例如: '红富士', '返回起点', '充电', '停止'")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号")
        except Exception as e:
            print(f"\n❌ 运行错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        self.running = False
        self.voice.cleanup()
        self.navigation.stop_navigation()
        
        # 停止机器人
        try:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
        except:
            pass
        
        print("✅ 清理完成")


def main():
    """主函数"""
    print("🚀 启动苹果配送机器人语音控制系统...")
    print("版本: 训练代码兼容版 (不依赖TF树)")
    print("=" * 70)
    
    # 检查依赖
    try:
        import speech_recognition
        print("✅ speech_recognition 可用")
    except ImportError:
        print("❌ 需要安装: pip install SpeechRecognition")
        return
    
    # 检查squaternion库
    try:
        import squaternion
        print("✅ squaternion 可用")
    except ImportError:
        print("❌ 需要安装: pip install squaternion")
        return
    
    # 检查网络连接
    print("检查网络连接...")
    try:
        import urllib.request
        urllib.request.urlopen('http://google.com', timeout=1)
        print("✅ 网络连接正常")
    except:
        print("⚠️ 网络连接可能有问题，语音识别需要网络连接")
    
    try:
        robot = AppleDeliveryRobot()
        robot.run()
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    main()
