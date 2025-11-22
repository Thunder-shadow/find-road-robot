import os
import time
import math
import random
import pickle
import heapq
import subprocess
import signal
import rospy
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA
from squaternion import Quaternion

# 配置参数类
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    max_timesteps = 5_000_000
    max_episode_steps = 200
    batch_size = 64
    save_model = True
    load_model = False
    heuristic_weight = 3.0
    odom_topic = "/p3dx/odom"
    cmd_vel_topic = "/p3dx/cmd_vel"
    laser_topic = "/p3dx/front_laser/scan"
    
    # 地图参数
    laser_max_range = 8.0
    map_resolution = 0.05
    map_size = 220
    map_origin_x = -5.5
    map_origin_y = -5.5
    
    # 导航点配置 - 根据您的环境调整
    shelf_locations = [
        (2.0, 2.0),      # 货架1
        (0.5, 2.5),      # 货架2
        (-1.5, 2.0),     # 货架3
        (-2.0, -1.0),    # 货架4
        (1.0, -2.0)      # 货架5
    ]
    charging_station = (0.0, 0.0)  # 充电桩位置
    
    # 训练参数
    collision_penalty = -10.0
    success_reward = 50.0
    step_penalty = -0.1
    min_obstacle_distance = 0.5


class RVizVisualizer:
    """RViz可视化类 - 适配multi_robot_scenario环境"""
    def __init__(self):
        # 使用与multi_robot_scenario兼容的话题名称
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10, latch=True)
        self.path_pub = rospy.Publisher('/navigation_path', Marker, queue_size=10, latch=True)
        self.status_pub = rospy.Publisher('/training_status', Marker, queue_size=10, latch=True)
        
        # 等待发布器初始化
        time.sleep(1)
        print("RViz visualizer initialized with multi_robot_scenario topics")
        
    def clear_all_markers(self):
        """清除所有标记"""
        try:
            marker_array = MarkerArray()
            
            # 清除目标标记
            clear_marker = Marker()
            clear_marker.header.frame_id = "odom"
            clear_marker.ns = "navigation_goals"
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            
            # 清除路径标记
            clear_marker2 = Marker()
            clear_marker2.header.frame_id = "odom"
            clear_marker2.ns = "navigation_path"
            clear_marker2.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker2)
            
            # 清除状态标记
            clear_marker3 = Marker()
            clear_marker3.header.frame_id = "odom"
            clear_marker3.ns = "training_status"
            clear_marker3.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker3)
            
            self.marker_pub.publish(marker_array)
            time.sleep(0.2)
            print("Cleared all RViz markers")
        except Exception as e:
            print(f"Error clearing markers: {e}")
    
    def visualize_goals(self, shelf_locations, charging_station, current_goal=None):
        """可视化所有目标点"""
        try:
            marker_array = MarkerArray()
            
            # 可视化货架点（绿色球体）
            for i, (x, y) in enumerate(shelf_locations):
                marker = Marker()
                marker.header.frame_id = "odom"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "navigation_goals"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = 0.2  # 稍微抬高避免与地面重叠
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.4
                marker.scale.y = 0.4
                marker.scale.z = 0.4
                
                # 如果是当前目标，显示为红色，否则为绿色
                if current_goal and abs(x - current_goal[0]) < 0.1 and abs(y - current_goal[1]) < 0.1:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.9
                else:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.7
                
                marker.lifetime = rospy.Duration(0)  # 永久显示
                marker_array.markers.append(marker)
                
                # 添加货架标签
                text_marker = Marker()
                text_marker.header.frame_id = "odom"
                text_marker.header.stamp = rospy.Time.now()
                text_marker.ns = "navigation_goals"
                text_marker.id = i + 100  # 避免ID冲突
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = x
                text_marker.pose.position.y = y
                text_marker.pose.position.z = 0.8
                text_marker.pose.orientation.w = 1.0
                
                text_marker.scale.z = 0.3  # 文字大小
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                text_marker.text = f"Shelf {i+1}"
                text_marker.lifetime = rospy.Duration(0)
                marker_array.markers.append(text_marker)
            
            # 可视化充电桩（蓝色立方体）
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "navigation_goals"
            marker.id = len(shelf_locations)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = charging_station[0]
            marker.pose.position.y = charging_station[1]
            marker.pose.position.z = 0.2
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.3
            
            # 如果是当前目标，显示为红色，否则为蓝色
            if current_goal and abs(charging_station[0] - current_goal[0]) < 0.1 and abs(charging_station[1] - current_goal[1]) < 0.1:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            marker.color.a = 0.8
            
            marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(marker)
            
            # 充电桩标签
            text_marker = Marker()
            text_marker.header.frame_id = "odom"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "navigation_goals"
            text_marker.id = len(shelf_locations) + 100
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = charging_station[0]
            text_marker.pose.position.y = charging_station[1]
            text_marker.pose.position.z = 0.8
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = "Charging Station"
            text_marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(text_marker)
            
            self.marker_pub.publish(marker_array)
            print(f"Visualized {len(shelf_locations)} shelves and charging station")
            
        except Exception as e:
            print(f"Error visualizing goals: {e}")
    
    def visualize_path(self, path_points):
        """可视化路径"""
        if not path_points:
            return
            
        try:
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "navigation_path"
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            marker.scale.x = 0.08  # 线宽
            
            marker.color.r = 1.0
            marker.color.g = 0.6
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            for point in path_points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0.1  # 稍微抬高
                marker.points.append(p)
            
            marker.lifetime = rospy.Duration(0)
            self.path_pub.publish(marker)
            print(f"Visualized path with {len(path_points)} points")
            
        except Exception as e:
            print(f"Error visualizing path: {e}")
    
    def visualize_training_status(self, episode_count, success_count, collision_count, current_goal_type, position=(0, 4, 0)):
        """可视化训练状态"""
        try:
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "training_status"
            marker.id = 0
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            marker.pose.orientation.w = 1.0
            
            marker.scale.z = 0.4  # 文字大小
            
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            success_rate = (success_count / episode_count * 100) if episode_count > 0 else 0
            status_text = (f"Training Status\n"
                          f"Episode: {episode_count}\n"
                          f"Current Goal: {current_goal_type}\n"
                          f"Success: {success_count} ({success_rate:.1f}%)\n"
                          f"Collisions: {collision_count}")
            
            marker.text = status_text
            marker.lifetime = rospy.Duration(0)  # 永久显示
            
            self.status_pub.publish(marker)
            
        except Exception as e:
            print(f"Error visualizing training status: {e}")


class GazeboController:
    """Gazebo环境控制器"""
    def __init__(self):
        try:
            # 等待服务可用
            rospy.wait_for_service('/gazebo/reset_world', timeout=10)
            rospy.wait_for_service('/gazebo/reset_simulation', timeout=10)
            self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            print("Gazebo services connected successfully")
        except rospy.ServiceException as e:
            print(f"Failed to connect to Gazebo services: {e}")
            self.reset_world = None
            self.reset_simulation = None
        
    def reset_environment(self):
        """重置Gazebo环境"""
        try:
            if self.reset_world:
                self.reset_world()
                print("Gazebo world reset successfully")
                time.sleep(2)  # 等待重置完成
                return True
            return False
        except rospy.ServiceException as e:
            print(f"Failed to reset Gazebo: {e}")
            return False
    
    def reset_simulation_completely(self):
        """完全重置仿真（包括模型位置）"""
        try:
            if self.reset_simulation:
                self.reset_simulation()
                print("Gazebo simulation reset completely")
                time.sleep(3)  # 等待重置完成
                return True
            return False
        except rospy.ServiceException as e:
            print(f"Failed to reset simulation: {e}")
            return False


def cleanup_ros():
    """清理ROS进程"""
    try:
        subprocess.run(['pkill', '-9', '-f', 'odometry_monitor'], timeout=5)
        time.sleep(1)
        print("Training-related ROS processes cleaned up")
    except Exception as e:
        print(f"Cleanup warning: {e}")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class OdometryMonitor:
    """里程计数据监视器"""
    def __init__(self, odom_topic: str = Config.odom_topic):
        self.odom_data = None
        self.odom_received = False
        self.odom_sub = None
        self.odom_topic = odom_topic
        
    def start_monitoring(self):
        try:
            if not rospy.core.is_initialized():
                rospy.init_node('odometry_monitor', anonymous=True)
            
            print(f"Starting to monitor {self.odom_topic} topic")
            self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
            return True
        except Exception as e:
            print(f"Failed to start odometry monitoring: {e}")
            return False
    
    def odom_callback(self, msg):
        self.odom_data = msg
        self.odom_received = True
    
    def wait_for_odometry(self, timeout=30.0):
        print(f"Waiting for odometry data from {self.odom_topic}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.odom_received and self.odom_data is not None:
                print("Odometry data received!")
                return True
            time.sleep(0.1)
            if int(time.time() - start_time) % 5 == 0:
                elapsed = int(time.time() - start_time)
                print(f"Still waiting for odometry... {elapsed}s elapsed")
        
        print(f"Timeout: No odometry data received after {timeout} seconds")
        return False
    
    def get_odom_data(self):
        return self.odom_data
    
    def stop_monitoring(self):
        if self.odom_sub is not None:
            self.odom_sub.unregister()
            self.odom_sub = None


class LaserMonitor:
    """激光雷达监视器"""
    def __init__(self, laser_topic: str = Config.laser_topic):
        self.laser_data = None
        self.laser_received = False
        self.laser_sub = None
        self.laser_topic = laser_topic
        
    def start_monitoring(self):
        try:
            if not rospy.core.is_initialized():
                rospy.init_node('laser_monitor', anonymous=True)
            
            print(f"Starting to monitor {self.laser_topic} topic")
            self.laser_sub = rospy.Subscriber(self.laser_topic, LaserScan, self.laser_callback)
            return True
        except Exception as e:
            print(f"Failed to start laser monitoring: {e}")
            return False
    
    def laser_callback(self, msg):
        self.laser_data = msg
        self.laser_received = True
    
    def wait_for_laser(self, timeout=30.0):
        print(f"Waiting for laser data from {self.laser_topic}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.laser_received and self.laser_data is not None:
                print("Laser data received!")
                return True
            time.sleep(0.1)
            if int(time.time() - start_time) % 5 == 0:
                elapsed = int(time.time() - start_time)
                print(f"Still waiting for laser... {elapsed}s elapsed")
        
        print(f"Timeout: No laser data received after {timeout} seconds")
        return False
    
    def get_laser_data(self):
        return self.laser_data
    
    def stop_monitoring(self):
        if self.laser_sub is not None:
            self.laser_sub.unregister()
            self.laser_sub = None


class DynamicOccupancyGrid:
    """占据栅格地图"""
    def __init__(self):
        self.resolution = Config.map_resolution
        self.size = Config.map_size
        self.origin = np.array([Config.map_origin_x, Config.map_origin_y])
        self.grid = np.zeros((self.size, self.size), dtype=bool)
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        grid_x = int((world_x - self.origin[0]) / self.resolution)
        grid_y = int((world_y - self.origin[1]) / self.resolution)
        grid_x = np.clip(grid_x, 0, self.size - 1)
        grid_y = np.clip(grid_y, 0, self.size - 1)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        world_x = self.origin[0] + (grid_x + 0.5) * self.resolution
        world_y = self.origin[1] + (grid_y + 0.5) * self.resolution
        return world_x, world_y
    
    def update_from_laser(self, robot_x: float, robot_y: float, robot_theta: float, 
                         laser_data: LaserScan):
        if laser_data is None:
            return
        
        ranges = np.array(laser_data.ranges)
        angle_min = laser_data.angle_min
        angle_increment = laser_data.angle_increment
        range_max = laser_data.range_max
        
        ranges = np.nan_to_num(ranges, nan=range_max)
        ranges = np.clip(ranges, 0, range_max)
        
        for i, range_val in enumerate(ranges):
            if range_val >= range_max - 0.1:
                continue
                
            laser_angle = angle_min + i * angle_increment + robot_theta
            end_x = robot_x + range_val * math.cos(laser_angle)
            end_y = robot_y + range_val * math.sin(laser_angle)
            
            end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
            self.grid[end_grid_x, end_grid_y] = True
    
    def is_occupied(self, grid_x: int, grid_y: int) -> bool:
        if not (0 <= grid_x < self.size and 0 <= grid_y < self.size):
            return True
        return self.grid[grid_x, grid_y]
    
    def reset(self):
        self.grid.fill(False)


class AStarPlanner:
    """A*路径规划器"""
    def __init__(self, occupancy_grid: DynamicOccupancyGrid):
        self.occupancy_grid = occupancy_grid
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return self.occupancy_grid.world_to_grid(x, y)
    
    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        return self.occupancy_grid.grid_to_world(i, j)
    
    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        i, j = cell
        neighbors = []
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < self.occupancy_grid.size and 
                0 <= nj < self.occupancy_grid.size and 
                not self.occupancy_grid.is_occupied(ni, nj)):
                neighbors.append((ni, nj))
        
        return neighbors
    
    @staticmethod
    def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        start_cell = self.world_to_grid(*start)
        goal_cell = self.world_to_grid(*goal)
        
        if self.occupancy_grid.is_occupied(*start_cell):
            print(f"Warning: Start position {start} is in obstacle")
            return []
        if self.occupancy_grid.is_occupied(*goal_cell):
            print(f"Warning: Goal position {goal} is in obstacle")
            return []
        
        open_set = []
        heapq.heappush(open_set, (self.euclidean_distance(start_cell, goal_cell), 0, start_cell))
        
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self.euclidean_distance(start_cell, goal_cell)}
        closed_set = set()
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == goal_cell:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_cell)
                path.reverse()
                return [self.grid_to_world(i, j) for i, j in path]
            
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_g + 1.0
                
                if (tentative_g < g_score.get(neighbor, float('inf'))):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.euclidean_distance(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        print(f"No path found from {start} to {goal}")
        return []


def compute_heading_error(robot_x: float, robot_y: float, robot_heading: float, 
                         waypoint_x: float, waypoint_y: float) -> float:
    dx = waypoint_x - robot_x
    dy = waypoint_y - robot_y
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    
    target_heading = math.atan2(dy, dx)
    error = target_heading - robot_heading
    error = (error + math.pi) % (2 * math.pi) - math.pi
    
    return error


class SimpleDQN(nn.Module):
    """简化的DQN网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class HybridDQNTrainer:
    """混合DQN训练器"""
    def __init__(self, state_dim: int, action_size: int, device: torch.device, 
                 lr: float = 1e-4, gamma: float = 0.95):
        self.device = device
        self.gamma = gamma
        self.action_size = action_size
        
        self.model = SimpleDQN(state_dim, action_size).to(device)
        self.target_model = SimpleDQN(state_dim, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.memory = deque(maxlen=10000)
        self.batch_size = Config.batch_size
        
        self.epsilon = 1.0
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.995
        
        self.action_map = {
            0: (0.2, 0.0),   # 慢速前进
            1: (0.0, 0.3),   # 慢速左转
            2: (0.0, -0.3),  # 慢速右转
            3: (0.1, 0.2),   # 前进+左转
            4: (0.1, -0.2),  # 前进+右转
        }
        
        self.learn_step = 0
        self.update_frequency = 1
        
        self.training_losses = []
        self.episode_rewards = []
    
    def act(self, state: np.ndarray, heading_error: float = 0.0, min_laser: float = 10.0) -> Tuple[List[float], int]:
        if min_laser < 0.4:
            if random.random() < 0.6:
                return [0.0, 0.5], 1
            else:
                return [0.0, -0.5], 2
        
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            heuristic = self._compute_heuristic(heading_error, min_laser)
            combined_scores = q_values + Config.heuristic_weight * heuristic
            action_idx = np.argmax(combined_scores)
        
        linear, angular = self.action_map[action_idx]
        return [linear, angular], action_idx
    
    def _compute_heuristic(self, heading_error: float, min_laser: float) -> np.ndarray:
        heuristic = np.zeros(self.action_size)
        
        if min_laser < 0.8:
            heuristic[1] = 1.0
            heuristic[2] = 1.0
            heuristic[0] = -1.0
        
        if abs(heading_error) < 0.3:
            heuristic[0] += 1.0
        elif heading_error > 0.2:
            heuristic[1] += 1.0
            heuristic[3] += 0.5
        elif heading_error < -0.2:
            heuristic[2] += 1.0
            heuristic[4] += 0.5
        
        return heuristic
    
    def remember(self, state: np.ndarray, action_idx: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def soft_update(self):
        tau = 0.01
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, filename: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filename)
        print(f"DQN saved to {filename}")
    
    def load(self, filename: str):
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            print(f"DQN loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load DQN: {e}")
            return False
    
    def record_episode_reward(self, reward: float):
        self.episode_rewards.append(reward)


class AStarDQNTrainer:
    """基于A*的DQN训练管理器 - 适配multi_robot_scenario"""
    def __init__(self):
        print("Initializing A-Star DQN training for multi_robot_scenario environment...")
        
        self.env = None
        self.max_retries = 5
        self.retry_count = 0
        
        self.state_dim = 10 + 3 + 2
        self.action_size = 5
        
        # 初始化监视器
        self.odom_monitor = OdometryMonitor(Config.odom_topic)
        self.laser_monitor = LaserMonitor(Config.laser_topic)
        
        # 初始化发布器
        self.cmd_vel_pub = None
        
        # 初始化Gazebo控制器和RViz可视化
        self.gazebo_controller = GazeboController()
        self.rviz_visualizer = RVizVisualizer()
        
        # 初始化占据栅格和路径规划器
        self.occupancy_grid = DynamicOccupancyGrid()
        self.planner = AStarPlanner(self.occupancy_grid)
        self.trainer = HybridDQNTrainer(self.state_dim, self.action_size, Config.device)
        
        # 训练状态
        self.timestep = 0
        self.episode_count = 0
        self.path = []
        self.path_index = 0
        
        # 创建输出目录
        os.makedirs("./pytorch_models", exist_ok=True)
        os.makedirs("./results", exist_ok=True)
        
        # 统计信息
        self.success_count = 0
        self.collision_count = 0
        self.episode_start_time = 0
        
        # 当前目标信息
        self.current_goal = None
        self.current_goal_type = None
    
    def initialize_environment(self):
        """初始化环境"""
        try:
            print("Initializing multi_robot_scenario environment...")
            
            if not rospy.core.is_initialized():
                rospy.init_node('astar_dqn_trainer', anonymous=True)
            
            self.cmd_vel_pub = rospy.Publisher(Config.cmd_vel_topic, Twist, queue_size=10)
            time.sleep(2)
            
            # 重置Gazebo环境
            print("Resetting Gazebo environment...")
            if not self.gazebo_controller.reset_environment():
                print("Warning: Failed to reset Gazebo environment, continuing anyway...")
            
            # 清除RViz标记
            self.rviz_visualizer.clear_all_markers()
            
            if not self.odom_monitor.start_monitoring():
                return False
            if not self.laser_monitor.start_monitoring():
                return False
            
            if not self.odom_monitor.wait_for_odometry(30.0):
                return False
            if not self.laser_monitor.wait_for_laser(30.0):
                return False
            
            # 等待机器人稳定
            time.sleep(2)
            
            class DummyEnv:
                def __init__(self, outer):
                    self.outer = outer
                    self.goal_x = 0.0
                    self.goal_y = 0.0
                    self.goal_type = None
                    self._generate_navigation_goal()
                
                def reset(self):
                    """重置环境并更新可视化"""
                    self._generate_navigation_goal()
                    self.outer.occupancy_grid.reset()
                    
                    # 发布停止指令
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.outer.cmd_vel_pub.publish(twist)
                    time.sleep(0.5)
                    
                    # 更新RViz可视化
                    self.outer.update_rviz_visualization()
                    
                    return self.outer.get_state_from_sensors()
                
                def _generate_navigation_goal(self):
                    """生成导航目标"""
                    if random.random() < 0.8:
                        self.goal_x, self.goal_y = random.choice(Config.shelf_locations)
                        self.goal_type = "shelf"
                    else:
                        self.goal_x, self.goal_y = Config.charging_station
                        self.goal_type = "charging"
                    
                    # 更新当前目标信息
                    self.outer.current_goal = (self.goal_x, self.goal_y)
                    self.outer.current_goal_type = self.goal_type
                    
                    print(f"🎯 Navigating to {self.goal_type} at ({self.goal_x:.2f}, {self.goal_y:.2f})")
            
            self.env = DummyEnv(self)
            print("multi_robot_scenario environment initialized successfully!")
            return True
                
        except Exception as e:
            print(f"Environment initialization failed: {e}")
            return False
    
    def update_rviz_visualization(self):
        """更新RViz可视化"""
        try:
            # 可视化所有目标点，当前目标高亮显示
            self.rviz_visualizer.visualize_goals(
                Config.shelf_locations, 
                Config.charging_station, 
                self.current_goal
            )
            
            # 可视化路径
            if self.path:
                self.rviz_visualizer.visualize_path(self.path)
            
            # 显示训练状态
            self.rviz_visualizer.visualize_training_status(
                self.episode_count,
                self.success_count,
                self.collision_count,
                self.current_goal_type
            )
            
            print("RViz visualization updated")
            
        except Exception as e:
            print(f"Error updating RViz visualization: {e}")
    
    def get_robot_pose(self) -> Tuple[float, float, float]:
        """获取机器人位姿"""
        odom_data = self.odom_monitor.get_odom_data()
        if odom_data is not None:
            try:
                x = odom_data.pose.pose.position.x
                y = odom_data.pose.pose.position.y
                q = odom_data.pose.pose.orientation
                heading = Quaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)[2]
                return x, y, heading
            except Exception as e:
                print(f"Error getting pose from odometry: {e}")
        return 0.0, 0.0, 0.0
    
    def check_collision(self) -> bool:
        """检查碰撞"""
        laser_data = self.laser_monitor.get_laser_data()
        if laser_data is not None:
            min_range = min(laser_data.ranges)
            return min_range < 0.25
        return False
    
    def get_min_laser_distance(self) -> float:
        """获取最小激光距离"""
        laser_data = self.laser_monitor.get_laser_data()
        if laser_data is not None:
            return min(laser_data.ranges)
        return 10.0
    
    def update_occupancy_grid(self):
        """更新占据栅格"""
        robot_x, robot_y, robot_theta = self.get_robot_pose()
        laser_data = self.laser_monitor.get_laser_data()
        if laser_data is not None:
            self.occupancy_grid.update_from_laser(robot_x, robot_y, robot_theta, laser_data)
    
    def get_next_waypoint(self) -> Tuple[float, float]:
        """获取下一个航点"""
        self.update_occupancy_grid()
        
        robot_x, robot_y, _ = self.get_robot_pose()
        goal_x, goal_y = self.env.goal_x, self.env.goal_y
        
        distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
        if distance_to_goal < 0.6:
            return goal_x, goal_y
        
        if not self.path or self.path_index >= len(self.path):
            self.path = self.planner.plan((robot_x, robot_y), (goal_x, goal_y))
            self.path_index = 0
            
            if not self.path:
                print("No path found, using direct goal")
                return goal_x, goal_y
            
            # 更新路径可视化
            self.update_rviz_visualization()
        
        if self.path_index < len(self.path):
            waypoint_x, waypoint_y = self.path[self.path_index]
            
            distance_to_waypoint = math.hypot(robot_x - waypoint_x, robot_y - waypoint_y)
            if distance_to_waypoint < 0.4 and self.path_index < len(self.path) - 1:
                self.path_index += 1
                waypoint_x, waypoint_y = self.path[self.path_index]
            
            return waypoint_x, waypoint_y
        
        return goal_x, goal_y
    
    def get_state_from_sensors(self) -> np.ndarray:
        """从传感器数据构建状态向量"""
        laser_data = self.laser_monitor.get_laser_data()
        if laser_data is None:
            laser_state = np.zeros(10)
        else:
            laser_ranges = np.array(laser_data.ranges)
            laser_ranges = np.nan_to_num(laser_ranges, nan=laser_data.range_max)
            indices = np.linspace(0, len(laser_ranges)-1, 10, dtype=int)
            laser_state = laser_ranges[indices] / laser_data.range_max
        
        x, y, heading = self.get_robot_pose()
        pose_state = np.array([
            x / 5.0,
            y / 5.0,
            math.sin(heading)
        ])
        
        goal_x, goal_y = self.env.goal_x, self.env.goal_y
        goal_state = np.array([
            (goal_x - x) / 10.0,
            (goal_y - y) / 10.0
        ])
        
        return np.concatenate([laser_state, pose_state, goal_state])
    
    def safe_step(self, action):
        """安全执行一步动作"""
        try:
            if not (self.odom_monitor.odom_received and self.laser_monitor.laser_received):
                return self.get_state_from_sensors(), -10, True, {"error": "sensor_unavailable"}
            
            if self.check_collision():
                self.collision_count += 1
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                return self.get_state_from_sensors(), Config.collision_penalty, True, {"collision": True}
            
            twist = Twist()
            twist.linear.x = action[0]
            twist.angular.z = action[1]
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.2)
            
            next_state = self.get_state_from_sensors()
            
            robot_x, robot_y, heading = self.get_robot_pose()
            waypoint_x, waypoint_y = self.get_next_waypoint()
            goal_x, goal_y = self.env.goal_x, self.env.goal_y
            
            distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
            distance_reward = -0.05 * distance_to_goal
            
            heading_error = compute_heading_error(robot_x, robot_y, heading, waypoint_x, waypoint_y)
            heading_reward = max(0, 1.0 - abs(heading_error)) * 0.2
            
            min_laser = self.get_min_laser_distance()
            obstacle_reward = 0.0
            if min_laser < 0.6:
                obstacle_reward = -1.0
            elif min_laser > 1.0:
                obstacle_reward = 0.05
            
            step_penalty = Config.step_penalty
            
            done = False
            goal_reward = 0.0
            if distance_to_goal < 0.6:
                goal_reward = Config.success_reward
                done = True
                self.success_count += 1
            
            total_reward = distance_reward + heading_reward + obstacle_reward + step_penalty + goal_reward
            
            return next_state, total_reward, done, {
                "distance_to_goal": distance_to_goal,
                "heading_error": heading_error,
                "min_laser": min_laser
            }
            
        except Exception as e:
            print(f"Error in environment step: {e}")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return self.get_state_from_sensors(), -10, True, {"error": str(e)}
    
    def run_training_episode(self):
        """运行训练回合"""
        if self.env is None and not self.initialize_environment():
            return -50
        
        try:
            print(f"\n=== Starting Episode {self.episode_count + 1} ===")
            state = self.env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            self.path = []
            self.path_index = 0
            self.episode_start_time = time.time()
            
            initial_min_laser = self.get_min_laser_distance()
            if initial_min_laser < 0.4:
                print(f"Warning: Starting in dangerous position (min_laser: {initial_min_laser:.3f})")
            
            while not done and step_count < Config.max_episode_steps:
                robot_x, robot_y, heading = self.get_robot_pose()
                waypoint_x, waypoint_y = self.get_next_waypoint()
                min_laser = self.get_min_laser_distance()
                
                heading_error = compute_heading_error(robot_x, robot_y, heading, waypoint_x, waypoint_y)
                
                action, action_idx = self.trainer.act(state, heading_error, min_laser)
                
                next_state, reward, done, info = self.safe_step(action)
                
                self.trainer.remember(state, action_idx, reward, next_state, done)
                
                self.trainer.replay()
                self.trainer.soft_update()
                
                state = next_state
                episode_reward += reward
                step_count += 1
                self.timestep += 1
                
                if step_count % 10 == 0:
                    print(f"Episode {self.episode_count + 1}, Step {step_count}, "
                          f"Reward: {episode_reward:.2f}, Distance: {info['distance_to_goal']:.2f}")
            
            self.trainer.record_episode_reward(episode_reward)
            
            episode_time = time.time() - self.episode_start_time
            print(f"=== Episode {self.episode_count + 1} Completed ===")
            print(f"Reward: {episode_reward:.2f}, Steps: {step_count}, Time: {episode_time:.1f}s")
            print(f"Successes: {self.success_count}, Collisions: {self.collision_count}")
            print(f"Current Goal: {self.current_goal_type} at {self.current_goal}")
            
            return episode_reward
            
        except Exception as e:
            print(f"Error in training episode: {e}")
            return -50
    
    def train(self):
        """主训练循环"""
        print("Starting DQN training with multi_robot_scenario integration...")
        set_seed(Config.seed)
        
        if Config.load_model:
            self.trainer.load("pytorch_models/hybrid_dqn_latest.pth")
        
        best_reward = -float('inf')
        
        try:
            while self.timestep < Config.max_timesteps and self.retry_count < self.max_retries:
                episode_reward = self.run_training_episode()
                self.episode_count += 1
                
                # 每5个回合保存一次模型
                if self.episode_count % 5 == 0:
                    self.trainer.save("pytorch_models/hybrid_dqn_latest.pth")
                    print(f"💾 Model saved at episode {self.episode_count}")
                
                if episode_reward > best_reward and episode_reward > 0:
                    best_reward = episode_reward
                    self.trainer.save("pytorch_models/hybrid_dqn_best.pth")
                    print(f"🎉 New best model saved with reward: {best_reward:.2f}")
                
                if episode_reward <= -40:
                    self.retry_count += 1
                    print(f"⚠️ Environment unstable, retry {self.retry_count}/{self.max_retries}")
                else:
                    self.retry_count = 0
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            self.odom_monitor.stop_monitoring()
            self.laser_monitor.stop_monitoring()
            self.trainer.save("pytorch_models/hybrid_dqn_final.pth")
            success_rate = (self.success_count / self.episode_count * 100) if self.episode_count > 0 else 0
            print(f"\n=== Training Summary ===")
            print(f"Total Episodes: {self.episode_count}")
            print(f"Success Rate: {success_rate:.1f}% ({self.success_count}/{self.episode_count})")
            print(f"Collisions: {self.collision_count}")
            print(f"Best Reward: {best_reward:.2f}")
            # 停止Gazebo和RViz
	    os.system("pkill -f 'gzserver'")  # 停止Gazebo
	    os.system("pkill -f 'rviz'")      # 停止RViz


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nInterrupt received, cleaning up...")
        cleanup_ros()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 等待ROS核心启动
    print("Waiting for ROS core...")
    time.sleep(3)
    
    trainer = AStarDQNTrainer()
    trainer.train()
