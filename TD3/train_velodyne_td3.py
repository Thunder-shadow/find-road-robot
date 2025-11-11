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
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion

# 配置参数类（保持不变）
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    max_timesteps = 5_000_000
    max_episode_steps = 500
    batch_size = 128
    save_model = True
    load_model = False
    heuristic_weight = 1.5
    odom_topic = "/p3dx/odom"  # 匹配p3dx机器人的里程计话题
    cmd_vel_topic = "/p3dx/cmd_vel"  # 匹配p3dx机器人的速度控制话题
    laser_topic = "/p3dx/front_laser/scan"  # p3dx激光雷达话题（根据实际配置调整）


def cleanup_ros():
    """仅在主动退出时清理临时进程，不影响手动启动的环境"""
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
    """里程计数据监视器（保持核心逻辑）"""
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
            print(f"Started monitoring {self.odom_topic} topic")
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
    """新增激光雷达监视器，获取环境状态"""
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
            print(f"Started monitoring {self.laser_topic} topic")
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


class AStarPlanner:
    """A*路径规划器（保持不变）"""
    def __init__(self, world_range: Tuple[float, float] = (-4.5, 4.5), grid_size: int = 50, 
                 cache_file: str = "gazebo_path_cache.pkl"):
        self.world_min, self.world_max = world_range
        self.grid_size = grid_size
        self.cache_file = cache_file
        self.cell_size = (self.world_max - self.world_min) / grid_size
        
        self.occupancy_grid = self._build_occupancy_grid()
        self.cache = self._load_cache()
    
    def _build_occupancy_grid(self) -> np.ndarray:
        grid = np.ones((self.grid_size, self.grid_size), dtype=bool)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                world_x = self.world_min + (i + 0.5) * self.cell_size
                world_y = self.world_min + (j + 0.5) * self.cell_size
                grid[i, j] = self._check_pos(world_x, world_y)  # 修改为内部方法
        
        return grid
    
    def _check_pos(self, x, y):
        """简化的碰撞检测（实际应基于激光雷达数据，这里仅作示例）"""
        obstacles = [  # 可根据实际地图添加障碍物坐标
            (0, 0, 0.5),  # (x, y, 半径)
            (2, 2, 0.5),
            (-2, -2, 0.5)
        ]
        for (ox, oy, r) in obstacles:
            if math.hypot(x - ox, y - oy) < r:
                return False
        return True
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        i = int((x - self.world_min) / self.cell_size)
        j = int((y - self.world_min) / self.cell_size)
        return np.clip(i, 0, self.grid_size - 1), np.clip(j, 0, self.grid_size - 1)
    
    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x = self.world_min + (i + 0.5) * self.cell_size
        y = self.world_min + (j + 0.5) * self.cell_size
        return x, y
    
    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        i, j = cell
        neighbors = []
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < self.grid_size and 0 <= nj < self.grid_size and 
                self.occupancy_grid[ni, nj]):
                neighbors.append((ni, nj))
        
        return neighbors
    
    @staticmethod
    def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load path cache: {e}")
        
        return {}
    
    def save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Path cache saved to {self.cache_file}")
        except Exception as e:
            print(f"Failed to save path cache: {e}")
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        start_cell = self.world_to_grid(*start)
        goal_cell = self.world_to_grid(*goal)
        
        if not self.occupancy_grid[start_cell[0], start_cell[1]]:
            print(f"Warning: Start position {start} is in obstacle")
            return []
        if not self.occupancy_grid[goal_cell[0], goal_cell[1]]:
            print(f"Warning: Goal position {goal} is in obstacle")
            return []
        
        cache_key = (*start_cell, *goal_cell)
        if cache_key in self.cache:
            return [self.grid_to_world(i, j) for i, j in self.cache[cache_key]]
        
        open_set = []
        heapq.heappush(open_set, (self.manhattan_distance(start_cell, goal_cell), 0, start_cell))
        
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self.manhattan_distance(start_cell, goal_cell)}
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
                
                self.cache[cache_key] = path
                return [self.grid_to_world(i, j) for i, j in path]
            
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_g + 1
                
                if (tentative_g < g_score.get(neighbor, float('inf'))):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, goal_cell)
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


class DuelingDQN(nn.Module):
    """Dueling DQN网络（保持不变）"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class HybridDQNTrainer:
    """混合DQN训练器（保持核心逻辑）"""
    def __init__(self, state_dim: int, action_size: int, device: torch.device, 
                 lr: float = 5e-4, gamma: float = 0.99):
        self.device = device
        self.gamma = gamma
        self.action_size = action_size
        
        self.model = DuelingDQN(state_dim, action_size).to(device)
        self.target_model = DuelingDQN(state_dim, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.memory = deque(maxlen=100000)
        self.batch_size = Config.batch_size
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
        self.action_map = {
            0: (0.6, 0.0),   # 前进
            1: (0.0, 0.8),   # 左转
            2: (0.0, -0.8),  # 右转
            3: (0.5, 0.5),   # 前进+左转
            4: (0.5, -0.5),  # 前进+右转
        }
        
        self.learn_step = 0
        self.update_frequency = 4
        
        self.training_losses = []
        self.episode_rewards = []
    
    def act(self, state: np.ndarray, heading_error: float = 0.0) -> Tuple[List[float], int]:
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            heuristic = self._compute_heuristic(heading_error)
            combined_scores = q_values + Config.heuristic_weight * heuristic
            action_idx = np.argmax(combined_scores)
        
        linear, angular = self.action_map[action_idx]
        action = [max(0.0, min(1.0, linear)), max(-1.0, min(1.0, angular))]
        
        return action, action_idx
    
    def _compute_heuristic(self, heading_error: float) -> np.ndarray:
        heuristic = np.zeros(self.action_size)
        
        if abs(heading_error) < 0.3:
            heuristic[0] = 1.0
        if heading_error > 0.2:
            heuristic[1] = 1.0
            heuristic[3] = 0.8
        if heading_error < -0.2:
            heuristic[2] = 1.0
            heuristic[4] = 0.8
        
        return heuristic
    
    def remember(self, state: np.ndarray, action_idx: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.learn_step += 1
        if self.learn_step % self.update_frequency != 0:
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
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        self.soft_update()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def soft_update(self, tau: float = 0.005):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, filename: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards
        }, filename)
        print(f"Hybrid DQN saved to {filename}")
    
    def load(self, filename: str):
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.training_losses = checkpoint.get('training_losses', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            print(f"Hybrid DQN loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load Hybrid DQN: {e}")
            return False
    
    def record_episode_reward(self, reward: float):
        self.episode_rewards.append(reward)


class AStarDQNTrainer:
    """基于A*的DQN训练管理器（核心修改）"""
    def __init__(self):
        print("Initializing A-Star DQN training environment...")
        
        # 环境状态
        self.env = None
        self.max_retries = 2
        self.retry_count = 0
        
        # 状态维度：激光雷达数据(20维) + 机器人位姿(4维)
        self.state_dim = 20 + 4
        
        # 动作数量
        self.action_size = 5
        
        # 初始化监视器
        self.odom_monitor = OdometryMonitor(Config.odom_topic)
        self.laser_monitor = LaserMonitor(Config.laser_topic)
        
        # 初始化发布器（用于发送速度指令）
        self.cmd_vel_pub = None
        
        # 路径规划器和训练器
        self.planner = AStarPlanner()
        self.trainer = HybridDQNTrainer(self.state_dim, self.action_size, Config.device)
        
        # 训练状态
        self.timestep = 0
        self.episode_count = 0
        self.path = []
        self.path_index = 0
        
        # 创建输出目录
        os.makedirs("./pytorch_models", exist_ok=True)
        os.makedirs("./results", exist_ok=True)
    
    def check_topic_availability(self):
        """检查必要的ROS话题是否可用"""
        try:
            if not rospy.core.is_initialized():
                rospy.init_node('topic_checker', anonymous=True)
            
            topics = rospy.get_published_topics()
            topic_names = [topic[0] for topic in topics]
            
            print("Available ROS topics:")
            for topic in topic_names:
                print(f"  - {topic}")
            
            required_topics = [Config.odom_topic, Config.cmd_vel_topic, Config.laser_topic]
            missing_topics = []
            
            for topic in required_topics:
                if topic in topic_names:
                    print(f"✓ {topic} is available")
                else:
                    print(f"✗ {topic} is NOT available")
                    missing_topics.append(topic)
            
            return len(missing_topics) == 0
            
        except Exception as e:
            print(f"Error checking topic availability: {e}")
            return False
    
    def initialize_environment(self):
        """复用手动启动的Gazebo环境，不启动新进程"""
        try:
            print("Reusing existing Gazebo environment...")
            
            # 初始化ROS节点（仅一次）
            if not rospy.core.is_initialized():
                rospy.init_node('astar_dqn_trainer', anonymous=True)
            
            # 初始化速度指令发布器
            self.cmd_vel_pub = rospy.Publisher(Config.cmd_vel_topic, Twist, queue_size=10)
            time.sleep(1)  # 等待发布器初始化
            
            # 启动里程计和激光雷达监视
            if not self.odom_monitor.start_monitoring():
                print("Failed to start odometry monitoring")
                return False
            if not self.laser_monitor.start_monitoring():
                print("Failed to start laser monitoring")
                return False
            
            # 检查话题可用性
            print("Checking ROS topic availability...")
            if not self.check_topic_availability():
                print("Required ROS topics are not available")
                return False
            
            # 等待传感器数据
            if not self.odom_monitor.wait_for_odometry(30.0):
                print("Failed to receive odometry data")
                return False
            if not self.laser_monitor.wait_for_laser(30.0):
                print("Failed to receive laser data")
                return False
            
            # 创建虚拟环境（仅存储目标点和状态）
            class DummyEnv:
                def __init__(self):
                    self.goal_x = random.uniform(-3.0, 3.0)  # 目标点范围（根据地图调整）
                    self.goal_y = random.uniform(-3.0, 3.0)
                    self.last_odom = None
                
                def reset(self):
                    """重置目标点，不重启Gazebo"""
                    self.goal_x = random.uniform(-3.0, 3.0)
                    self.goal_y = random.uniform(-3.0, 3.0)
                    return self._get_initial_state()
                
                def _get_initial_state(self):
                    """获取初始状态（激光+位姿）"""
                    laser_data = self.laser_monitor.get_laser_data()
                    if laser_data is None:
                        return np.zeros(20 + 4)
                    
                    # 激光数据降维到20维
                    laser_ranges = np.array(laser_data.ranges)
                    laser_ranges = np.nan_to_num(laser_ranges, nan=laser_data.range_max)
                    laser_sample = np.linspace(0, len(laser_ranges)-1, 20, dtype=int)
                    laser_state = laser_ranges[laser_sample] / laser_data.range_max  # 归一化
                    
                    # 位姿数据（x, y, heading, 0）
                    x, y, heading = self.get_robot_pose()
                    pose_state = np.array([x/5.0, y/5.0, np.sin(heading), np.cos(heading)])  # 归一化
                    
                    return np.concatenate([laser_state, pose_state])
            
            # 绑定方法到虚拟环境（需要访问外部监视器）
            DummyEnv.get_robot_pose = self.get_robot_pose
            DummyEnv.laser_monitor = self.laser_monitor
            self.env = DummyEnv()
            self.env.last_odom = self.odom_monitor.get_odom_data()
            
            print("Environment initialized successfully (reused existing Gazebo)!")
            self.retry_count = 0
            return True
                
        except Exception as e:
            print(f"Environment initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.retry_count += 1
            return False
    
    def get_robot_pose(self) -> Tuple[float, float, float]:
        """获取机器人位姿（x, y, 航向角）"""
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
    
    def get_next_waypoint(self) -> Tuple[float, float]:
        """获取下一个航点（基于A*路径）"""
        robot_x, robot_y, _ = self.get_robot_pose()
        goal_x, goal_y = self.env.goal_x, self.env.goal_y
        
        # 检查是否接近目标
        distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
        if distance_to_goal < 0.5:
            return goal_x, goal_y
        
        # 路径规划
        if not self.path or self.path_index >= len(self.path):
            print(f"Planning path from ({robot_x:.2f}, {robot_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})")
            self.path = self.planner.plan((robot_x, robot_y), (goal_x, goal_y))
            self.path_index = 0
            
            if not self.path:
                print("No path found, using direct goal")
                return goal_x, goal_y
            else:
                print(f"Path found with {len(self.path)} waypoints")
        
        # 获取当前航点
        waypoint_x, waypoint_y = self.path[self.path_index]
        
        # 检查是否到达航点
        distance_to_waypoint = math.hypot(robot_x - waypoint_x, robot_y - waypoint_y)
        if distance_to_waypoint < 0.3 and self.path_index < len(self.path) - 1:
            self.path_index += 1
            waypoint_x, waypoint_y = self.path[self.path_index]
            print(f"Moving to next waypoint {self.path_index}/{len(self.path)}")
        
        return waypoint_x, waypoint_y
    
    def get_state_from_sensors(self) -> np.ndarray:
        """从传感器数据构建状态向量"""
        # 1. 激光雷达数据（20维）
        laser_data = self.laser_monitor.get_laser_data()
        if laser_data is None:
            laser_state = np.zeros(20)
        else:
            laser_ranges = np.array(laser_data.ranges)
            laser_ranges = np.nan_to_num(laser_ranges, nan=laser_data.range_max)
            laser_sample = np.linspace(0, len(laser_ranges)-1, 20, dtype=int)  # 均匀采样20个点
            laser_state = laser_ranges[laser_sample] / laser_data.range_max  # 归一化到[0,1]
        
        # 2. 机器人位姿数据（4维）
        x, y, heading = self.get_robot_pose()
        pose_state = np.array([
            x / 5.0,  # x坐标归一化（假设地图范围±5m）
            y / 5.0,  # y坐标归一化
            np.sin(heading),  # 航向角正弦（处理周期性）
            np.cos(heading)   # 航向角余弦
        ])
        
        return np.concatenate([laser_state, pose_state])
    
    def safe_step(self, action):
        """安全执行一步动作（直接发布速度指令）"""
        try:
            # 检查传感器数据是否可用
            if not (self.odom_monitor.odom_received and self.laser_monitor.laser_received):
                print("Sensor data unavailable, waiting...")
                if not (self.odom_monitor.wait_for_odometry(5.0) and self.laser_monitor.wait_for_laser(5.0)):
                    print("Sensor data still unavailable, resetting goal")
                    self.env.reset()
                    return self.get_state_from_sensors(), -100, True, {"error": "sensor_unavailable"}
            
            # 发布速度指令
            twist = Twist()
            twist.linear.x = action[0]  # 线速度
            twist.angular.z = action[1]  # 角速度
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)  # 等待动作执行
            
            # 获取下一状态
            next_state = self.get_state_from_sensors()
            
            # 计算奖励
            robot_x, robot_y, heading = self.get_robot_pose()
            waypoint_x, waypoint_y = self.get_next_waypoint()
            goal_x, goal_y = self.env.goal_x, self.env.goal_y
            
            # 距离奖励（接近目标/航点）
            distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
            distance_to_waypoint = math.hypot(robot_x - waypoint_x, robot_y - waypoint_y)
            distance_reward = -0.1 * distance_to_waypoint  # 惩罚远离航点
            
            # 航向奖励（对准航点）
            heading_error = compute_heading_error(robot_x, robot_y, heading, waypoint_x, waypoint_y)
            heading_reward = max(0, 1.0 - abs(heading_error)) * 0.5  # 奖励对准方向
            
            # 碰撞惩罚（激光雷达检测到近距离障碍物）
            laser_data = self.laser_monitor.get_laser_data()
            collision_reward = 0.0
            if laser_data is not None:
                min_range = min(laser_data.ranges)
                if min_range < 0.3:  # 距离障碍物小于0.3m
                    collision_reward = -5.0  # 碰撞惩罚
            
            # 到达目标奖励
            done = False
            goal_reward = 0.0
            if distance_to_goal < 0.5:
                goal_reward = 100.0  # 到达目标的大奖励
                done = True
            
            total_reward = distance_reward + heading_reward + collision_reward + goal_reward
            
            return next_state, total_reward, done, {
                "distance_to_goal": distance_to_goal,
                "heading_error": heading_error
            }
            
        except Exception as e:
            print(f"Error in environment step: {e}")
            self.env.reset()
            return self.get_state_from_sensors(), -100, True, {"error": str(e)}
    
    def run_training_episode(self):
        """运行一个训练回合"""
        if self.env is None:
            if not self.initialize_environment():
                return -100
        
        try:
            print(f"Starting episode {self.episode_count + 1}")
            state = self.env.reset()  # 仅重置目标点，不重启环境
            episode_reward = 0
            done = False
            step_count = 0
            
            # 重置路径规划
            self.path = []
            self.path_index = 0
            
            while not done and step_count < Config.max_episode_steps:
                # 获取机器人位姿和下一个航点
                robot_x, robot_y, heading = self.get_robot_pose()
                waypoint_x, waypoint_y = self.get_next_waypoint()
                
                # 计算航向误差
                heading_error = compute_heading_error(robot_x, robot_y, heading, waypoint_x, waypoint_y)
                
                # 选择动作
                action, action_idx = self.trainer.act(state, heading_error)
                
                # 执行动作
                next_state, reward, done, info = self.safe_step(action)
                
                # 存储经验
                done_bool = float(done) if step_count < Config.max_episode_steps - 1 else 0.0
                self.trainer.remember(state, action_idx, reward, next_state, done_bool)
                
                # 学习
                self.trainer.replay()
                
                # 更新状态
                state = next_state
                episode_reward += reward
                step_count += 1
                self.timestep += 1
                
                # 定期打印进度
                if step_count % 20 == 0:
                    print(f"Episode {self.episode_count + 1}, Step {step_count}, "
                          f"Reward: {episode_reward:.2f}, Heading Error: {heading_error:.3f}")
            
            # 记录回合奖励
            self.trainer.record_episode_reward(episode_reward)
            
            print(f"Episode {self.episode_count + 1} completed with reward: {episode_reward:.2f}")
            return episode_reward
            
        except Exception as e:
            print(f"Error in training episode: {e}")
            import traceback
            traceback.print_exc()
            return -100
    
    def train(self):
        """主训练循环"""
        print("Starting A-Star DQN training...")
        set_seed(Config.seed)
        
        # 加载已有模型
        if Config.load_model:
            self.trainer.load("pytorch_models/hybrid_dqn_latest.pth")
        
        best_reward = -float('inf')
        
        try:
            while self.timestep < Config.max_timesteps and self.retry_count < self.max_retries:
                # 运行训练回合
                episode_reward = self.run_training_episode()
                self.episode_count += 1
                
                print(f"Episode {self.episode_count} completed: "
                      f"Total Reward = {episode_reward:.2f}, "
                      f"Total Timesteps = {self.timestep}, "
                      f"Epsilon = {self.trainer.epsilon:.3f}")
                
                # 定期保存模型
                if self.episode_count % 3 == 0:
                    self.save_models()
                    
                    if episode_reward > best_reward and episode_reward > -50:
                        best_reward = episode_reward
                        self.trainer.save("pytorch_models/hybrid_dqn_best.pth")
                        self.planner.save_cache()
                        print(f"New best model saved with reward: {best_reward:.2f}")
                
                # 环境不稳定时重置目标点（不重启Gazebo）
                if episode_reward <= -100 and self.retry_count < self.max_retries:
                    print("Environment seems unstable, resetting goal...")
                    self.env.reset()
                    self.retry_count += 1
                elif episode_reward > -50:
                    self.retry_count = 0
        
        except KeyboardInterrupt:
            print("Training interrupted by user")
        
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 最终保存
            self.odom_monitor.stop_monitoring()
            self.laser_monitor.stop_monitoring()
            self.save_models(final=True)
            print("A-Star DQN training completed")
    
    def save_models(self, final: bool = False):
        if final:
            self.trainer.save("pytorch_models/hybrid_dqn_final.pth")
        else:
            self.trainer.save("pytorch_models/hybrid_dqn_latest.pth")
        self.planner.save_cache()


if __name__ == "__main__":
    # 注册信号处理
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, cleaning up...")
        cleanup_ros()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 初始化训练管理器并开始训练
    trainer = AStarDQNTrainer()
    trainer.train()
