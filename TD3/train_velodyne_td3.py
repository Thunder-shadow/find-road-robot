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
from squaternion import Quaternion

from velodyne_env import GazeboEnv, check_pos


class Config:
    """配置参数类"""
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    
    # 训练参数
    max_timesteps = 5_000_000
    max_episode_steps = 500
    batch_size = 128
    
    # 文件设置
    save_model = True
    load_model = False
    
    # 混合训练设置
    heuristic_weight = 1.5
    
    # ROS话题设置（根据实际机器人配置调整）
    odom_topic = "/r1/odom"  # 根据日志修改为实际的话题
    cmd_vel_topic = "/p3dx/cmd_vel"  # 根据日志修改为实际的话题


def cleanup_ros():
    """彻底清理ROS进程"""
    try:
        # 杀死所有相关的ROS进程
        subprocess.run(['pkill', '-9', '-f', 'roslaunch'], timeout=5)
        subprocess.run(['pkill', '-9', '-f', 'roscore'], timeout=5)
        subprocess.run(['pkill', '-9', '-f', 'gazebo'], timeout=5)
        subprocess.run(['pkill', '-9', '-f', 'rviz'], timeout=5)
        subprocess.run(['pkill', '-9', '-f', 'robot_state_publisher'], timeout=5)
        subprocess.run(['pkill', '-9', '-f', 'joint_state_publisher'], timeout=5)
        time.sleep(3)
        print("ROS processes cleaned up")
    except Exception as e:
        print(f"Error during ROS cleanup: {e}")


def set_seed(seed: int):
    """设置随机种子"""
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
        """开始监视里程计话题"""
        try:
            # 初始化ROS节点（如果尚未初始化）
            try:
                rospy.init_node('odometry_monitor', anonymous=True)
            except:
                pass  # 如果已经初始化，忽略错误
            
            print(f"Starting to monitor {self.odom_topic} topic")
            self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
            print(f"Started monitoring {self.odom_topic} topic")
            return True
        except Exception as e:
            print(f"Failed to start odometry monitoring: {e}")
            return False
    
    def odom_callback(self, msg):
        """里程计数据回调"""
        self.odom_data = msg
        self.odom_received = True
    
    def wait_for_odometry(self, timeout=30.0):
        """等待接收到里程计数据"""
        print(f"Waiting for odometry data from {self.odom_topic}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.odom_received and self.odom_data is not None:
                print("Odometry data received!")
                return True
            time.sleep(0.1)
            
            # 定期打印状态
            if int(time.time() - start_time) % 5 == 0:
                elapsed = int(time.time() - start_time)
                print(f"Still waiting for odometry... {elapsed}s elapsed")
        
        print(f"Timeout: No odometry data received after {timeout} seconds")
        return False
    
    def get_odom_data(self):
        """获取最新的里程计数据"""
        return self.odom_data
    
    def stop_monitoring(self):
        """停止监视"""
        if self.odom_sub is not None:
            self.odom_sub.unregister()
            self.odom_sub = None


class AStarPlanner:
    """A*路径规划器"""
    def __init__(self, world_range: Tuple[float, float] = (-4.5, 4.5), grid_size: int = 50, 
                 cache_file: str = "gazebo_path_cache.pkl"):
        self.world_min, self.world_max = world_range
        self.grid_size = grid_size
        self.cache_file = cache_file
        self.cell_size = (self.world_max - self.world_min) / grid_size
        
        self.occupancy_grid = self._build_occupancy_grid()
        self.cache = self._load_cache()
    
    def _build_occupancy_grid(self) -> np.ndarray:
        """构建占据栅格地图"""
        grid = np.ones((self.grid_size, self.grid_size), dtype=bool)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                world_x = self.world_min + (i + 0.5) * self.cell_size
                world_y = self.world_min + (j + 0.5) * self.cell_size
                grid[i, j] = check_pos(world_x, world_y)
        
        return grid
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        i = int((x - self.world_min) / self.cell_size)
        j = int((y - self.world_min) / self.cell_size)
        return np.clip(i, 0, self.grid_size - 1), np.clip(j, 0, self.grid_size - 1)
    
    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        x = self.world_min + (i + 0.5) * self.cell_size
        y = self.world_min + (j + 0.5) * self.cell_size
        return x, y
    
    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取相邻栅格"""
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
        """曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _load_cache(self) -> Dict:
        """加载路径缓存"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load path cache: {e}")
        
        return {}
    
    def save_cache(self):
        """保存路径缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Path cache saved to {self.cache_file}")
        except Exception as e:
            print(f"Failed to save path cache: {e}")
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """A*路径规划"""
        start_cell = self.world_to_grid(*start)
        goal_cell = self.world_to_grid(*goal)
        
        # 检查起点和终点是否可达
        if not self.occupancy_grid[start_cell[0], start_cell[1]]:
            print(f"Warning: Start position {start} is in obstacle")
            return []
        if not self.occupancy_grid[goal_cell[0], goal_cell[1]]:
            print(f"Warning: Goal position {goal} is in obstacle")
            return []
        
        # 检查缓存
        cache_key = (*start_cell, *goal_cell)
        if cache_key in self.cache:
            return [self.grid_to_world(i, j) for i, j in self.cache[cache_key]]
        
        # A*算法
        open_set = []
        heapq.heappush(open_set, (self.manhattan_distance(start_cell, goal_cell), 0, start_cell))
        
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: self.manhattan_distance(start_cell, goal_cell)}
        closed_set = set()
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == goal_cell:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_cell)
                path.reverse()
                
                # 缓存路径
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
        return []  # 未找到路径


def compute_heading_error(robot_x: float, robot_y: float, robot_heading: float, 
                         waypoint_x: float, waypoint_y: float) -> float:
    """计算机器人到航向点的角度误差"""
    dx = waypoint_x - robot_x
    dy = waypoint_y - robot_y
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    
    # 计算目标方向角
    target_heading = math.atan2(dy, dx)
    
    # 计算角度误差（归一化到[-π, π]）
    error = target_heading - robot_heading
    error = (error + math.pi) % (2 * math.pi) - math.pi
    
    return error


class DuelingDQN(nn.Module):
    """Dueling DQN网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 组合价值和优势
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class HybridDQNTrainer:
    """混合DQN训练器"""
    def __init__(self, state_dim: int, action_size: int, device: torch.device, 
                 lr: float = 5e-4, gamma: float = 0.99):
        self.device = device
        self.gamma = gamma
        self.action_size = action_size
        
        # 网络
        self.model = DuelingDQN(state_dim, action_size).to(device)
        self.target_model = DuelingDQN(state_dim, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 经验回放
        self.memory = deque(maxlen=100000)
        self.batch_size = Config.batch_size
        
        # 探索参数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        
        # 动作映射
        self.action_map = {
            0: (0.6, 0.0),   # 前进
            1: (0.0, 0.8),   # 左转
            2: (0.0, -0.8),  # 右转
            3: (0.5, 0.5),   # 前进+左转
            4: (0.5, -0.5),  # 前进+右转
        }
        
        self.learn_step = 0
        self.update_frequency = 4
        
        # 训练统计
        self.training_losses = []
        self.episode_rewards = []
    
    def act(self, state: np.ndarray, heading_error: float = 0.0) -> Tuple[List[float], int]:
        """选择动作"""
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            # 基于航向误差的启发式偏差
            heuristic = self._compute_heuristic(heading_error)
            
            # 结合Q值和启发式
            combined_scores = q_values + Config.heuristic_weight * heuristic
            action_idx = np.argmax(combined_scores)
        
        linear, angular = self.action_map[action_idx]
        action = [max(0.0, min(1.0, linear)), max(-1.0, min(1.0, angular))]
        
        return action, action_idx
    
    def _compute_heuristic(self, heading_error: float) -> np.ndarray:
        """计算基于航向误差的启发式函数"""
        heuristic = np.zeros(self.action_size)
        
        # 基于航向误差的启发式规则
        if abs(heading_error) < 0.3:  # 对准方向
            heuristic[0] = 1.0  # 前进
        if heading_error > 0.2:  # 需要左转
            heuristic[1] = 1.0  # 左转
            heuristic[3] = 0.8  # 前进+左转
        if heading_error < -0.2:  # 需要右转
            heuristic[2] = 1.0  # 右转
            heuristic[4] = 0.8  # 前进+右转
        
        return heuristic
    
    def remember(self, state: np.ndarray, action_idx: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self):
        """经验回放学习"""
        if len(self.memory) < self.batch_size:
            return
        
        self.learn_step += 1
        if self.learn_step % self.update_frequency != 0:
            return
        
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # 计算当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        # 记录损失
        self.training_losses.append(loss.item())
        
        # 更新目标网络
        self.soft_update()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def soft_update(self, tau: float = 0.005):
        """软更新目标网络"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, filename: str):
        """保存模型"""
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
        """加载模型"""
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
        """记录回合奖励"""
        self.episode_rewards.append(reward)


class AStarDQNTrainer:
    """基于A*的DQN训练管理器"""
    def __init__(self):
        print("Initializing A-Star DQN training environment...")
        
        # 初始化环境
        self.env = None
        self.max_retries = 2
        self.retry_count = 0
        
        self.state_dim = 20 + 4  # environment_dim + robot_dim
        self.action_size = 5  # 5个动作
        
        # 初始化里程计监视器（使用正确的话题）
        self.odom_monitor = OdometryMonitor(Config.odom_topic)
        
        # 初始化A*路径规划器
        self.planner = AStarPlanner()
        
        # 初始化DQN训练器
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
            # 获取所有活跃的话题
            topics = rospy.get_published_topics()
            topic_names = [topic[0] for topic in topics]
            
            print("Available ROS topics:")
            for topic in topic_names:
                print(f"  - {topic}")
            
            # 检查关键话题（使用实际的话题名称）
            required_topics = [Config.odom_topic, Config.cmd_vel_topic]
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
        """初始化Gazebo环境"""
        try:
            print("Starting Gazebo environment...")
            # 传递正确的话题名称给GazeboEnv
            self.env = GazeboEnv("multi_robot_scenario.launch", 20)
            
            # 等待环境稳定
            print("Waiting for environment to stabilize...")
            time.sleep(15)
            
            # 启动里程计监视
            if not self.odom_monitor.start_monitoring():
                print("Failed to start odometry monitoring")
                return False
            
            # 检查话题可用性
            print("Checking ROS topic availability...")
            if not self.check_topic_availability():
                print("Required ROS topics are not available")
                return False
            
            # 等待里程计数据
            if not self.odom_monitor.wait_for_odometry(30.0):
                print("Failed to receive odometry data within timeout")
                return False
            
            # 检查环境是否正常
            print("Resetting environment...")
            state = self.env.reset()
            if state is not None:
                print("Environment initialized successfully!")
                
                # 将里程计数据传递给环境
                odom_data = self.odom_monitor.get_odom_data()
                if odom_data is not None:
                    self.env.last_odom = odom_data
                    print("Odometry data linked to environment")
                
                self.retry_count = 0
                return True
            else:
                print("Environment reset failed - state is None")
                return False
                
        except Exception as e:
            print(f"Environment initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.retry_count += 1
            return False
    
    def get_robot_pose(self) -> Tuple[float, float, float]:
        """获取机器人位姿"""
        # 首先尝试从里程计监视器获取数据
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
        
        # 回退到环境中的方法
        if hasattr(self.env, 'last_odom') and self.env.last_odom is not None:
            try:
                x = self.env.last_odom.pose.pose.position.x
                y = self.env.last_odom.pose.pose.position.y
                q = self.env.last_odom.pose.pose.orientation
                heading = Quaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)[2]
                return x, y, heading
            except Exception as e:
                print(f"Error getting pose from env odometry: {e}")
        
        return 0.0, 0.0, 0.0
    
    def get_next_waypoint(self) -> Tuple[float, float]:
        """获取下一个航点（基于A*路径规划）"""
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
    
    def safe_step(self, action):
        """安全的环境步骤执行"""
        try:
            # 确保里程计数据可用
            if not self.odom_monitor.odom_received:
                print("Odometry not available before step, waiting...")
                if not self.odom_monitor.wait_for_odometry(5.0):
                    print("Odometry still not available, resetting environment")
                    next_state = self.env.reset()
                    return next_state, -100, True, {"error": "odometry_unavailable"}
            
            next_state, reward, done, info = self.env.step(action)
            return next_state, reward, done, info
        except AttributeError as e:
            if "'NoneType' object has no attribute 'pose'" in str(e):
                print("Odometry data lost during step, resetting environment...")
                next_state = self.env.reset()
                return next_state, -100, True, {"error": "odometry_lost"}
            else:
                print(f"Unexpected AttributeError in environment step: {e}")
                next_state = self.env.reset()
                return next_state, -100, True, {"error": str(e)}
        except Exception as e:
            print(f"Error in environment step: {e}")
            if hasattr(self.env, 'reset'):
                next_state = self.env.reset()
            else:
                next_state = np.zeros(self.state_dim)
            return next_state, -100, True, {"error": str(e)}
    
    def run_training_episode(self):
        """运行一个训练回合"""
        if self.env is None:
            if not self.initialize_environment():
                return -100
        
        try:
            print(f"Starting episode {self.episode_count + 1}")
            state = self.env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            # 重置路径规划
            self.path = []
            self.path_index = 0
            
            # 等待初始状态稳定
            time.sleep(2)
            
            while not done and step_count < Config.max_episode_steps:
                # 获取机器人位姿和下一个航点
                robot_x, robot_y, heading = self.get_robot_pose()
                waypoint_x, waypoint_y = self.get_next_waypoint()
                
                # 计算航向误差
                heading_error = compute_heading_error(robot_x, robot_y, heading, waypoint_x, waypoint_y)
                
                # 选择动作（结合A*路径规划的启发式）
                action, action_idx = self.trainer.act(state, heading_error)
                
                # 执行动作
                next_state, reward, done, info = self.safe_step(action)
                
                # 调整奖励（基于航向误差）
                heading_reward = max(0, 1.0 - abs(heading_error)) * 0.5
                total_reward = reward + heading_reward
                
                done_bool = float(done) if step_count < Config.max_episode_steps - 1 else 0.0
                
                # 存储经验
                self.trainer.remember(state, action_idx, total_reward, next_state, done_bool)
                
                # 学习
                self.trainer.replay()
                
                # 更新状态
                state = next_state
                episode_reward += total_reward
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
                
                # 如果连续失败多次，重新初始化环境
                if episode_reward <= -100 and self.retry_count < self.max_retries:
                    print("Environment seems unstable, reinitializing...")
                    self.odom_monitor.stop_monitoring()
                    if not self.initialize_environment():
                        print("Failed to reinitialize environment, stopping training")
                        break
                elif episode_reward > -50:
                    # 如果表现不错，重置重试计数
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
            self.save_models(final=True)
            print("A-Star DQN training completed")
    
    def save_models(self, final: bool = False):
        """保存模型"""
        if final:
            self.trainer.save("pytorch_models/hybrid_dqn_final.pth")
        else:
            self.trainer.save("pytorch_models/hybrid_dqn_latest.pth")
        self.planner.save_cache()


if __name__ == "__main__":
    # 注册信号处理，确保程序退出时清理ROS进程
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, cleaning up...")
        cleanup_ros()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 初始化训练管理器并开始训练
    trainer = AStarDQNTrainer()
    trainer.train()