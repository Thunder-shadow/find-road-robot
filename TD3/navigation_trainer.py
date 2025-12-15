"""
寻路模型训练器 - 简化版
基于Dueling DQN的A*启发式学习
"""

import os
import sys
import time
import math
import random
import pickle
import heapq
import threading
import json
from typing import List, Tuple, Optional, Dict, Any
from collections import deque, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion as Squaternion

from config import Config, SimulationConfig


class DuelingDQN(nn.Module):
    """Dueling DQN网络"""
    def __init__(self, state_dim: int, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        # 特征提取层
        self.feature = nn.Sequential(
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
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class HeuristicLearner:
    """启发式学习器"""
    def __init__(self, state_dim: int = 12, device: torch.device = None):
        self.device = device or Config.device
        self.state_dim = state_dim
        
        # 创建网络
        self.model = DuelingDQN(state_dim).to(self.device)
        self.target_model = DuelingDQN(state_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=Config.learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=50000)
        self.batch_size = Config.batch_size
        
        # 训练参数
        self.gamma = Config.gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # 统计
        self.losses = []
        self.update_count = 0
        
    def create_state(self, node: Tuple[int, int], goal: Tuple[int, int], 
                    occupancy_map) -> np.ndarray:
        """创建状态特征"""
        node_x, node_y = node
        goal_x, goal_y = goal
        
        # 基础特征
        dx = goal_x - node_x
        dy = goal_y - node_y
        euclidean = math.sqrt(dx*dx + dy*dy)
        manhattan = abs(dx) + abs(dy)
        
        # 障碍物特征
        obstacle_density = 0.0
        clearance = 0.0
        
        # 3x3邻域
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = node_x + i, node_y + j
                if 0 <= nx < occupancy_map.size and 0 <= ny < occupancy_map.size:
                    if occupancy_map.is_occupied(nx, ny):
                        obstacle_density += 1.0
                    clearance += occupancy_map.get_obstacle_distance(nx, ny, 3)
        
        obstacle_density /= 9.0
        clearance /= 9.0
        
        # 创建特征向量
        features = np.array([
            dx / occupancy_map.size,
            dy / occupancy_map.size,
            euclidean / (occupancy_map.size * math.sqrt(2)),
            manhattan / (occupancy_map.size * 2),
            math.atan2(dy, dx) / math.pi,
            obstacle_density,
            clearance / 10.0,
            occupancy_map.get_obstacle_density(node_x, node_y, 2),
            node_x / occupancy_map.size,
            node_y / occupancy_map.size,
            goal_x / occupancy_map.size,
            goal_y / occupancy_map.size
        ], dtype=np.float32)
        
        return features
    
    def predict_heuristic(self, state: np.ndarray) -> float:
        """预测启发式值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            heuristic = q_values.max().item()
            return max(0.0, heuristic)
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # 当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 记录
        self.losses.append(loss.item())
        self.update_count += 1
        
        # 探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新目标网络
        if self.update_count % 100 == 0:
            self.soft_update()
        
        return loss.item()
    
    def soft_update(self, tau: float = 0.01):
        """软更新目标网络"""
        for target_param, param in zip(self.target_model.parameters(), 
                                       self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, filename: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'update_count': self.update_count
        }
        torch.save(checkpoint, filename)
        print(f"✅ 模型保存到: {filename}")
    
    def load(self, filename: str) -> bool:
        """加载模型"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.losses = checkpoint.get('losses', [])
            self.update_count = checkpoint.get('update_count', 0)
            print(f"✅ 模型加载成功: {filename}")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False


class OccupancyGridMap:
    """占据栅格地图"""
    def __init__(self, resolution=0.2, size=100, origin=(-10.0, -10.0)):
        self.resolution = resolution
        self.size = size
        self.origin = np.array(origin)
        self.grid = np.zeros((size, size), dtype=np.float32)
        
    def world_to_grid(self, world_x, world_y):
        grid_x = int((world_x - self.origin[0]) / self.resolution)
        grid_y = int((world_y - self.origin[1]) / self.resolution)
        grid_x = np.clip(grid_x, 0, self.size - 1)
        grid_y = np.clip(grid_y, 0, self.size - 1)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        world_x = self.origin[0] + (grid_x + 0.5) * self.resolution
        world_y = self.origin[1] + (grid_y + 0.5) * self.resolution
        return world_x, world_y
    
    def update_from_laser(self, robot_x, robot_y, robot_theta, laser_data):
        if laser_data is None:
            return
        
        ranges = np.array(laser_data.ranges)
        angle_min = laser_data.angle_min
        angle_increment = laser_data.angle_increment
        max_range = 8.0
        
        ranges = np.nan_to_num(ranges, nan=max_range)
        ranges = np.clip(ranges, 0, max_range)
        
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        for i, range_val in enumerate(ranges):
            laser_angle = angle_min + i * angle_increment + robot_theta
            
            if range_val < max_range - 0.1:
                end_x = robot_x + range_val * math.cos(laser_angle)
                end_y = robot_y + range_val * math.sin(laser_angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                self.mark_obstacle(end_grid_x, end_grid_y)
    
    def mark_obstacle(self, grid_x, grid_y, probability=0.8):
        if 0 <= grid_x < self.size and 0 <= grid_y < self.size:
            old_value = self.grid[grid_x, grid_y]
            self.grid[grid_x, grid_y] = min(1.0, old_value + probability * (1 - old_value))
    
    def is_occupied(self, grid_x, grid_y, threshold=0.5):
        if not (0 <= grid_x < self.size and 0 <= grid_y < self.size):
            return True
        return self.grid[grid_x, grid_y] > threshold
    
    def get_obstacle_distance(self, grid_x, grid_y, radius=5):
        min_distance = float('inf')
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.is_occupied(nx, ny):
                        distance = math.sqrt(dx*dx + dy*dy)
                        min_distance = min(min_distance, distance)
        return min_distance if min_distance < float('inf') else radius
    
    def get_obstacle_density(self, grid_x, grid_y, radius=2):
        count = 0
        total = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    total += 1
                    if self.is_occupied(nx, ny):
                        count += 1
        return count / total if total > 0 else 0.0


class DuelingDQNAStarTrainer:
    """Dueling DQN A*训练器"""
    
    def __init__(self):
        print("初始化Dueling DQN A*训练器...")
        
        # 初始化ROS
        try:
            rospy.init_node('dueling_dqn_trainer', anonymous=True)
        except:
            pass
        
        # 传感器数据
        self.odom_data = None
        self.laser_data = None
        self.robot_pose = Config.initial_position
        
        # ROS订阅器
        self.odom_sub = rospy.Subscriber(Config.odom_topic, Odometry, self.odom_callback)
        self.laser_sub = rospy.Subscriber(Config.laser_topic, LaserScan, self.laser_callback)
        
        # 初始化组件
        self.occupancy_map = OccupancyGridMap(
            resolution=Config.map_resolution,
            size=Config.map_size,
            origin=Config.map_origin
        )
        
        self.heuristic_learner = HeuristicLearner()
        
        # 训练状态
        self.training = False
        self.current_episode = 0
        self.total_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
        # 创建模型目录
        os.makedirs("models", exist_ok=True)
        os.makedirs("training_logs", exist_ok=True)
        
        print("✅ Dueling DQN A*训练器初始化完成")
    
    def odom_callback(self, msg):
        self.odom_data = msg
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            heading = Squaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)[2]
            self.robot_pose = (x, y, heading)
        except:
            pass
    
    def laser_callback(self, msg):
        self.laser_data = msg
    
    def get_robot_pose(self):
        return self.robot_pose
    
    def wait_for_sensors(self, timeout=30):
        """等待传感器数据"""
        print("等待传感器数据...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.odom_data is not None and self.laser_data is not None:
                print("✅ 传感器数据接收成功")
                return True
            time.sleep(0.5)
        
        print("⚠️ 传感器数据等待超时")
        return False
    
    def run_episode(self):
        """运行一个训练回合"""
        if not self.wait_for_sensors():
            return -10
        
        print(f"\n=== 训练回合 {self.current_episode + 1} ===")
        
        # 重置环境
        robot_x, robot_y, robot_theta = self.get_robot_pose()
        start_pos = (robot_x, robot_y)
        
        # 随机选择目标
        shelf_names = list(Config.shelf_locations.keys())
        goal_name = random.choice(shelf_names)
        goal_pos = Config.shelf_locations[goal_name]
        
        print(f"起始位置: ({robot_x:.2f}, {robot_y:.2f})")
        print(f"目标位置: {goal_name} ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
        
        # 更新地图
        self.occupancy_map.update_from_laser(robot_x, robot_y, robot_theta, self.laser_data)
        
        # A*规划（使用学习的启发式）
        path = self.a_star_with_learned_heuristic(start_pos, goal_pos)
        
        # 跟随路径并收集经验
        episode_reward = 0
        steps = 0
        success = False
        
        if path:
            print(f"找到路径，长度: {len(path)}")
            # 这里可以添加路径跟随和奖励计算逻辑
            # 简化版：假设成功到达
            success = True
            episode_reward = 50
        
        # 记录结果
        self.total_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        self.success_rates.append(1 if success else 0)
        
        # 经验回放
        for _ in range(3):
            loss = self.heuristic_learner.replay()
            if loss > 0:
                print(f"训练损失: {loss:.4f}")
        
        self.current_episode += 1
        
        # 定期保存模型
        if self.current_episode % 10 == 0:
            self.save_training_progress()
        
        return episode_reward
    
    def a_star_with_learned_heuristic(self, start_world, goal_world):
        """使用学习启发式的A*算法"""
        start_cell = self.occupancy_map.world_to_grid(*start_world)
        goal_cell = self.occupancy_map.world_to_grid(*goal_world)
        
        # A*算法
        open_set = []
        start_state = self.heuristic_learner.create_state(start_cell, goal_cell, self.occupancy_map)
        start_h = self.heuristic_learner.predict_heuristic(start_state)
        heapq.heappush(open_set, (start_h, 0, start_cell))
        
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: start_h}
        
        closed_set = set()
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == goal_cell:
                # 重建路径
                path = self.reconstruct_path(came_from, current, start_cell)
                world_path = [self.occupancy_map.grid_to_world(i, j) for i, j in path]
                return world_path
            
            closed_set.add(current)
            
            # 探索邻居（4方向）
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_cell(*neighbor):
                    continue
                
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_g + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # 使用学习启发式
                    neighbor_state = self.heuristic_learner.create_state(neighbor, goal_cell, self.occupancy_map)
                    h_score = self.heuristic_learner.predict_heuristic(neighbor_state)
                    f_score[neighbor] = tentative_g + h_score
                    
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        return []
    
    def is_valid_cell(self, grid_x, grid_y):
        if not (0 <= grid_x < self.occupancy_map.size and 
                0 <= grid_y < self.occupancy_map.size):
            return False
        return not self.occupancy_map.is_occupied(grid_x, grid_y)
    
    def reconstruct_path(self, came_from, current, start):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path
    
    def save_training_progress(self):
        """保存训练进度"""
        # 保存模型
        model_path = f"models/dueling_dqn_heuristic_ep{self.current_episode}.pth"
        self.heuristic_learner.save(model_path)
        
        # 保存训练日志
        log_data = {
            'episodes': self.current_episode,
            'rewards': self.total_rewards,
            'lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'timestamp': time.time()
        }
        
        log_path = f"training_logs/training_ep{self.current_episode}.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✅ 训练进度保存到: {model_path}, {log_path}")
    
    def train(self, episodes=100):
        """主训练循环"""
        print(f"开始训练 {episodes} 回合...")
        
        self.training = True
        
        # 加载现有模型
        if os.path.exists(Config.dueling_dqn_model_path):
            self.heuristic_learner.load(Config.dueling_dqn_model_path)
        
        try:
            for i in range(episodes):
                if not self.training:
                    break
                
                reward = self.run_episode()
                print(f"回合 {i+1}: 奖励 = {reward}")
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n训练被中断")
        finally:
            self.training = False
            self.save_training_progress()
            
            # 生成训练报告
            self.generate_training_report()
    
    def generate_training_report(self):
        """生成训练报告"""
        if not self.total_rewards:
            return
        
        print("\n=== 训练报告 ===")
        print(f"总回合数: {self.current_episode}")
        print(f"平均奖励: {np.mean(self.total_rewards):.2f}")
        print(f"成功率: {np.mean(self.success_rates)*100:.1f}%")
        print(f"平均回合长度: {np.mean(self.episode_lengths):.2f}")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 奖励曲线
        axes[0, 0].plot(self.total_rewards)
        axes[0, 0].set_title('回合奖励')
        axes[0, 0].set_xlabel('回合')
        axes[0, 0].set_ylabel('奖励')
        
        # 成功率
        window = 10
        success_moving_avg = []
        for i in range(window, len(self.success_rates)):
            avg = np.mean(self.success_rates[i-window:i])
            success_moving_avg.append(avg)
        
        axes[0, 1].plot(range(window, len(self.success_rates)), success_moving_avg)
        axes[0, 1].set_title(f'成功率 ({window}回合移动平均)')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('成功率')
        
        # 回合长度
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('回合长度')
        axes[1, 0].set_xlabel('回合')
        axes[1, 0].set_ylabel('步数')
        
        # 损失曲线
        if self.heuristic_learner.losses:
            axes[1, 1].plot(self.heuristic_learner.losses[-100:])
            axes[1, 1].set_title('训练损失 (最近100次)')
            axes[1, 1].set_xlabel('更新次数')
            axes[1, 1].set_ylabel('损失')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"training_logs/training_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"训练图表保存到: {plot_path}")


class TrainingPlotCanvas(FigureCanvas):
    """训练图表画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.rewards = []
        self.success_rates = []
        self.losses = []
        
    def update_plot(self, rewards, success_rates, losses):
        """更新图表"""
        self.ax.clear()
        
        if rewards:
            self.ax.plot(rewards, 'b-', label='奖励', alpha=0.7)
        
        if success_rates:
            window = min(10, len(success_rates))
            if len(success_rates) >= window:
                moving_avg = np.convolve(success_rates, np.ones(window)/window, mode='valid')
                self.ax.plot(range(window-1, len(success_rates)), moving_avg, 
                           'g-', label=f'成功率({window}回合平均)', alpha=0.7)
        
        if losses and len(losses) > 0:
            recent_losses = losses[-min(100, len(losses)):]
            self.ax_twin = self.ax.twinx()
            self.ax_twin.plot(range(len(losses)-len(recent_losses), len(losses)), 
                            recent_losses, 'r-', label='损失', alpha=0.5)
            self.ax_twin.set_ylabel('损失', color='r')
            self.ax_twin.tick_params(axis='y', labelcolor='r')
        
        self.ax.set_xlabel('回合')
        self.ax.set_ylabel('奖励/成功率')
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


# 测试函数
def test_navigation_trainer():
    """测试导航训练器"""
    print("测试导航训练模块...")
    
    trainer = DuelingDQNAStarTrainer()
    
    # 简单测试
    print("导航训练器初始化成功")
    print("可以进行训练或使用现有模型")
    
    # 加载现有模型
    if os.path.exists(Config.dueling_dqn_model_path):
        print(f"找到现有模型: {Config.dueling_dqn_model_path}")
    else:
        print("未找到现有模型，可以开始训练")


if __name__ == "__main__":
    test_navigation_trainer()
