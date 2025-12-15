#!/usr/bin/env python3
"""
Map-based A-Star DQN with Path Memory and Escape Learning
Complete implementation with:
1. 3x speed optimization
2. Path memory system
3. Escape learning for stuck situations
4. Real-time visualization
"""

import os
import sys
import time
import math
import random
import pickle
import heapq
import subprocess
import signal
import rospy
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict, Any, Set
from datetime import datetime
import seaborn as sns

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as GeoPoint, Pose, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA
from squaternion import Quaternion

# 添加gazebo_msgs导入
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

# ========================== 配置参数类 ==========================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    max_timesteps = 5_000_000
    max_episode_steps = 150  # 由于速度提高，减少最大步数
    batch_size = 64
    save_model = True
    load_model = False
    heuristic_weight = 1.5
    
    # ROS topics
    odom_topic = "/p3dx/odom"
    cmd_vel_topic = "/p3dx/cmd_vel"
    laser_topic = "/p3dx/front_laser/scan"
    
    # 地图参数
    laser_max_range = 8.0
    map_resolution = 0.2  # 提高地图分辨率
    map_size = 100  # 增大地图尺寸
    map_origin_x = -10.0  # 扩大地图范围
    map_origin_y = -10.0
    
    # 速度参数 - 三倍速度提升
    max_linear_speed = 1.2  # 三倍于原来的0.4
    max_angular_speed = 3.0  # 三倍于原来的1.0
    
    # 目标位置
    shelf_locations = [
        (1.0, 3.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (5.0, 4.0),
        (5.0, 1.0)
    ]
    charging_station = (-9.0, 7.0)
    
    # 训练参数
    collision_penalty = -30.0  # 由于速度提高，增加碰撞惩罚
    success_reward = 60.0
    step_penalty = -0.02
    progress_reward_weight = 3.0  # 增加进展奖励
    min_obstacle_distance = 0.3  # 减小最小障碍距离
    
    # 地图保存参数
    map_save_path = "saved_maps"
    map_update_frequency = 10  # 每10步更新一次地图
    map_save_frequency = 50  # 每50回合保存一次地图
    
    # 可视化参数
    plot_save_path = "training_plots"
    plot_save_frequency = 10  # 每10回合保存一次图像
    smoothing_window = 10  # 平滑窗口大小
    
    # 新增：路径记忆参数
    path_memory_path = "path_memory"
    path_memory_size = 100  # 记忆的最大路径数
    path_similarity_threshold = 0.8  # 路径相似度阈值
    reuse_memory_chance = 0.7  # 重用记忆路径的概率
    
    # 新增：困境学习参数
    stuck_threshold = 20  # 卡住阈值
    escape_learning_rate = 0.1  # 困境学习率
    escape_memory_size = 50  # 困境记忆大小


# ========================== 训练监控器 ==========================
class TrainingMonitor:
    """训练监控器，用于记录和可视化训练过程"""
    def __init__(self, save_path="training_plots"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # 训练记录
        self.episode_rewards = []  # 每回合奖励
        self.episode_lengths = []  # 每回合步数
        self.episode_successes = []  # 每回合是否成功
        self.episode_collisions = []  # 每回合碰撞次数
        self.episode_q_values = []  # 每回合平均Q值
        self.episode_losses = []  # 每回合平均损失
        self.epsilon_values = []  # 探索率记录
        
        # 累计数据
        self.cumulative_rewards = []
        self.moving_averages = []  # 移动平均奖励
        self.success_rates = []  # 成功率
        self.stability_scores = []  # 稳定性分数
        
        # 时间记录
        self.episode_times = []
        self.start_time = time.time()
        
        # 创建图形
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Map-Based A-Star DQN Training Performance (3x Speed)', fontsize=16, fontweight='bold')
    
    def record_episode(self, episode_idx, reward, steps, success, collisions, 
                      avg_q_value, avg_loss, epsilon):
        """记录一回合的数据"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(steps)
        self.episode_successes.append(1 if success else 0)
        self.episode_collisions.append(collisions)
        self.episode_q_values.append(avg_q_value)
        self.episode_losses.append(avg_loss)
        self.epsilon_values.append(epsilon)
        
        # 计算累计奖励
        if self.cumulative_rewards:
            self.cumulative_rewards.append(self.cumulative_rewards[-1] + reward)
        else:
            self.cumulative_rewards.append(reward)
        
        # 计算移动平均奖励（使用指定窗口）
        window = min(Config.smoothing_window, len(self.episode_rewards))
        if len(self.episode_rewards) >= window:
            moving_avg = np.mean(self.episode_rewards[-window:])
            self.moving_averages.append(moving_avg)
        
        # 计算成功率
        if len(self.episode_successes) >= 20:
            success_rate = np.mean(self.episode_successes[-20:]) * 100
            self.success_rates.append(success_rate)
        
        # 计算稳定性分数（奖励方差）
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            stability = 1.0 / (1.0 + np.std(recent_rewards))
            self.stability_scores.append(stability)
        
        # 记录时间
        self.episode_times.append(time.time() - self.start_time)
    
    def _smooth_data(self, data, window=Config.smoothing_window):
        """平滑数据"""
        if len(data) < window:
            return data
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        # 扩展平滑数据到原始长度
        pad_size = len(data) - len(smoothed)
        smoothed_padded = np.pad(smoothed, (pad_size, 0), 'edge')
        return smoothed_padded
    
    def generate_plots(self, episode_idx, save=True):
        """生成所有训练图像"""
        if len(self.episode_rewards) < 5:
            return  # 数据太少时不生成图像
        
        # 1. 累计奖励图像
        ax = self.axes[0, 0]
        ax.clear()
        episodes = range(1, len(self.cumulative_rewards) + 1)
        ax.plot(episodes, self.cumulative_rewards, 'b-', linewidth=2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward over Episodes', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(self.cumulative_rewards) > 10:
            z = np.polyfit(episodes, self.cumulative_rewards, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), "r--", linewidth=2, alpha=0.8, 
                   label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
            ax.legend()
        
        # 2. 稳定性图像
        ax = self.axes[0, 1]
        ax.clear()
        if len(self.episode_rewards) > 1:
            # 奖励分布
            recent_rewards = self.episode_rewards[-min(100, len(self.episode_rewards)):]
            ax.hist(recent_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(recent_rewards), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(recent_rewards):.2f}')
            ax.axvline(np.median(recent_rewards), color='green', linestyle='--', 
                      linewidth=2, label=f'Median: {np.median(recent_rewards):.2f}')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')
            ax.set_title('Reward Distribution (Stability)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. 收敛速度图像
        ax = self.axes[0, 2]
        ax.clear()
        if len(self.moving_averages) > 1:
            episodes_ma = range(Config.smoothing_window, len(self.episode_rewards) + 1)
            ax.plot(episodes_ma, self.moving_averages, 'g-', linewidth=2, label='Moving Avg')
            
            # 添加原始奖励（透明度较低）
            ax.plot(range(1, len(self.episode_rewards) + 1), self.episode_rewards, 
                   'gray', alpha=0.3, linewidth=1, label='Raw Reward')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Convergence Speed (Moving Average)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 标注收敛点
            if len(self.moving_averages) > 30:
                for threshold in [0.8, 0.9, 0.95]:
                    target = threshold * np.max(self.moving_averages)
                    converged = np.where(np.array(self.moving_averages) >= target)[0]
                    if len(converged) > 0:
                        conv_episode = episodes_ma[converged[0]]
                        ax.axvline(conv_episode, color='orange', linestyle=':', 
                                  alpha=0.7, linewidth=1)
                        ax.text(conv_episode, ax.get_ylim()[0], 
                               f'{threshold*100:.0f}%', fontsize=8)
        
        # 4. 平均得分图像
        ax = self.axes[1, 0]
        ax.clear()
        
        # 创建子图：平均奖励和成功率
        ax2 = ax.twinx()
        
        # 平滑奖励
        smoothed_rewards = self._smooth_data(self.episode_rewards)
        episodes_smoothed = range(1, len(smoothed_rewards) + 1)
        
        # 绘制平均奖励
        line1, = ax.plot(episodes_smoothed, smoothed_rewards, 'b-', linewidth=2, 
                        alpha=0.7, label='Avg Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # 绘制成功率
        if len(self.success_rates) > 0:
            success_episodes = range(20, len(self.episode_successes) + 1)
            line2, = ax2.plot(success_episodes, self.success_rates, 'r-', 
                            linewidth=2, alpha=0.7, label='Success Rate')
            ax2.set_ylabel('Success Rate (%)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim([0, 100])
        
        ax.set_title('Average Score and Success Rate', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 合并图例
        lines = [line1]
        labels = ['Avg Reward']
        if 'line2' in locals():
            lines.append(line2)
            labels.append('Success Rate')
        ax.legend(lines, labels, loc='upper left')
        
        # 5. 平均Q值图像
        ax = self.axes[1, 1]
        ax.clear()
        if len(self.episode_q_values) > 1:
            episodes_q = range(1, len(self.episode_q_values) + 1)
            ax.plot(episodes_q, self.episode_q_values, 'purple', linewidth=2, 
                   label='Average Q-value')
            
            # 添加探索率（次坐标轴）
            ax2 = ax.twinx()
            ax2.plot(episodes_q, self.epsilon_values, 'orange', linewidth=1.5, 
                    alpha=0.7, label='Epsilon')
            ax2.set_ylabel('Exploration Rate', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.set_ylim([0, 1.1])
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Q-value', color='purple')
            ax.set_title('Q-value and Exploration Rate', fontweight='bold')
            ax.tick_params(axis='y', labelcolor='purple')
            ax.grid(True, alpha=0.3)
            
            # 合并图例
            lines = ax.get_lines() + ax2.get_lines()
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
        
        # 6. 综合性能图像
        ax = self.axes[1, 2]
        ax.clear()
        
        # 准备数据
        metrics = ['Reward', 'Steps', 'Collisions']
        recent_data = {
            'Reward': self.episode_rewards[-min(20, len(self.episode_rewards)):],
            'Steps': self.episode_lengths[-min(20, len(self.episode_lengths)):],
            'Collisions': self.episode_collisions[-min(20, len(self.episode_collisions)):]
        }
        
        # 计算箱线图数据
        data_to_plot = [recent_data[metric] for metric in metrics if recent_data[metric]]
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=metrics, patch_artist=True)
            
            # 美化箱线图
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            for whisker in bp['whiskers']:
                whisker.set(color='gray', linewidth=1.5, linestyle='--')
            
            for median in bp['medians']:
                median.set(color='red', linewidth=2)
            
            # 添加数据点
            for i, metric in enumerate(metrics):
                if i < len(data_to_plot):
                    y = data_to_plot[i]
                    x = np.random.normal(i + 1, 0.04, size=len(y))
                    ax.plot(x, y, 'o', alpha=0.4, markersize=4)
            
            ax.set_ylabel('Value')
            ax.set_title('Performance Metrics Distribution (Recent 20 Episodes)', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 调整布局
        plt.tight_layout()
        
        if save and episode_idx % Config.plot_save_frequency == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_path, f"training_plots_ep{episode_idx}_{timestamp}.png")
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Training plots saved to {filename}")
        
        # 实时显示
        plt.pause(0.01)
    
    def generate_summary_report(self):
        """生成训练总结报告"""
        if not self.episode_rewards:
            return
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY REPORT")
        print("="*60)
        
        total_episodes = len(self.episode_rewards)
        total_time = time.time() - self.start_time
        avg_episode_time = np.mean(self.episode_times) if self.episode_times else 0
        
        print(f"Total Episodes: {total_episodes}")
        print(f"Total Training Time: {total_time:.1f} seconds")
        print(f"Average Episode Time: {avg_episode_time:.1f} seconds")
        print(f"Average Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Median Reward: {np.median(self.episode_rewards):.2f}")
        print(f"Std Reward: {np.std(self.episode_rewards):.2f}")
        print(f"Max Reward: {np.max(self.episode_rewards):.2f}")
        print(f"Min Reward: {np.min(self.episode_rewards):.2f}")
        print(f"Success Rate: {np.mean(self.episode_successes)*100:.1f}%")
        print(f"Average Episode Length: {np.mean(self.episode_lengths):.1f} steps")
        print(f"Average Collisions per Episode: {np.mean(self.episode_collisions):.2f}")
        print(f"Final Exploration Rate: {self.epsilon_values[-1]:.3f}")
        
        # 计算收敛指标
        if len(self.moving_averages) > 0:
            print(f"Final Moving Average Reward: {self.moving_averages[-1]:.2f}")
            
            # 收敛到80%最大性能的回合数
            if np.max(self.moving_averages) > 0:
                eighty_percent = 0.8 * np.max(self.moving_averages)
                converged_episodes = np.where(np.array(self.moving_averages) >= eighty_percent)[0]
                if len(converged_episodes) > 0:
                    convergence_episode = converged_episodes[0] + Config.smoothing_window
                    print(f"Convergence to 80% max performance: Episode {convergence_episode}")
        
        print("="*60)
        
        # 保存总结报告
        report_filename = os.path.join(self.save_path, "training_summary.txt")
        with open(report_filename, 'w') as f:
            f.write("Training Summary Report\n")
            f.write("="*50 + "\n")
            f.write(f"Total Episodes: {total_episodes}\n")
            f.write(f"Total Training Time: {total_time:.1f} seconds\n")
            f.write(f"Average Reward: {np.mean(self.episode_rewards):.2f}\n")
            f.write(f"Success Rate: {np.mean(self.episode_successes)*100:.1f}%\n")
            f.write(f"Final Exploration Rate: {self.epsilon_values[-1]:.3f}\n")
        
        # 生成最终可视化
        final_plot_filename = os.path.join(self.save_path, "final_training_analysis.png")
        self._generate_final_analysis(final_plot_filename)
    
    def _generate_final_analysis(self, filename):
        """生成最终分析图像"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Final Training Analysis Report', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # 1. 学习曲线
        ax = axes[0, 0]
        if episodes:
            ax.plot(episodes, self.episode_rewards, 'b-', alpha=0.5, label='Raw Reward')
            
            # 只有在有足够数据时才绘制移动平均
            if len(self.moving_averages) > 0 and len(self.episode_rewards) >= Config.smoothing_window:
                ma_episodes = range(Config.smoothing_window, len(self.episode_rewards) + 1)
                # 确保移动平均的长度与ma_episodes匹配
                if len(ma_episodes) == len(self.moving_averages):
                    ax.plot(ma_episodes, self.moving_averages, 'r-', linewidth=2, 
                           label=f'{Config.smoothing_window}-episode Moving Avg')
                else:
                    # 如果长度不匹配，重新计算移动平均
                    if len(self.episode_rewards) >= Config.smoothing_window:
                        moving_avg = np.convolve(self.episode_rewards, 
                                               np.ones(Config.smoothing_window)/Config.smoothing_window, 
                                               mode='valid')
                        ma_episodes = range(Config.smoothing_window, len(self.episode_rewards) + 1)
                        ax.plot(ma_episodes, moving_avg, 'r-', linewidth=2, 
                               label=f'{Config.smoothing_window}-episode Moving Avg')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Learning Curve')
        if episodes:  # 只在有数据时添加图例
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 累计奖励
        ax = axes[0, 1]
        if episodes and len(self.cumulative_rewards) == len(episodes):
            ax.plot(episodes, self.cumulative_rewards, 'g-', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title('Cumulative Reward Growth')
            ax.grid(True, alpha=0.3)
        
        # 3. 成功率和探索率
        ax = axes[1, 0]
        if len(self.success_rates) > 0 and len(self.episode_successes) >= 20:
            success_episodes = range(20, len(self.episode_successes) + 1)
            # 确保success_rates与success_episodes长度匹配
            if len(success_episodes) == len(self.success_rates):
                ax.plot(success_episodes, self.success_rates, 'b-', linewidth=2, 
                       label='Success Rate')
        if episodes and len(self.epsilon_values) == len(episodes):
            ax2 = ax.twinx()
            ax2.plot(episodes, self.epsilon_values, 'r--', linewidth=1.5, 
                    label='Exploration Rate', alpha=0.7)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Success Rate (%)', color='b')
            ax2.set_ylabel('Exploration Rate', color='r')
            ax.set_title('Success Rate vs Exploration')
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            lines = ax.get_lines() + ax2.get_lines()
            labels = [l.get_label() for l in lines]
            if lines:
                ax.legend(lines, labels, loc='best')
        
        # 4. Q值和损失
        ax = axes[1, 1]
        if len(self.episode_q_values) > 0:
            episodes_q = range(1, len(self.episode_q_values) + 1)
            if len(episodes_q) == len(self.episode_q_values):
                ax.plot(episodes_q, self.episode_q_values, 'purple', linewidth=2, 
                       label='Avg Q-value')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Average Q-value')
                ax.set_title('Q-value Progression')
                ax.grid(True, alpha=0.3)
        
        # 5. 性能指标分布
        ax = axes[2, 0]
        metrics = ['Reward', 'Steps', 'Collisions']
        
        # 收集最近的数据（最多50个回合）
        recent_rewards = self.episode_rewards[-min(50, len(self.episode_rewards)):] if self.episode_rewards else []
        recent_lengths = self.episode_lengths[-min(50, len(self.episode_lengths)):] if self.episode_lengths else []
        recent_collisions = self.episode_collisions[-min(50, len(self.episode_collisions)):] if self.episode_collisions else []
        
        recent_data = []
        if recent_rewards:
            recent_data.append(recent_rewards)
        if recent_lengths:
            recent_data.append(recent_lengths)
        if recent_collisions:
            recent_data.append(recent_collisions)
        
        if recent_data:
            # 调整标签以匹配实际数据
            labels_to_use = []
            if recent_rewards:
                labels_to_use.append('Reward')
            if recent_lengths:
                labels_to_use.append('Steps')
            if recent_collisions:
                labels_to_use.append('Collisions')
            
            ax.boxplot(recent_data, labels=labels_to_use)
            ax.set_title('Performance Metrics Distribution')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 6. 时间分析
        ax = axes[2, 1]
        if episodes and len(self.episode_times) == len(episodes):
            ax.plot(episodes, self.episode_times, 'orange', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Episode Duration')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Final analysis saved to {filename}")
        plt.close(fig)  # 关闭图像以释放内存


# ========================== 路径记忆系统 ==========================
class PathMemory:
    """路径记忆系统 - 记住成功路径和困境解决方案"""
    def __init__(self, save_path="path_memory"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # 成功路径记忆：起点 -> 终点 -> 路径列表
        self.success_paths = defaultdict(lambda: defaultdict(list))
        
        # 困境解决方案：困境状态 -> 解决方案
        self.escape_solutions = {}
        
        # 路径统计信息
        self.path_usage_count = defaultdict(int)
        self.path_success_rate = defaultdict(float)
        
        # 加载现有记忆
        self.load_memory()
    
    def add_success_path(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float], 
                         path: List[Tuple[float, float]], success_info: Dict = None):
        """添加成功路径到记忆"""
        start_key = self._quantize_position(start_pos)
        goal_key = self._quantize_position(goal_pos)
        
        # 检查是否已有相似路径
        existing_paths = self.success_paths[start_key][goal_key]
        for existing_path in existing_paths:
            if self._calculate_path_similarity(path, existing_path['path']) > Config.path_similarity_threshold:
                # 更新现有路径的成功率
                existing_path['success_count'] += 1
                existing_path['total_usage'] += 1
                existing_path['last_used'] = time.time()
                if success_info:
                    existing_path['success_info'] = success_info
                return
        
        # 添加新路径
        path_entry = {
            'path': path,
            'success_count': 1,
            'total_usage': 1,
            'first_used': time.time(),
            'last_used': time.time(),
            'length': len(path),
            'success_info': success_info or {}
        }
        
        self.success_paths[start_key][goal_key].append(path_entry)
        
        # 限制每个起点-终点对的路径数量
        if len(self.success_paths[start_key][goal_key]) > 5:
            self.success_paths[start_key][goal_key].sort(key=lambda x: x['success_count']/x['total_usage'], reverse=True)
            self.success_paths[start_key][goal_key] = self.success_paths[start_key][goal_key][:5]
        
        print(f"Path memory: Added path from {start_pos} to {goal_pos} ({len(path)} waypoints)")
    
    def get_best_path(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float], 
                      occupancy_map=None) -> Optional[List[Tuple[float, float]]]:
        """获取最佳记忆路径"""
        start_key = self._quantize_position(start_pos)
        goal_key = self._quantize_position(goal_pos)
        
        if start_key not in self.success_paths or goal_key not in self.success_paths[start_key]:
            return None
        
        paths = self.success_paths[start_key][goal_key]
        if not paths:
            return None
        
        # 按成功率排序
        valid_paths = []
        for path_entry in paths:
            if self._check_path_validity(path_entry['path'], occupancy_map):
                # 计算综合评分：成功率 * 时效性
                time_factor = 1.0 / (1.0 + (time.time() - path_entry['last_used']) / 3600)  # 小时衰减
                score = (path_entry['success_count'] / path_entry['total_usage']) * time_factor
                valid_paths.append((score, path_entry['path']))
        
        if not valid_paths:
            return None
        
        # 返回评分最高的路径
        valid_paths.sort(key=lambda x: x[0], reverse=True)
        return valid_paths[0][1]
    
    def add_escape_solution(self, stuck_state: Tuple[float, float, float], 
                            solution_actions: List[Tuple[float, float]], success: bool):
        """添加困境解决方案"""
        state_key = self._quantize_stuck_state(stuck_state)
        
        if state_key not in self.escape_solutions:
            self.escape_solutions[state_key] = {
                'solutions': [],
                'success_count': 0,
                'total_attempts': 0
            }
        
        entry = {
            'actions': solution_actions,
            'success': success,
            'timestamp': time.time()
        }
        
        self.escape_solutions[state_key]['solutions'].append(entry)
        self.escape_solutions[state_key]['total_attempts'] += 1
        if success:
            self.escape_solutions[state_key]['success_count'] += 1
        
        # 限制解决方案数量
        if len(self.escape_solutions[state_key]['solutions']) > Config.escape_memory_size:
            self.escape_solutions[state_key]['solutions'] = \
                self.escape_solutions[state_key]['solutions'][-Config.escape_memory_size:]
        
        # 如果成功率太低，清除记忆
        success_rate = self.escape_solutions[state_key]['success_count'] / \
                      self.escape_solutions[state_key]['total_attempts']
        if success_rate < 0.1 and self.escape_solutions[state_key]['total_attempts'] > 10:
            del self.escape_solutions[state_key]
    
    def get_escape_solution(self, stuck_state: Tuple[float, float, float]) -> Optional[List[Tuple[float, float]]]:
        """获取困境解决方案"""
        state_key = self._quantize_stuck_state(stuck_state)
        
        if state_key not in self.escape_solutions:
            return None
        
        solutions = self.escape_solutions[state_key]['solutions']
        if not solutions:
            return None
        
        # 优先选择成功的解决方案
        successful = [s for s in solutions if s['success']]
        if successful:
            # 选择最近的成功解决方案
            successful.sort(key=lambda x: x['timestamp'], reverse=True)
            return successful[0]['actions']
        
        # 如果没有成功方案，返回最近尝试的方案
        solutions.sort(key=lambda x: x['timestamp'], reverse=True)
        return solutions[0]['actions']
    
    def _quantize_position(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """量化位置（用于记忆索引）"""
        x, y = pos
        return (int(round(x * 2)), int(round(y * 2)))  # 0.5米精度
    
    def _quantize_stuck_state(self, state: Tuple[float, float, float]) -> str:
        """量化困境状态"""
        x, y, heading = state
        # 使用更粗略的量化
        quantized_x = int(round(x))
        quantized_y = int(round(y))
        quantized_heading = int(round(heading / (math.pi / 4))) % 8  # 8个方向
        return f"{quantized_x}_{quantized_y}_{quantized_heading}"
    
    def _calculate_path_similarity(self, path1: List[Tuple[float, float]], 
                                  path2: List[Tuple[float, float]]) -> float:
        """计算两条路径的相似度"""
        if not path1 or not path2:
            return 0.0
        
        # 使用动态时间规整(DTW)计算路径相似度
        def dtw_distance(seq1, seq2):
            n, m = len(seq1), len(seq2)
            dtw_matrix = np.zeros((n+1, m+1))
            dtw_matrix[0, 1:] = float('inf')
            dtw_matrix[1:, 0] = float('inf')
            
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = np.linalg.norm(np.array(seq1[i-1]) - np.array(seq2[j-1]))
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], 
                                                 dtw_matrix[i, j-1], 
                                                 dtw_matrix[i-1, j-1])
            return dtw_matrix[n, m]
        
        try:
            distance = dtw_distance(path1, path2)
            max_len = max(len(path1), len(path2))
            normalized_distance = distance / (max_len * 2.0)  # 假设最大距离为2米
            similarity = 1.0 / (1.0 + normalized_distance)
            return similarity
        except:
            return 0.0
    
    def _check_path_validity(self, path: List[Tuple[float, float]], 
                            occupancy_map) -> bool:
        """检查路径在当前地图中是否有效"""
        if not path or occupancy_map is None:
            return True
        
        for point in path:
            grid_x, grid_y = occupancy_map.world_to_grid(*point)
            if occupancy_map.is_occupied(grid_x, grid_y):
                return False
        
        return True
    
    def save_memory(self):
        """保存记忆到文件"""
        memory_data = {
            'success_paths': dict(self.success_paths),
            'escape_solutions': self.escape_solutions,
            'path_usage_count': dict(self.path_usage_count),
            'path_success_rate': dict(self.path_success_rate),
            'save_time': time.time()
        }
        
        filepath = os.path.join(self.save_path, "path_memory.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(memory_data, f)
        
        print(f"Path memory saved: {len(self.success_paths)} start positions, "
              f"{len(self.escape_solutions)} escape solutions")
    
    def load_memory(self):
        """从文件加载记忆"""
        filepath = os.path.join(self.save_path, "path_memory.pkl")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    memory_data = pickle.load(f)
                
                self.success_paths = defaultdict(lambda: defaultdict(list), 
                                                memory_data.get('success_paths', {}))
                self.escape_solutions = memory_data.get('escape_solutions', {})
                self.path_usage_count = defaultdict(int, 
                                                  memory_data.get('path_usage_count', {}))
                self.path_success_rate = defaultdict(float, 
                                                   memory_data.get('path_success_rate', {}))
                
                print(f"Path memory loaded: {len(self.success_paths)} start positions, "
                      f"{len(self.escape_solutions)} escape solutions")
                return True
            except Exception as e:
                print(f"Failed to load path memory: {e}")
        
        return False
    
    def get_statistics(self) -> Dict:
        """获取记忆统计信息"""
        total_paths = sum(len(paths) for goals in self.success_paths.values() 
                         for paths in goals.values())
        
        return {
            'total_start_positions': len(self.success_paths),
            'total_paths': total_paths,
            'escape_solutions': len(self.escape_solutions),
            'memory_size_mb': self._estimate_memory_size() / (1024 * 1024)
        }
    
    def _estimate_memory_size(self) -> int:
        """估计内存使用量"""
        import sys
        return sys.getsizeof(self.success_paths) + sys.getsizeof(self.escape_solutions)


# ========================== 困境学习器 ==========================
class EscapeLearner:
    """困境学习器 - 学习如何从困境中摆脱"""
    def __init__(self):
        self.stuck_states = []  # 记录困境状态
        self.escape_sequences = []  # 摆脱困境的动作序列
        self.successful_escapes = []  # 成功的摆脱方案
        
        # 困境检测参数
        self.stuck_duration = 0  # 卡住持续时间
        self.last_progress = 0  # 上次进展
        self.stuck_start_time = 0  # 卡住开始时间
        
        # 学习参数
        self.learning_rate = Config.escape_learning_rate
        self.exploration_rate = 0.3  # 探索新解决方案的概率
        
        # 状态历史
        self.position_history = deque(maxlen=20)
        self.laser_data = None
    
    def detect_stuck(self, robot_x: float, robot_y: float, robot_heading: float,
                    min_laser: float, progress: float, time_elapsed: float) -> Tuple[bool, Dict]:
        """检测是否陷入困境"""
        stuck_info = {
            'type': None,
            'severity': 0,
            'suggested_action': None
        }
        
        # 检测不同类型的困境
        situations = []
        
        # 1. 完全卡住（长时间无进展）
        if progress < 0.05 and time_elapsed > 5.0:
            situations.append(('complete_stuck', 0.9))
        
        # 2. 靠近障碍物
        if min_laser < 0.4:
            situations.append(('near_obstacle', 0.8 - min_laser))
        
        # 3. 原地打转（通过位置历史检测）
        if len(self.position_history) > 10:
            # 计算移动距离的标准差
            movements = []
            for i in range(1, len(self.position_history)):
                dist = math.hypot(self.position_history[i][0] - self.position_history[i-1][0],
                                self.position_history[i][1] - self.position_history[i-1][1])
                movements.append(dist)
            
            if np.std(movements) < 0.1 and np.mean(movements) < 0.05:
                situations.append(('spinning', 0.7))
        
        if not situations:
            self.stuck_duration = max(0, self.stuck_duration - 1)
            return False, stuck_info
        
        # 选择最严重的困境
        situations.sort(key=lambda x: x[1], reverse=True)
        stuck_type, severity = situations[0]
        
        self.stuck_duration += 1
        stuck_info['type'] = stuck_type
        stuck_info['severity'] = severity
        
        # 根据困境类型建议动作
        if stuck_type == 'complete_stuck':
            stuck_info['suggested_action'] = 'backup_and_turn'
        elif stuck_type == 'near_obstacle':
            stuck_info['suggested_action'] = 'sideways_movement'
        elif stuck_type == 'spinning':
            stuck_info['suggested_action'] = 'stop_and_reassess'
        
        return self.stuck_duration > Config.stuck_threshold, stuck_info
    
    def generate_escape_sequence(self, stuck_info: Dict, current_state: Tuple) -> List[Tuple[float, float]]:
        """生成摆脱困境的动作序列"""
        escape_type = stuck_info.get('type', 'unknown')
        severity = stuck_info.get('severity', 0.5)
        
        # 基本摆脱策略
        if escape_type == 'complete_stuck':
            # 后退并大角度转向
            return [
                (-0.4, 0.0),        # 后退
                (-0.3, 1.5),        # 后退左转
                (0.0, 2.0),         # 原地左转
                (0.3, 0.0),         # 前进
            ]
        elif escape_type == 'near_obstacle':
            # 横向移动远离障碍物
            # 根据激光数据选择方向
            if self.laser_data is not None and hasattr(self.laser_data, 'ranges'):
                laser_ranges = self.laser_data.ranges
                if laser_ranges:
                    left_avg = np.mean(laser_ranges[:len(laser_ranges)//3])
                    right_avg = np.mean(laser_ranges[2*len(laser_ranges)//3:])
                    turn_direction = 1.0 if left_avg > right_avg else -1.0
                else:
                    turn_direction = 1.0 if random.random() > 0.5 else -1.0
                
                return [
                    (-0.2, 0.0),                    # 轻微后退
                    (0.0, turn_direction * 1.5),    # 转向
                    (0.3, 0.0),                     # 前进
                ]
        elif escape_type == 'spinning':
            # 停止并重新评估
            return [
                (0.0, 0.0),         # 完全停止
                (0.0, 0.5),         # 小角度转向
                (0.2, 0.0),         # 缓慢前进
            ]
        
        # 默认摆脱策略：随机探索
        return self._generate_random_escape()
    
    def _generate_random_escape(self) -> List[Tuple[float, float]]:
        """生成随机摆脱序列"""
        sequence = []
        for _ in range(random.randint(3, 6)):
            # 随机选择动作
            if random.random() < 0.5:
                # 转向动作
                linear = random.uniform(-0.1, 0.1)
                angular = random.uniform(-2.0, 2.0)
            else:
                # 直线移动
                linear = random.uniform(-0.3, 0.5)
                angular = random.uniform(-0.5, 0.5)
            
            sequence.append((linear, angular))
        
        return sequence
    
    def learn_from_escape(self, initial_state: Tuple, escape_sequence: List[Tuple[float, float]],
                         final_state: Tuple, success: bool, progress_improvement: float):
        """从摆脱经验中学习"""
        learning_entry = {
            'initial_state': initial_state,
            'escape_sequence': escape_sequence,
            'final_state': final_state,
            'success': success,
            'progress_improvement': progress_improvement,
            'timestamp': time.time()
        }
        
        if success and progress_improvement > 0.1:
            self.successful_escapes.append(learning_entry)
            # 保留最佳解决方案
            if len(self.successful_escapes) > 20:
                self.successful_escapes.sort(key=lambda x: x['progress_improvement'], reverse=True)
                self.successful_escapes = self.successful_escapes[:20]
            
            print(f"Learned successful escape (improvement: {progress_improvement:.2f})")
        
        # 更新探索率
        if success:
            self.exploration_rate = max(0.1, self.exploration_rate * 0.95)
        else:
            self.exploration_rate = min(0.8, self.exploration_rate * 1.05)
    
    def get_best_escape(self, current_state: Tuple, path_memory: PathMemory = None) -> Optional[List[Tuple[float, float]]]:
        """获取最佳摆脱方案"""
        # 首先尝试路径记忆
        if path_memory:
            memory_solution = path_memory.get_escape_solution(current_state)
            if memory_solution and random.random() > self.exploration_rate:
                return memory_solution
        
        # 然后尝试已学习的解决方案
        if self.successful_escapes:
            # 寻找类似状态的解决方案
            similar_escapes = []
            for escape in self.successful_escapes:
                similarity = self._calculate_state_similarity(current_state, escape['initial_state'])
                if similarity > 0.7:  # 相似度阈值
                    similar_escapes.append((similarity, escape))
            
            if similar_escapes:
                similar_escapes.sort(key=lambda x: x[0], reverse=True)
                return similar_escapes[0][1]['escape_sequence']
        
        return None
    
    def _calculate_state_similarity(self, state1: Tuple, state2: Tuple) -> float:
        """计算状态相似度"""
        if len(state1) != len(state2):
            return 0.0
        
        # 位置相似度
        pos_sim = 1.0 / (1.0 + math.hypot(state1[0] - state2[0], state1[1] - state2[1]))
        
        # 航向相似度
        if len(state1) > 2:
            heading_diff = abs(state1[2] - state2[2])
            heading_sim = 1.0 / (1.0 + heading_diff)
            return (pos_sim + heading_sim) / 2.0
        
        return pos_sim
    
    def update_state_history(self, robot_x: float, robot_y: float, laser_data=None):
        """更新状态历史"""
        self.position_history.append((robot_x, robot_y))
        self.laser_data = laser_data
    
    def reset_stuck_detection(self):
        """重置困境检测"""
        self.stuck_duration = 0
        self.last_progress = 0
        self.stuck_start_time = 0


# ========================== RViz可视化类 ==========================
class RVizVisualizer:
    """RViz可视化类"""
    def __init__(self, path_memory=None):
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10, latch=True)
        self.path_pub = rospy.Publisher('/navigation_path', Marker, queue_size=10, latch=True)
        self.status_pub = rospy.Publisher('/training_status', Marker, queue_size=10, latch=True)
        self.memory_pub = rospy.Publisher('/memory_paths', MarkerArray, queue_size=10, latch=True)
        
        self.path_memory = path_memory
        
        time.sleep(1)
        print("RViz visualizer initialized")
    
    def clear_all_markers(self):
        """清除所有标记"""
        try:
            marker_array = MarkerArray()
            
            clear_marker = Marker()
            clear_marker.header.frame_id = "odom"
            clear_marker.ns = "navigation_goals"
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            
            clear_marker2 = Marker()
            clear_marker2.header.frame_id = "odom"
            clear_marker2.ns = "navigation_path"
            clear_marker2.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker2)
            
            clear_marker3 = Marker()
            clear_marker3.header.frame_id = "odom"
            clear_marker3.ns = "training_status"
            clear_marker3.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker3)
            
            clear_marker4 = Marker()
            clear_marker4.header.frame_id = "odom"
            clear_marker4.ns = "memory_paths"
            clear_marker4.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker4)
            
            self.marker_pub.publish(marker_array)
            time.sleep(0.2)
            print("Cleared all RViz markers")
        except Exception as e:
            print(f"Error clearing markers: {e}")
    
    def visualize_goals(self, shelf_locations, charging_station, current_goal=None, goal_type=None):
        """可视化所有目标点"""
        try:
            marker_array = MarkerArray()
            
            # 可视化货架点
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
                marker.pose.position.z = 0.2
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.4
                marker.scale.y = 0.4
                marker.scale.z = 0.4
                
                # 检查是否是当前目标
                is_current_goal = (current_goal is not None and 
                                 abs(x - current_goal[0]) < 0.01 and 
                                 abs(y - current_goal[1]) < 0.01)
                
                if is_current_goal:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.9
                else:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.7
                
                marker.lifetime = rospy.Duration(0)
                marker_array.markers.append(marker)
            
            # 可视化充电桩
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
            
            is_charging_goal = (current_goal is not None and 
                              abs(charging_station[0] - current_goal[0]) < 0.01 and 
                              abs(charging_station[1] - current_goal[1]) < 0.01 and
                              goal_type == "charging")
            
            if is_charging_goal:
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
            
            self.marker_pub.publish(marker_array)
            
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
            
            marker.scale.x = 0.08
            marker.color.r = 1.0
            marker.color.g = 0.6
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            for point in path_points:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0.1
                marker.points.append(p)
            
            marker.lifetime = rospy.Duration(0)
            self.path_pub.publish(marker)
            
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
            
            marker.scale.z = 0.4
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
            marker.lifetime = rospy.Duration(0)
            self.status_pub.publish(marker)
            
        except Exception as e:
            print(f"Error visualizing training status: {e}")
    
    def visualize_map(self, occupancy_grid):
        """可视化占据栅格地图"""
        try:
            marker_array = MarkerArray()
            
            # 清除旧的地图标记
            clear_marker = Marker()
            clear_marker.header.frame_id = "odom"
            clear_marker.ns = "occupancy_grid"
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            
            # 添加新的地图标记
            marker_id = 0
            for i in range(occupancy_grid.size):
                for j in range(occupancy_grid.size):
                    if occupancy_grid.grid[i, j] > 0.5:  # 只显示高概率障碍物
                        world_x, world_y = occupancy_grid.grid_to_world(i, j)
                        
                        marker = Marker()
                        marker.header.frame_id = "odom"
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "occupancy_grid"
                        marker.id = marker_id
                        marker.type = Marker.CUBE
                        marker.action = Marker.ADD
                        
                        marker.pose.position.x = world_x
                        marker.pose.position.y = world_y
                        marker.pose.position.z = 0.05
                        marker.pose.orientation.w = 1.0
                        
                        # 根据占据概率设置颜色
                        probability = occupancy_grid.grid[i, j]
                        marker.scale.x = occupancy_grid.resolution * 0.8
                        marker.scale.y = occupancy_grid.resolution * 0.8
                        marker.scale.z = 0.05
                        
                        marker.color.r = probability
                        marker.color.g = 0.0
                        marker.color.b = 1.0 - probability
                        marker.color.a = 0.6
                        
                        marker.lifetime = rospy.Duration(1.0)  # 1秒后自动消失
                        marker_array.markers.append(marker)
                        marker_id += 1
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            print(f"Error visualizing map: {e}")
    
    def visualize_memory_paths(self, start_pos=None, goal_pos=None):
        """可视化记忆中的路径"""
        if self.path_memory is None:
            return
        
        try:
            marker_array = MarkerArray()
            
            # 清除旧的内存路径标记
            clear_marker = Marker()
            clear_marker.header.frame_id = "odom"
            clear_marker.ns = "memory_paths"
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            
            # 获取要显示的路径
            paths_to_show = []
            
            if start_pos and goal_pos:
                # 显示特定起点到终点的路径
                start_key = self._quantize_position(start_pos)
                goal_key = self._quantize_position(goal_pos)
                
                if start_key in self.path_memory.success_paths and \
                   goal_key in self.path_memory.success_paths[start_key]:
                    for path_entry in self.path_memory.success_paths[start_key][goal_key]:
                        paths_to_show.append(path_entry['path'])
            
            else:
                # 显示所有高成功率路径
                for start_key, goals in self.path_memory.success_paths.items():
                    for goal_key, paths in goals.items():
                        for path_entry in paths:
                            if path_entry['success_count'] / path_entry['total_usage'] > 0.7:
                                paths_to_show.append(path_entry['path'])
            
            # 可视化路径
            for i, path in enumerate(paths_to_show):
                if len(path) < 2:
                    continue
                
                marker = Marker()
                marker.header.frame_id = "odom"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "memory_paths"
                marker.id = i
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                
                marker.scale.x = 0.05
                marker.color.r = 0.0
                marker.color.g = 0.8
                marker.color.b = 0.8
                marker.color.a = 0.4
                
                for point in path:
                    p = Point()
                    p.x = point[0]
                    p.y = point[1]
                    p.z = 0.05
                    marker.points.append(p)
                
                marker.lifetime = rospy.Duration(5.0)
                marker_array.markers.append(marker)
            
            if len(marker_array.markers) > 1:  # 大于1是因为有一个清除标记
                self.memory_pub.publish(marker_array)
                
        except Exception as e:
            print(f"Error visualizing memory paths: {e}")
    
    def _quantize_position(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """量化位置（与PathMemory一致）"""
        x, y = pos
        return (int(round(x * 2)), int(round(y * 2)))


# ========================== Gazebo控制器 ==========================
class GazeboController:
    """Gazebo环境控制器"""
    def __init__(self):
        self.set_model_state = None
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
            self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            print("Gazebo set_model_state service connected successfully")
        except rospy.ServiceException as e:
            print(f"Failed to connect to Gazebo set_model_state service: {e}")
            self.set_model_state = None
        
    def reset_robot_pose(self, x, y, theta=0.0):
        """重置机器人位姿"""
        try:
            if self.set_model_state is None:
                print("set_model_state service not available")
                return False
                
            model_state = ModelState()
            model_state.model_name = "p3dx"
            model_state.pose.position.x = x
            model_state.pose.position.y = y
            model_state.pose.position.z = 0.02
            
            q = Quaternion.from_euler(0, 0, theta)
            model_state.pose.orientation.x = q.x
            model_state.pose.orientation.y = q.y
            model_state.pose.orientation.z = q.z
            model_state.pose.orientation.w = q.w
            
            response = self.set_model_state(model_state)
            if response.success:
                print(f"Robot pose reset to ({x:.2f}, {y:.2f}, {theta:.2f})")
                time.sleep(0.5)
                return True
            else:
                print("Failed to reset robot pose via set_model_state")
                return False
        except Exception as e:
            print(f"Failed to reset robot pose: {e}")
            return False


# ========================== 清理函数 ==========================
def cleanup_ros():
    """清理ROS进程"""
    try:
        subprocess.run(['pkill', '-9', '-f', 'odometry_monitor'], timeout=5)
        subprocess.run(['pkill', '-9', '-f', 'laser_monitor'], timeout=5)
        subprocess.run(['pkill', '-9', '-f', 'map_based_astar_dqn_trainer'], timeout=5)
        time.sleep(1)
        print("Training-related ROS processes cleaned up")
    except Exception as e:
        print(f"Cleanup warning: {e}")


# ========================== 随机种子设置 ==========================
def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ========================== 里程计监视器 ==========================
class OdometryMonitor:
    """里程计数据监视器"""
    def __init__(self, odom_topic: str = Config.odom_topic):
        self.odom_data = None
        self.odom_received = False
        self.odom_sub = None
        self.odom_topic = odom_topic
        
    def start_monitoring(self):
        """开始监听里程计数据"""
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
        """里程计数据回调函数"""
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
            if int(time.time() - start_time) % 5 == 0:
                elapsed = int(time.time() - start_time)
                print(f"Still waiting for odometry... {elapsed}s elapsed")
        
        print(f"Timeout: No odometry data received after {timeout} seconds")
        return False
    
    def get_odom_data(self):
        """获取当前里程计数据"""
        return self.odom_data
    
    def stop_monitoring(self):
        """停止监听"""
        if self.odom_sub is not None:
            self.odom_sub.unregister()
            self.odom_sub = None


# ========================== 激光雷达监视器 ==========================
class LaserMonitor:
    """激光雷达监视器"""
    def __init__(self, laser_topic: str = Config.laser_topic):
        self.laser_data = None
        self.laser_received = False
        self.laser_sub = None
        self.laser_topic = laser_topic
        
    def start_monitoring(self):
        """开始监听激光雷达数据"""
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
        """激光雷达数据回调函数"""
        self.laser_data = msg
        self.laser_received = True
    
    def wait_for_laser(self, timeout=30.0):
        """等待接收到激光雷达数据"""
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
        """获取当前激光雷达数据"""
        return self.laser_data
    
    def stop_monitoring(self):
        """停止监听"""
        if self.laser_sub is not None:
            self.laser_sub.unregister()
            self.laser_sub = None


# ========================== 占据栅格地图 ==========================
class OccupancyGridMap:
    """改进的占据栅格地图系统"""
    def __init__(self, resolution=0.2, size=100, origin=(-10.0, -10.0)):
        self.resolution = resolution
        self.size = size
        self.origin = np.array(origin)
        
        # 使用概率网格（0-1之间的值）
        self.grid = np.zeros((size, size), dtype=np.float32)
        
        # 地图统计
        self.obstacle_cells = set()
        self.free_cells = set()
        
        # 地图边界
        self.min_x = origin[0]
        self.min_y = origin[1]
        self.max_x = origin[0] + size * resolution
        self.max_y = origin[1] + size * resolution
        
        print(f"Occupancy grid map initialized: {size}x{size}, resolution: {resolution}m")
        print(f"World bounds: [{self.min_x:.1f}, {self.max_x:.1f}] x [{self.min_y:.1f}, {self.max_y:.1f}]")
    
    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """将世界坐标转换为网格坐标"""
        grid_x = int((world_x - self.origin[0]) / self.resolution)
        grid_y = int((world_y - self.origin[1]) / self.resolution)
        grid_x = np.clip(grid_x, 0, self.size - 1)
        grid_y = np.clip(grid_y, 0, self.size - 1)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """将网格坐标转换为世界坐标"""
        world_x = self.origin[0] + (grid_x + 0.5) * self.resolution
        world_y = self.origin[1] + (grid_y + 0.5) * self.resolution
        return world_x, world_y
    
    def update_from_laser(self, robot_x: float, robot_y: float, robot_theta: float, 
                         laser_data: LaserScan, max_range: float = 8.0):
        """使用激光雷达数据更新地图（概率更新）"""
        if laser_data is None:
            return
        
        ranges = np.array(laser_data.ranges)
        angle_min = laser_data.angle_min
        angle_increment = laser_data.angle_increment
        
        # 处理无效数据
        ranges = np.nan_to_num(ranges, nan=max_range)
        ranges = np.clip(ranges, 0, max_range)
        
        # 获取机器人网格坐标
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        for i, range_val in enumerate(ranges):
            laser_angle = angle_min + i * angle_increment + robot_theta
            
            # 计算激光终点
            if range_val < max_range - 0.1:  # 有效障碍物
                end_x = robot_x + range_val * math.cos(laser_angle)
                end_y = robot_y + range_val * math.sin(laser_angle)
                
                # 标记障碍物
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                self.mark_obstacle(end_grid_x, end_grid_y)
                
                # 标记从机器人到障碍物的射线为自由空间
                self.mark_free_line(robot_grid_x, robot_grid_y, end_grid_x, end_grid_y)
            else:
                # 超出最大范围，标记射线上的所有点
                end_x = robot_x + max_range * math.cos(laser_angle)
                end_y = robot_y + max_range * math.sin(laser_angle)
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                self.mark_free_line(robot_grid_x, robot_grid_y, end_grid_x, end_grid_y)
    
    def mark_obstacle(self, grid_x: int, grid_y: int, probability: float = 0.9):
        """标记障碍物"""
        if 0 <= grid_x < self.size and 0 <= grid_y < self.size:
            # 概率更新
            old_value = self.grid[grid_x, grid_y]
            self.grid[grid_x, grid_y] = min(1.0, old_value + probability * (1 - old_value))
            
            if self.grid[grid_x, grid_y] > 0.7:  # 高概率障碍物
                self.obstacle_cells.add((grid_x, grid_y))
                if (grid_x, grid_y) in self.free_cells:
                    self.free_cells.remove((grid_x, grid_y))
    
    def mark_free(self, grid_x: int, grid_y: int, probability: float = 0.7):
        """标记自由空间"""
        if 0 <= grid_x < self.size and 0 <= grid_y < self.size:
            old_value = self.grid[grid_x, grid_y]
            self.grid[grid_x, grid_y] = max(0.0, old_value - probability * old_value)
            
            if self.grid[grid_x, grid_y] < 0.3:  # 高概率自由空间
                self.free_cells.add((grid_x, grid_y))
                if (grid_x, grid_y) in self.obstacle_cells:
                    self.obstacle_cells.remove((grid_x, grid_y))
    
    def mark_free_line(self, x0: int, y0: int, x1: int, y1: int):
        """使用Bresenham算法标记直线上的自由空间"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        steps = 0
        
        while True:
            # 标记当前点为自由空间（不包括终点）
            if not (x == x1 and y == y1):
                self.mark_free(x, y)
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            
            steps += 1
            if steps > 1000:  # 防止无限循环
                break
    
    def is_occupied(self, grid_x: int, grid_y: int, threshold: float = 0.5) -> bool:
        """检查网格是否被占据"""
        if not (0 <= grid_x < self.size and 0 <= grid_y < self.size):
            return True  # 边界外视为障碍物
        return self.grid[grid_x, grid_y] > threshold
    
    def is_free(self, grid_x: int, grid_y: int, threshold: float = 0.3) -> bool:
        """检查网格是否为自由空间"""
        if not (0 <= grid_x < self.size and 0 <= grid_y < self.size):
            return False  # 边界外不为自由空间
        return self.grid[grid_x, grid_y] < threshold
    
    def get_obstacle_distance(self, grid_x: int, grid_y: int, radius: int = 5) -> float:
        """获取到最近障碍物的距离（网格单位）"""
        min_distance = float('inf')
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.is_occupied(nx, ny):
                        distance = math.sqrt(dx*dx + dy*dy)
                        min_distance = min(min_distance, distance)
        
        return min_distance if min_distance < float('inf') else radius
    
    def save_map(self, filename: str):
        """保存地图到文件"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        map_data = {
            'grid': self.grid,
            'resolution': self.resolution,
            'size': self.size,
            'origin': self.origin,
            'obstacle_cells': list(self.obstacle_cells),
            'free_cells': list(self.free_cells)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)
        
        print(f"Map saved to {filename}")
        print(f"Obstacle cells: {len(self.obstacle_cells)}, Free cells: {len(self.free_cells)}")
    
    def load_map(self, filename: str) -> bool:
        """从文件加载地图"""
        try:
            with open(filename, 'rb') as f:
                map_data = pickle.load(f)
            
            self.grid = map_data['grid']
            self.resolution = map_data['resolution']
            self.size = map_data['size']
            self.origin = map_data['origin']
            self.obstacle_cells = set(map_data['obstacle_cells'])
            self.free_cells = set(map_data['free_cells'])
            
            print(f"Map loaded from {filename}")
            print(f"Obstacle cells: {len(self.obstacle_cells)}, Free cells: {len(self.free_cells)}")
            return True
        except Exception as e:
            print(f"Failed to load map: {e}")
            return False
    
    def reset(self):
        """重置地图"""
        self.grid.fill(0.0)
        self.obstacle_cells.clear()
        self.free_cells.clear()
    
    def get_map_for_display(self) -> np.ndarray:
        """获取用于显示的地图（0-255灰度图）"""
        display_map = (self.grid * 255).astype(np.uint8)
        return display_map


# ========================== 增强A*路径规划器 ==========================
class EnhancedAStarPlanner:
    """增强的A*路径规划器，支持路径记忆"""
    def __init__(self, occupancy_map: OccupancyGridMap, path_memory: PathMemory = None):
        self.map = occupancy_map
        self.path_memory = path_memory
        self.inflation_radius = 3  # 障碍物膨胀半径（网格单位）
        
        # 方向定义（8方向）
        self.directions = [
            (0, 1, 1.0),    # 上
            (1, 0, 1.0),    # 右
            (0, -1, 1.0),   # 下
            (-1, 0, 1.0),   # 左
            (1, 1, 1.414),  # 右上
            (1, -1, 1.414), # 右下
            (-1, 1, 1.414), # 左上
            (-1, -1, 1.414) # 左下
        ]
        
        # 路径平滑参数
        self.smooth_path = True
        self.max_smooth_iterations = 100
        
        # 路径重用统计
        self.path_reuse_count = 0
        self.path_plan_count = 0
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return self.map.world_to_grid(x, y)
    
    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        return self.map.grid_to_world(i, j)
    
    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """检查单元格是否有效（非障碍物且有一定安全距离）"""
        if not (0 <= grid_x < self.map.size and 0 <= grid_y < self.map.size):
            return False
        
        # 检查是否为障碍物
        if self.map.is_occupied(grid_x, grid_y):
            return False
        
        # 检查安全距离（膨胀半径）
        obstacle_distance = self.map.get_obstacle_distance(grid_x, grid_y, self.inflation_radius)
        if obstacle_distance < 2:  # 距离障碍物太近
            return False
        
        return True
    
    def get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """获取有效邻居"""
        i, j = cell
        neighbors = []
        
        for di, dj, cost in self.directions:
            ni, nj = i + di, j + dj
            
            if not self.is_valid_cell(ni, nj):
                continue
            
            # 检查对角线是否被障碍物阻挡
            if abs(di) == 1 and abs(dj) == 1:
                if not self.is_valid_cell(i + di, j) or not self.is_valid_cell(i, j + dj):
                    continue
            
            # 添加成本启发式：更偏好远离障碍物的路径
            obstacle_distance = self.map.get_obstacle_distance(ni, nj, self.inflation_radius)
            safety_bonus = max(0, 5 - obstacle_distance) * 0.1  # 安全奖励
            adjusted_cost = cost + safety_bonus
            
            neighbors.append(((ni, nj), adjusted_cost))
        
        return neighbors
    
    @staticmethod
    def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    @staticmethod
    def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def plan_with_memory(self, start_world: Tuple[float, float], 
                        goal_world: Tuple[float, float]) -> List[Tuple[float, float]]:
        """使用记忆的路径规划"""
        self.path_plan_count += 1
        
        # 尝试从记忆获取路径
        if self.path_memory and random.random() < Config.reuse_memory_chance:
            memory_path = self.path_memory.get_best_path(start_world, goal_world, self.map)
            if memory_path:
                self.path_reuse_count += 1
                reuse_rate = self.path_reuse_count / self.path_plan_count * 100
                print(f"Using memorized path (reuse rate: {reuse_rate:.1f}%)")
                return memory_path
        
        # 使用A*规划新路径
        return self.plan(start_world, goal_world)
    
    def plan(self, start_world: Tuple[float, float], goal_world: Tuple[float, float]) -> List[Tuple[float, float]]:
        """主路径规划函数"""
        start_cell = self.world_to_grid(*start_world)
        goal_cell = self.world_to_grid(*goal_world)
        
        print(f"Planning from {start_world} (grid: {start_cell}) to {goal_world} (grid: {goal_cell})")
        
        # 检查起始和目标点
        if not self.is_valid_cell(*start_cell):
            print(f"警告：起始位置 {start_world} 在障碍物附近")
            # 尝试寻找附近的可行点
            start_cell = self.find_nearby_valid_cell(*start_cell)
        
        if not self.is_valid_cell(*goal_cell):
            print(f"警告：目标位置 {goal_world} 在障碍物附近")
            # 尝试寻找附近的可行点
            goal_cell = self.find_nearby_valid_cell(*goal_cell)
        
        # A*算法
        open_set = []
        heapq.heappush(open_set, (0, 0, start_cell))
        
        came_from = {}
        g_score = {start_cell: 0}
        h_score = self.euclidean_distance(start_cell, goal_cell)
        f_score = {start_cell: h_score}
        
        closed_set = set()
        iteration = 0
        
        while open_set and iteration < 10000:
            current_f, current_g, current = heapq.heappop(open_set)
            iteration += 1
            
            if current in closed_set:
                continue
            
            if current == goal_cell:
                # 重建路径
                path = self.reconstruct_path(came_from, current, start_cell)
                
                if self.smooth_path:
                    path = self.smooth_path_points(path)
                
                # 转换为世界坐标
                world_path = [self.grid_to_world(i, j) for i, j in path]
                print(f"路径规划成功！找到 {len(world_path)} 个航点")
                return world_path
            
            closed_set.add(current)
            
            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_g + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self.euclidean_distance(neighbor, goal_cell)
                    f_score[neighbor] = tentative_g + h_score
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        print(f"未找到从 {start_world} 到 {goal_world} 的路径")
        return []
    
    def smooth_path_points(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """路径平滑"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = smoothed[-1]
            current = path[i]
            next_point = path[i + 1]
            
            # 检查是否可以跳过中间点
            if self.line_of_sight(prev[0], prev[1], next_point[0], next_point[1]):
                continue  # 跳过当前点
            
            smoothed.append(current)
        
        smoothed.append(path[-1])
        return smoothed
    
    def line_of_sight(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        """检查两点之间是否有直线可见性"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # 检查当前点是否有效
            if not self.is_valid_cell(x, y):
                return False
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def find_nearby_valid_cell(self, grid_x: int, grid_y: int, max_radius: int = 10) -> Tuple[int, int]:
        """寻找附近的有效单元格"""
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if self.is_valid_cell(nx, ny):
                        return (nx, ny)
        return (grid_x, grid_y)  # 返回原始点作为最后手段
    
    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int], 
                         start: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重建路径"""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path


# ========================== 辅助函数 ==========================
def compute_heading_error(robot_x: float, robot_y: float, robot_heading: float, 
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


# ========================== DQN网络 ==========================
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


# ========================== DQN训练器 ==========================
class SpeedOptimizedDQNTrainer:
    """三倍速度优化的DQN训练器"""
    def __init__(self, state_dim: int, action_size: int, device: torch.device, 
                 lr: float = 1e-4, gamma: float = 0.95):
        self.device = device
        self.gamma = gamma
        self.action_size = action_size
        
        self.model = SimpleDQN(state_dim, action_size).to(device)
        self.target_model = SimpleDQN(state_dim, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.memory = deque(maxlen=20000)  # 增大经验回放缓冲区
        self.batch_size = Config.batch_size
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # 降低最小探索率
        self.epsilon_decay = 0.997  # 减慢探索衰减
        
        # 三倍速度动作映射
        self.action_map = {
            0: (Config.max_linear_speed * 0.8, 0.0),      # 高速前进
            1: (Config.max_linear_speed * 0.7, Config.max_angular_speed * 0.3),   # 前进+轻微左转
            2: (Config.max_linear_speed * 0.7, -Config.max_angular_speed * 0.3),  # 前进+轻微右转
            3: (Config.max_linear_speed * 0.5, Config.max_angular_speed * 0.6),   # 中速前进+左转
            4: (Config.max_linear_speed * 0.5, -Config.max_angular_speed * 0.6),  # 中速前进+右转
            5: (0.0, Config.max_angular_speed * 0.8),     # 原地左转
            6: (0.0, -Config.max_angular_speed * 0.8),    # 原地右转
            7: (-Config.max_linear_speed * 0.3, 0.0),     # 后退
        }
        
        self.learn_step = 0
        self.update_frequency = 2
        
        self.training_losses = []
        self.episode_rewards = []
        self.episode_q_values = []  # 记录每个episode的平均Q值
        self.episode_losses = []
    
    def act(self, state: np.ndarray, heading_error: float = 0.0, 
            min_laser: float = 10.0, distance_to_goal: float = 10.0) -> Tuple[List[float], int]:
        
        # 紧急避障处理
        if min_laser < 0.25:
            # 非常近距离障碍物：紧急停止并转向
            return [0.0, Config.max_angular_speed * 0.8], 5 if heading_error > 0 else 6
        
        # 探索/利用决策
        if random.random() < self.epsilon:
            # 随机探索，根据情况调整权重
            if min_laser < 0.6:
                weights = [0.1, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.0]  # 更多转向
            elif distance_to_goal > 3.0:
                weights = [0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.0]  # 更多前进
            else:
                weights = [0.3, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.0]  # 平衡
            action_idx = random.choices(range(self.action_size), weights=weights)[0]
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy()[0]
            
            # 记录Q值用于分析
            if hasattr(self, 'current_episode_q_values'):
                self.current_episode_q_values.append(np.max(q_values))
            
            # 启发式引导
            heuristic = np.zeros(self.action_size)
            
            # 基于航向误差的启发式
            if abs(heading_error) < 0.2:
                heuristic[0] = 1.0  # 直行
                heuristic[1] = 0.3
                heuristic[2] = 0.3
            elif heading_error > 0.5:
                heuristic[3] = 1.0  # 左转
                heuristic[5] = 0.5  # 原地左转
            elif heading_error > 0:
                heuristic[1] = 1.0  # 轻微左转
                heuristic[3] = 0.5
            elif heading_error < -0.5:
                heuristic[4] = 1.0  # 右转
                heuristic[6] = 0.5  # 原地右转
            else:
                heuristic[2] = 1.0  # 轻微右转
                heuristic[4] = 0.5
            
            # 基于障碍物的启发式
            if min_laser < 1.0:
                heuristic[0] = -0.5  # 避免直行
                if heading_error > 0:
                    heuristic[2] += 0.3  # 偏向右侧
                    heuristic[4] += 0.2
                else:
                    heuristic[1] += 0.3  # 偏向左侧
                    heuristic[3] += 0.2
            
            # 基于距离的启发式
            if distance_to_goal < 1.0:
                heuristic[0] *= 0.5  # 接近目标时减速
            
            # 组合分数
            combined_scores = q_values + Config.heuristic_weight * heuristic
            action_idx = np.argmax(combined_scores)
        
        linear, angular = self.action_map[action_idx]
        return [linear, angular], action_idx
    
    def remember(self, state: np.ndarray, action_idx: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
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
            'action_map': self.action_map,
        }, filename)
        print(f"DQN saved to {filename}")
    
    def load(self, filename: str):
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.action_map = checkpoint.get('action_map', self.action_map)
            print(f"DQN loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load DQN: {e}")
            return False
    
    def record_episode_reward(self, reward: float):
        self.episode_rewards.append(reward)
    
    def start_episode(self):
        """开始新回合的记录"""
        self.current_episode_q_values = []
        self.current_episode_losses = []
    
    def end_episode(self):
        """结束回合记录"""
        if hasattr(self, 'current_episode_q_values') and self.current_episode_q_values:
            avg_q = np.mean(self.current_episode_q_values)
            self.episode_q_values.append(avg_q)
        
        if hasattr(self, 'current_episode_losses') and self.current_episode_losses:
            avg_loss = np.mean(self.current_episode_losses)
            self.episode_losses.append(avg_loss)
    
    def get_episode_metrics(self):
        """获取当前回合的指标"""
        avg_q = np.mean(self.current_episode_q_values) if hasattr(self, 'current_episode_q_values') and self.current_episode_q_values else 0.0
        avg_loss = np.mean(self.training_losses[-10:]) if len(self.training_losses) > 0 else 0.0
        return avg_q, avg_loss


# ========================== 主训练器类 ==========================
class MapBasedAStarDQNTrainer:
    """基于地图的A* DQN训练管理器"""
    def __init__(self):
        print("Initializing Map-based A-Star DQN training with 3x speed and path memory...")
        
        self.env = None
        self.max_retries = 10
        self.retry_count = 0
        
        self.state_dim = 10 + 3 + 2  # 激光(10) + 位姿(3) + 目标(2)
        self.action_size = 8  # 增加动作空间
        
        # 初始化监视器
        self.odom_monitor = OdometryMonitor(Config.odom_topic)
        self.laser_monitor = LaserMonitor(Config.laser_topic)
        
        # 初始化发布器
        self.cmd_vel_pub = None
        
        # 初始化Gazebo控制器和RViz可视化
        self.gazebo_controller = GazeboController()
        
        # 初始化占据栅格地图和路径规划器
        self.occupancy_map = OccupancyGridMap(
            resolution=Config.map_resolution,
            size=Config.map_size,
            origin=(Config.map_origin_x, Config.map_origin_y)
        )
        
        # 初始化路径记忆系统
        self.path_memory = PathMemory(Config.path_memory_path)
        
        # 初始化困境学习器
        self.escape_learner = EscapeLearner()
        
        # 初始化RViz可视化（带路径记忆）
        self.rviz_visualizer = RVizVisualizer(self.path_memory)
        
        # 初始化路径规划器（带记忆）
        self.planner = EnhancedAStarPlanner(self.occupancy_map, self.path_memory)
        
        self.trainer = SpeedOptimizedDQNTrainer(self.state_dim, self.action_size, Config.device)
        
        # 训练状态
        self.timestep = 0
        self.episode_count = 0
        self.path = []
        self.path_index = 0
        
        # 统计信息
        self.success_count = 0
        self.collision_count = 0
        self.episode_collisions = 0  # 当前回合碰撞次数
        self.episode_start_time = 0
        
        # 当前目标信息
        self.current_goal = None
        self.current_goal_type = None
        
        # 路径跟踪参数
        self.stuck_counter = 0
        self.last_position = (0.0, 0.0)
        self.position_history = deque(maxlen=5)
        
        # 地图管理
        self.map_update_counter = 0
        self.map_save_counter = 0
        self.global_map_loaded = False
        
        # 训练监控器
        self.training_monitor = TrainingMonitor(Config.plot_save_path)
        
        # 安全起始位置
        self.safe_start_positions = [
            (3.0, 0.0),
            (3.0, 3.0),
            (-7.0, 4.0),
            (3.0, 6.0),
            (-5.0, 7.0)
        ]
        
        # 调试计数器
        self.debug_counter = 0
        self.no_progress_counter = 0
        
        # 尝试加载现有地图
        self._load_or_create_map()
    
    def _load_or_create_map(self):
        """加载现有地图或创建新地图"""
        map_path = os.path.join(Config.map_save_path, "global_map.pkl")
        
        if os.path.exists(map_path):
            if self.occupancy_map.load_map(map_path):
                self.global_map_loaded = True
                print("Global map loaded successfully")
            else:
                print("Failed to load global map, creating new one")
                self.occupancy_map.reset()
        else:
            print("No existing map found, creating new one")
            self.occupancy_map.reset()
            os.makedirs(Config.map_save_path, exist_ok=True)
    
    def _save_current_map(self):
        """保存当前地图"""
        map_path = os.path.join(Config.map_save_path, "global_map.pkl")
        self.occupancy_map.save_map(map_path)
    
    def initialize_environment(self):
        """初始化环境"""
        try:
            print("Initializing environment with 3x speed...")
            
            if not rospy.core.is_initialized():
                rospy.init_node('map_based_astar_dqn_trainer', anonymous=True)
            
            self.cmd_vel_pub = rospy.Publisher(Config.cmd_vel_topic, Twist, queue_size=10)
            time.sleep(2)
            
            # 清除RViz标记
            self.rviz_visualizer.clear_all_markers()
            
            if not self.odom_monitor.start_monitoring():
                return False
            if not self.laser_monitor.start_monitoring():
                return False
            
            # 等待传感器数据
            if not self.odom_monitor.wait_for_odometry(30.0):
                return False
                    
            if not self.laser_monitor.wait_for_laser(30.0):
                return False
            
            # 等待机器人稳定
            time.sleep(2)
            
            # 发布停止指令
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            time.sleep(1)
            
            class MapBasedEnv:
                def __init__(self, outer):
                    self.outer = outer
                    self.goal_x = 0.0
                    self.goal_y = 0.0
                    self.goal_type = None
                
                def reset(self):
                    """重置环境"""
                    print("Resetting environment for new episode...")
                    
                    # 停止机器人
                    twist = Twist()
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.outer.cmd_vel_pub.publish(twist)
                    time.sleep(0.3)  # 减少等待时间
                    
                    # 重置内部状态
                    self.outer.path = []
                    self.outer.path_index = 0
                    self.outer.stuck_counter = 0
                    self.outer.position_history.clear()
                    self.outer.no_progress_counter = 0
                    self.outer.episode_collisions = 0  # 重置当前回合碰撞计数
                    
                    # 获取当前位置
                    x, y, _ = self.outer.get_robot_pose()
                    self.outer.last_position = (x, y)
                    self.outer.position_history.extend([(x, y)] * 5)
                    
                    # 生成新目标
                    self._generate_navigation_goal()
                    
                    # 更新RViz可视化
                    self.outer.update_rviz_visualization()
                    
                    print(f"Starting at ({x:.2f}, {y:.2f})")
                    print(f"Target: {self.goal_type} at ({self.goal_x:.2f}, {self.goal_y:.2f})")
                    
                    return self.outer.get_state_from_sensors()
                
                def _generate_navigation_goal(self):
                    """生成导航目标"""
                    robot_x, robot_y, _ = self.outer.get_robot_pose()
                    
                    # 根据地图信息选择目标
                    if random.random() < 0.8:
                        # 选择货架目标
                        self.goal_index = random.randint(0, len(Config.shelf_locations) - 1)
                        self.goal_x, self.goal_y = Config.shelf_locations[self.goal_index]
                        self.goal_type = "shelf"
                    else:
                        # 选择充电站目标
                        self.goal_x, self.goal_y = Config.charging_station
                        self.goal_type = "charging"
                    
                    # 检查目标是否可达
                    grid_x, grid_y = self.outer.occupancy_map.world_to_grid(self.goal_x, self.goal_y)
                    if self.outer.occupancy_map.is_occupied(grid_x, grid_y):
                        print(f"警告：目标 {self.goal_x:.1f}, {self.goal_y:.1f} 在障碍物上，重新选择")
                        # 重新选择目标
                        if self.goal_type == "shelf":
                            self.goal_index = (self.goal_index + 1) % len(Config.shelf_locations)
                            self.goal_x, self.goal_y = Config.shelf_locations[self.goal_index]
                        else:
                            # 如果充电站不可达，选择一个货架
                            self.goal_index = random.randint(0, len(Config.shelf_locations) - 1)
                            self.goal_x, self.goal_y = Config.shelf_locations[self.goal_index]
                            self.goal_type = "shelf"
                    
                    # 更新当前目标信息
                    self.outer.current_goal = (self.goal_x, self.goal_y)
                    self.outer.current_goal_type = self.goal_type
            
            self.env = MapBasedEnv(self)
            print("Environment initialized successfully with 3x speed!")
            return True
                
        except Exception as e:
            print(f"Environment initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_rviz_visualization(self):
        """更新RViz可视化"""
        try:
            self.rviz_visualizer.visualize_goals(
                Config.shelf_locations, 
                Config.charging_station, 
                self.current_goal,
                self.current_goal_type
            )
            
            if self.path:
                self.rviz_visualizer.visualize_path(self.path)
            
            self.rviz_visualizer.visualize_training_status(
                self.episode_count,
                self.success_count,
                self.collision_count,
                self.current_goal_type
            )
            
            # 定期可视化地图和记忆路径
            if self.map_update_counter % 20 == 0:
                self.rviz_visualizer.visualize_map(self.occupancy_map)
                self.rviz_visualizer.visualize_memory_paths()
            
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
            ranges = np.array(laser_data.ranges)
            ranges = np.nan_to_num(ranges, nan=laser_data.range_max)
            if np.any(ranges < 0.25):
                self.episode_collisions += 1  # 记录当前回合碰撞
                return True
        return False
    
    def get_min_laser_distance(self) -> float:
        """获取最小激光距离"""
        laser_data = self.laser_monitor.get_laser_data()
        if laser_data is not None:
            ranges = np.array(laser_data.ranges)
            ranges = np.nan_to_num(ranges, nan=laser_data.range_max)
            return np.min(ranges) if len(ranges) > 0 else 10.0
        return 10.0
    
    def update_map_from_sensors(self):
        """从传感器更新地图"""
        robot_x, robot_y, robot_theta = self.get_robot_pose()
        laser_data = self.laser_monitor.get_laser_data()
        
        if laser_data is not None:
            self.occupancy_map.update_from_laser(
                robot_x, robot_y, robot_theta, laser_data, Config.laser_max_range
            )
            self.map_update_counter += 1
            
            # 定期保存地图
            if self.map_update_counter % 100 == 0:
                print(f"Map updated {self.map_update_counter} times")
    
    def get_next_waypoint(self) -> Tuple[float, float]:
        """获取下一个航点 - 基于地图和记忆的路径规划"""
        robot_x, robot_y, _ = self.get_robot_pose()
        goal_x, goal_y = self.env.goal_x, self.env.goal_y
        
        # 计算到目标的距离
        distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
        
        # 如果非常接近目标，直接返回目标点
        if distance_to_goal < 0.5:
            return goal_x, goal_y
        
        # 检查是否需要重新规划路径
        need_replan = False
        
        if not self.path or self.path_index >= len(self.path):
            need_replan = True
        elif self.path_index < len(self.path):
            # 检查当前航点是否可达
            waypoint_x, waypoint_y = self.path[self.path_index]
            grid_x, grid_y = self.occupancy_map.world_to_grid(waypoint_x, waypoint_y)
            if self.occupancy_map.is_occupied(grid_x, grid_y):
                print(f"当前航点 ({waypoint_x:.1f}, {waypoint_y:.1f}) 不可达，重新规划")
                need_replan = True
        
        # 如果需要重新规划或路径为空
        if need_replan:
            print("Planning new path using memory system...")
            
            # 使用带记忆的路径规划
            self.path = self.planner.plan_with_memory((robot_x, robot_y), (goal_x, goal_y))
            
            # 如果没有找到路径，尝试传统方法
            if not self.path:
                print("Memory planning failed, trying traditional A*...")
                self.path = self.planner.plan((robot_x, robot_y), (goal_x, goal_y))
            
            self.path_index = 0
            
            if self.path:
                print(f"New path planned with {len(self.path)} waypoints")
                self.update_rviz_visualization()
            else:
                print("Path planning failed, using direct approach")
                self.path = [(goal_x, goal_y)]
                self.path_index = 0
        
        # 路径跟踪
        if self.path_index < len(self.path):
            waypoint_x, waypoint_y = self.path[self.path_index]
            
            # 计算到当前航点的距离
            distance_to_waypoint = math.hypot(robot_x - waypoint_x, robot_y - waypoint_y)
            
            # 三倍速度下，增加切换阈值
            if distance_to_waypoint < 0.8 and self.path_index < len(self.path) - 1:
                self.path_index += 1
                waypoint_x, waypoint_y = self.path[self.path_index]
                print(f"Moving to next waypoint {self.path_index}/{len(self.path)}")
            
            return waypoint_x, waypoint_y
        
        return goal_x, goal_y
    
    def check_if_stuck(self, current_x: float, current_y: float) -> bool:
        """检查机器人是否卡住"""
        self.position_history.append((current_x, current_y))
        
        if len(self.position_history) < 3:
            return False
        
        # 计算平均移动距离
        total_movement = 0
        for i in range(1, len(self.position_history)):
            prev_x, prev_y = self.position_history[i-1]
            curr_x, curr_y = self.position_history[i]
            total_movement += math.hypot(curr_x - prev_x, curr_y - prev_y)
        
        avg_movement = total_movement / (len(self.position_history) - 1)
        
        # 如果移动距离很小，增加卡住计数器
        if avg_movement < 0.02:
            self.stuck_counter += 1
            if self.stuck_counter > 5:
                print(f"Robot stuck! Avg movement: {avg_movement:.4f}")
                return True
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        return False
    
    def get_state_from_sensors(self) -> np.ndarray:
        """从传感器数据构建状态向量"""
        laser_data = self.laser_monitor.get_laser_data()
        if laser_data is None:
            laser_state = np.zeros(10)
        else:
            laser_ranges = np.array(laser_data.ranges)
            laser_ranges = np.nan_to_num(laser_ranges, nan=laser_data.range_max)
            # 取10个均匀分布的激光点
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
    
    def _high_speed_action_adjustment(self, action: List[float], heading_error: float, 
                                      min_laser: float, distance_to_goal: float) -> List[float]:
        """高速动作调整"""
        linear, angular = action
        
        # 高速下的紧急避障
        if min_laser < 0.3:
            # 紧急停止并转向
            linear = -0.3  # 更快的后退
            angular = Config.max_angular_speed * 0.9
            return [linear, angular]
        
        # 障碍物处理
        if min_laser < 0.6:
            # 减速并转向
            linear = linear * 0.4  # 更大幅度的减速
            if heading_error > 0:
                angular = Config.max_angular_speed * 0.6
            else:
                angular = -Config.max_angular_speed * 0.6
        else:
            # 正常情况下的高速调整
            if abs(heading_error) > 1.0:
                # 大角度误差，主要转向
                linear = max(0.2, linear * 0.6)  # 保持一定前进速度
                angular = max(-Config.max_angular_speed * 0.8, 
                            min(Config.max_angular_speed * 0.8, heading_error * 1.5))
            elif abs(heading_error) > 0.3:
                # 中等角度误差
                linear = linear * 0.8
                angular = max(-Config.max_angular_speed * 0.6, 
                            min(Config.max_angular_speed * 0.6, heading_error * 1.8))
            else:
                # 小角度误差，高速前进
                linear = min(Config.max_linear_speed * 0.9, linear + 0.1)  # 加速更快
                angular = angular * 0.4  # 减少转向
        
        # 接近目标时减速
        if distance_to_goal < 1.5:
            linear = linear * 0.5
        
        # 限制速度范围
        linear = max(-0.5, min(Config.max_linear_speed, linear))
        angular = max(-Config.max_angular_speed, min(Config.max_angular_speed, angular))
        
        return [linear, angular]
    
    def _calculate_progress_reward(self, old_x, old_y, new_x, new_y,
                                  old_distance, new_distance,
                                  old_heading_error, new_heading_error,
                                  old_min_laser, new_min_laser):
        """基于进展的奖励计算"""
        done = False
        goal_reward = 0.0
        
        # 1. 成功到达目标
        if new_distance < 0.5:
            goal_reward = Config.success_reward
            done = True
            self.success_count += 1
            print("🎉 Goal reached!")
            return goal_reward, done
        
        # 2. 距离进展奖励（最重要）
        distance_improvement = old_distance - new_distance
        if distance_improvement > 0:
            # 正向进展
            progress_reward = distance_improvement * Config.progress_reward_weight
            self.no_progress_counter = 0
        else:
            # 负向进展
            progress_reward = distance_improvement * 1.5
            self.no_progress_counter += 1
        
        # 3. 航向奖励
        heading_improvement = abs(old_heading_error) - abs(new_heading_error)
        heading_reward = heading_improvement * 0.5
        
        # 4. 安全奖励
        safety_reward = 0.0
        if new_min_laser > 1.0:
            safety_reward = 0.01
        elif new_min_laser < 0.4:
            safety_reward = -0.2  # 增加不安全惩罚
        
        # 5. 步数惩罚
        step_penalty = Config.step_penalty
        
        # 6. 无进展惩罚
        no_progress_penalty = 0.0
        if self.no_progress_counter > 15:
            no_progress_penalty = -0.1
        
        total_reward = progress_reward + heading_reward + safety_reward + step_penalty + no_progress_penalty
        
        # 调试信息
        if self.debug_counter % 20 == 0:
            print(f"Rewards: progress={progress_reward:.3f}, heading={heading_reward:.3f}, "
                  f"safety={safety_reward:.3f}, no_progress={no_progress_penalty:.3f}")
            self.debug_counter = 0
        else:
            self.debug_counter += 1
        
        return total_reward, done
    
    def safe_step(self, action):
        """高速安全执行一步动作"""
        try:
            # 获取当前状态信息
            robot_x, robot_y, heading = self.get_robot_pose()
            waypoint_x, waypoint_y = self.get_next_waypoint()
            goal_x, goal_y = self.env.goal_x, self.env.goal_y
            min_laser = self.get_min_laser_distance()
            
            distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
            heading_error = compute_heading_error(robot_x, robot_y, heading, waypoint_x, waypoint_y)
            
            # 高速动作调整
            adjusted_action = self._high_speed_action_adjustment(
                action, heading_error, min_laser, distance_to_goal
            )
            
            # 执行调整后的动作
            twist = Twist()
            twist.linear.x = adjusted_action[0]
            twist.angular.z = adjusted_action[1]
            self.cmd_vel_pub.publish(twist)
            
            # 减少等待时间以适应高速
            time.sleep(0.1)  # 从0.15减少到0.1
            
            # 更新地图
            self.update_map_from_sensors()
            
            # 获取新状态
            next_state = self.get_state_from_sensors()
            new_robot_x, new_robot_y, new_heading = self.get_robot_pose()
            
            # 检查碰撞
            collision_detected = self.check_collision()
            if collision_detected:
                self.collision_count += 1
                print("Collision detected at high speed!")
                new_distance_to_goal = math.hypot(new_robot_x - goal_x, new_robot_y - goal_y)
                return next_state, Config.collision_penalty, True, {
                    "collision": True, 
                    "distance_to_goal": new_distance_to_goal,
                    "heading_error": 0.0,
                    "min_laser": 0.0
                }
            
            # 更新状态信息
            new_distance_to_goal = math.hypot(new_robot_x - goal_x, new_robot_y - goal_y)
            new_waypoint_x, new_waypoint_y = self.get_next_waypoint()
            new_heading_error = compute_heading_error(new_robot_x, new_robot_y, new_heading, new_waypoint_x, new_waypoint_y)
            new_min_laser = self.get_min_laser_distance()
            
            # 计算奖励
            reward, done = self._calculate_progress_reward(
                robot_x, robot_y, new_robot_x, new_robot_y,
                distance_to_goal, new_distance_to_goal,
                heading_error, new_heading_error,
                min_laser, new_min_laser
            )
            
            return next_state, reward, done, {
                "distance_to_goal": new_distance_to_goal,
                "heading_error": new_heading_error,
                "min_laser": new_min_laser,
                "progress": distance_to_goal - new_distance_to_goal
            }
            
        except Exception as e:
            print(f"Error in environment step: {e}")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            
            robot_x, robot_y, _ = self.get_robot_pose()
            goal_x, goal_y = self.env.goal_x, self.env.goal_y
            distance_to_goal = math.hypot(robot_x - goal_x, robot_y - goal_y)
            
            return self.get_state_from_sensors(), -2, True, {
                "error": str(e),
                "distance_to_goal": distance_to_goal
            }
    
    def _action_to_index(self, action: List[float]) -> int:
        """将动作转换为索引"""
        linear, angular = action
        
        # 根据动作映射查找最接近的索引
        min_distance = float('inf')
        best_index = 0
        
        for idx, (l, a) in self.trainer.action_map.items():
            distance = abs(linear - l) + abs(angular - a) * 0.5
            if distance < min_distance:
                min_distance = distance
                best_index = idx
        
        return best_index
    
    def run_training_episode(self):
        """运行训练回合"""
        if self.env is None and not self.initialize_environment():
            return -10
        
        try:
            print(f"\n=== Starting Episode {self.episode_count + 1} (3x Speed with Path Memory) ===")
            state = self.env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            episode_success = False
            
            # 重置回合状态
            self.path = []
            self.path_index = 0
            self.episode_start_time = time.time()
            self.stuck_counter = 0
            self.position_history.clear()
            self.debug_counter = 0
            self.no_progress_counter = 0
            self.episode_collisions = 0
            
            # 重置困境学习器
            self.escape_learner.reset_stuck_detection()
            
            # 记录起始位置（用于路径记忆）
            start_x, start_y, _ = self.get_robot_pose()
            start_position = (start_x, start_y)
            
            # 开始记录回合指标
            self.trainer.start_episode()
            
            # 记录初始位置
            start_x, start_y, _ = self.get_robot_pose()
            self.position_history.extend([(start_x, start_y)] * 3)
            
            # 记录初始路径
            initial_path = []
            
            while not done and step_count < Config.max_episode_steps:
                robot_x, robot_y, heading = self.get_robot_pose()
                waypoint_x, waypoint_y = self.get_next_waypoint()
                min_laser = self.get_min_laser_distance()
                distance_to_goal = math.hypot(robot_x - self.env.goal_x, robot_y - self.env.goal_y)
                
                heading_error = compute_heading_error(robot_x, robot_y, heading, waypoint_x, waypoint_y)
                
                # 更新困境学习器的状态历史
                self.escape_learner.update_state_history(robot_x, robot_y, 
                                                        self.laser_monitor.get_laser_data())
                
                # 检测是否陷入困境
                progress = self.last_progress - distance_to_goal if hasattr(self, 'last_progress') else 0
                time_elapsed = time.time() - self.episode_start_time
                
                is_stuck, stuck_info = self.escape_learner.detect_stuck(
                    robot_x, robot_y, heading, min_laser, progress, time_elapsed
                )
                
                # 如果陷入困境，使用学习器生成逃脱方案
                if is_stuck:
                    print(f"Robot stuck! Type: {stuck_info['type']}, Severity: {stuck_info['severity']:.2f}")
                    
                    # 获取最佳逃脱方案
                    current_state = (robot_x, robot_y, heading)
                    escape_sequence = self.escape_learner.get_best_escape(current_state, self.path_memory)
                    
                    if escape_sequence is None:
                        escape_sequence = self.escape_learner.generate_escape_sequence(stuck_info, current_state)
                    
                    # 记录初始状态
                    escape_start_state = current_state
                    escape_start_progress = distance_to_goal
                    
                    # 执行逃脱序列
                    for i, (linear, angular) in enumerate(escape_sequence):
                        if done or step_count >= Config.max_episode_steps:
                            break
                        
                        action = [linear, angular]
                        next_state, reward, done, info = self.safe_step(action)
                        
                        # 存储经验
                        action_idx = self._action_to_index(action)
                        self.trainer.remember(state, action_idx, reward, next_state, done)
                        
                        state = next_state
                        episode_reward += reward
                        step_count += 1
                        self.timestep += 1
                        
                        # 定期经验回放
                        if len(self.trainer.memory) > self.trainer.batch_size and step_count % 3 == 0:
                            loss = self.trainer.replay()
                            if step_count % 5 == 0:
                                self.trainer.soft_update()
                            
                            if hasattr(self.trainer, 'current_episode_losses'):
                                self.trainer.current_episode_losses.append(loss)
                    
                    # 学习逃脱效果
                    new_robot_x, new_robot_y, new_heading = self.get_robot_pose()
                    new_distance_to_goal = math.hypot(new_robot_x - self.env.goal_x, 
                                                     new_robot_y - self.env.goal_y)
                    progress_improvement = escape_start_progress - new_distance_to_goal
                    
                    # 评估逃脱是否成功
                    escape_success = progress_improvement > 0.1 or new_distance_to_goal < escape_start_progress
                    
                    # 学习这个逃脱方案
                    self.escape_learner.learn_from_escape(
                        escape_start_state, escape_sequence,
                        (new_robot_x, new_robot_y, new_heading),
                        escape_success, progress_improvement
                    )
                    
                    # 保存到路径记忆
                    self.path_memory.add_escape_solution(
                        escape_start_state, escape_sequence, escape_success
                    )
                    
                    # 重置困境检测
                    self.escape_learner.reset_stuck_detection()
                    continue
                
                # 正常导航决策
                action, action_idx = self.trainer.act(state, heading_error, min_laser, distance_to_goal)
                
                next_state, reward, done, info = self.safe_step(action)
                
                # 检查是否成功
                if reward == Config.success_reward:
                    episode_success = True
                    
                    # 保存成功路径到记忆
                    if initial_path and len(initial_path) > 1:
                        success_info = {
                            'episode': self.episode_count + 1,
                            'steps': step_count,
                            'reward': episode_reward,
                            'collisions': self.episode_collisions
                        }
                        self.path_memory.add_success_path(start_position, 
                                                        (self.env.goal_x, self.env.goal_y),
                                                        initial_path, success_info)
                
                # 存储经验
                self.trainer.remember(state, action_idx, reward, next_state, done)
                
                # 记录路径点（用于记忆）
                if not done and step_count % 5 == 0:
                    current_x, current_y, _ = self.get_robot_pose()
                    initial_path.append((current_x, current_y))
                
                # 经验回放
                if len(self.trainer.memory) > self.trainer.batch_size:
                    loss = self.trainer.replay()
                    if step_count % 5 == 0:  # 更频繁的软更新
                        self.trainer.soft_update()
                    
                    if hasattr(self.trainer, 'current_episode_losses'):
                        self.trainer.current_episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                step_count += 1
                self.timestep += 1
                
                # 记录上次进展
                self.last_progress = distance_to_goal
                
                # 定期输出状态
                if step_count % 10 == 0:
                    mem_stats = self.path_memory.get_statistics()
                    print(f"Step {step_count}, Reward: {reward:.2f}, "
                          f"Distance: {info['distance_to_goal']:.2f}, "
                          f"Memory: {mem_stats['total_paths']} paths, "
                          f"{mem_stats['escape_solutions']} escapes")
                
                # 卡住恢复
                if self.check_if_stuck(robot_x, robot_y):
                    print("Stuck recovery at high speed...")
                    twist = Twist()
                    twist.linear.x = -0.3
                    twist.angular.z = Config.max_angular_speed * 0.8
                    self.cmd_vel_pub.publish(twist)
                    time.sleep(0.2)
                    
                    # 重置状态
                    self.stuck_counter = 0
                    self.position_history.clear()
                    current_x, current_y, _ = self.get_robot_pose()
                    self.position_history.extend([(current_x, current_y)] * 3)
            
            # 结束回合记录
            self.trainer.end_episode()
            
            # 获取回合指标
            avg_q_value, avg_loss = self.trainer.get_episode_metrics()
            
            episode_time = time.time() - self.episode_start_time
            print(f"=== Episode {self.episode_count + 1} Completed ===")
            print(f"Reward: {episode_reward:.2f}, Steps: {step_count}, Time: {episode_time:.1f}s")
            print(f"Successes: {self.success_count}, Collisions: {self.collision_count}")
            print(f"Episode collisions: {self.episode_collisions}")
            print(f"Avg Q-value: {avg_q_value:.3f}, Avg Loss: {avg_loss:.3f}, Epsilon: {self.trainer.epsilon:.3f}")
            
            # 打印记忆统计
            mem_stats = self.path_memory.get_statistics()
            print(f"Path Memory: {mem_stats['total_paths']} paths, "
                  f"{mem_stats['escape_solutions']} escape solutions")
            print(f"Path reuse rate: {self.planner.path_reuse_count / max(1, self.planner.path_plan_count) * 100:.1f}%")
            
            # 记录训练数据
            self.training_monitor.record_episode(
                episode_idx=self.episode_count + 1,
                reward=episode_reward,
                steps=step_count,
                success=episode_success,
                collisions=self.episode_collisions,
                avg_q_value=avg_q_value,
                avg_loss=avg_loss,
                epsilon=self.trainer.epsilon
            )
            
            # 更新可视化
            self.training_monitor.generate_plots(self.episode_count + 1)
            
            # 定期保存地图和记忆
            self.map_save_counter += 1
            if self.map_save_counter % Config.map_save_frequency == 0:
                self._save_current_map()
                self.path_memory.save_memory()
                print(f"Map and memory saved at episode {self.episode_count + 1}")
            
            # 重置机器人位置
            if step_count >= Config.max_episode_steps or done:
                safe_x, safe_y = random.choice(self.safe_start_positions)
                theta = random.uniform(-math.pi, math.pi)
                self.gazebo_controller.reset_robot_pose(safe_x, safe_y, theta)
                time.sleep(0.3)
            
            return episode_reward
            
        except Exception as e:
            print(f"Error in training episode: {e}")
            import traceback
            traceback.print_exc()
            return -10
    
    def train(self):
        """主训练循环"""
        print("Starting Map-based DQN training with 3x speed and path memory...")
        set_seed(Config.seed)
        
        if Config.load_model:
            self.trainer.load("pytorch_models/map_based_dqn_latest.pth")
        
        best_reward = -float('inf')
        
        try:
            while self.timestep < Config.max_timesteps and self.retry_count < self.max_retries:
                episode_reward = self.run_training_episode()
                self.episode_count += 1
                
                # 定期保存模型和记忆
                if self.episode_count % 3 == 0:
                    self.trainer.save("pytorch_models/map_based_dqn_latest.pth")
                    self.path_memory.save_memory()
                    print(f"Model and memory saved at episode {self.episode_count}")
                
                if episode_reward > best_reward and episode_reward > 0:
                    best_reward = episode_reward
                    self.trainer.save("pytorch_models/map_based_dqn_best.pth")
                    print(f"New best model saved with reward: {best_reward:.2f}")
                
                if episode_reward <= -8:
                    self.retry_count += 1
                    print(f"Environment unstable, retry {self.retry_count}/{self.max_retries}")
                else:
                    self.retry_count = 0
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # 保存最终地图和记忆
            self._save_current_map()
            self.path_memory.save_memory()
            
            # 生成最终报告和图像
            self.training_monitor.generate_summary_report()
            
            # 打印最终记忆统计
            mem_stats = self.path_memory.get_statistics()
            print(f"\n=== Final Memory Statistics ===")
            print(f"Total paths memorized: {mem_stats['total_paths']}")
            print(f"Escape solutions: {mem_stats['escape_solutions']}")
            print(f"Memory size: {mem_stats['memory_size_mb']:.2f} MB")
            print(f"Path reuse rate: {self.planner.path_reuse_count / max(1, self.planner.path_plan_count) * 100:.1f}%")
            
            self.odom_monitor.stop_monitoring()
            self.laser_monitor.stop_monitoring()
            self.trainer.save("pytorch_models/map_based_dqn_final.pth")
            
            success_rate = (self.success_count / self.episode_count * 100) if self.episode_count > 0 else 0
            print(f"\n=== Training Summary ===")
            print(f"Total Episodes: {self.episode_count}")
            print(f"Success Rate: {success_rate:.1f}% ({self.success_count}/{self.episode_count})")
            print(f"Collisions: {self.collision_count}")
            print(f"Best Reward: {best_reward:.2f}")
            print(f"Map updated {self.map_update_counter} times")
            print(f"Training plots saved to: {Config.plot_save_path}")
            print(f"Path memory saved to: {Config.path_memory_path}")


# ========================== 主函数 ==========================
if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nInterrupt received, cleaning up...")
        cleanup_ros()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Waiting for ROS core...")
    time.sleep(3)
    
    try:
        trainer = MapBasedAStarDQNTrainer()
        trainer.train()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Training completed.")
