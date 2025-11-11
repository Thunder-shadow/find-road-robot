import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import heapq
import gym
from gym import spaces
import time
import pickle

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class WarehouseEnv(gym.Env):
    def __init__(self, interactive_mode=False):
        super(WarehouseEnv, self).__init__()
        self.interactive_mode = interactive_mode

        # 环境参数
        self.grid_size = 50
        self.robot_size = 2
        self.num_targets = 5
        self.num_obstacles = 15

        # 动作空间: 上,下,左,右,停留
        self.action_space = spaces.Discrete(5)

        # 观测空间
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(12 + self.num_targets,), dtype=np.float32)

        # 固定起始位置
        self.start_pos = np.array([5, 5])

        # 生成固定障碍物
        self._generate_fixed_obstacles()

        # 固定标靶位置
        self.target_positions = {
            'T1': (10, 10),
            'T2': (10, 40),
            'T3': (40, 10),
            'T4': (25, 25),
            'T5': (30, 30),
            'HOME': tuple(self.start_pos)
        }

        # 预计算所有路径
        self.path_cache = {}
        self._precompute_all_paths()

        # 初始化变量
        self.reset()

    def _generate_fixed_obstacles(self):
        """生成固定的障碍物布局"""
        self.obstacles = []
        random.seed(42)  # 固定随机种子确保可重复性
        for _ in range(self.num_obstacles):
            w, h = random.randint(2, 8), random.randint(2, 8)
            x = random.randint(0, self.grid_size - w)
            y = random.randint(0, self.grid_size - h)
            self.obstacles.append((x, y, w, h))
        random.seed()  # 重置随机种子

    def _check_collision(self, pos):
        """检查碰撞"""
        x, y = pos
        if x <= 0 or x >= self.grid_size or y <= 0 or y >= self.grid_size:
            return True
        for (ox, oy, w, h) in self.obstacles:
            if ox <= x <= ox + w and oy <= y <= oy + h:
                return True
        return False

    def _a_star_search(self, start, goal):
        """A*路径规划算法"""

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = [(fscore[start], start)]

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            close_set.add(current)

            for dx, dy in neighbors:
                neighbor = current[0] + dx, current[1] + dy

                if (neighbor[0] < 0 or neighbor[0] >= self.grid_size or
                        neighbor[1] < 0 or neighbor[1] >= self.grid_size or
                        self._check_collision(neighbor)):
                    continue

                tentative_g_score = gscore[current] + 1

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return []  # 如果没有找到路径返回空列表

    def _precompute_all_paths(self):
        """预计算所有关键点之间的路径"""
        print("Precomputing all possible paths...")
        start_time = time.time()

        # 获取所有关键点
        key_points = list(self.target_positions.items())

        # 计算所有点对之间的路径
        for i, (name1, pos1) in enumerate(key_points):
            for j, (name2, pos2) in enumerate(key_points[i + 1:], i + 1):
                path = self._a_star_search(pos1, pos2)
                reverse_path = path[::-1] if path else []

                # 存储路径
                self.path_cache[(name1, name2)] = path
                self.path_cache[(name2, name1)] = reverse_path

        print(f"Path precomputation completed in {time.time() - start_time:.2f} seconds")
        print(f"Total paths cached: {len(self.path_cache)}")

    def _get_state(self):
        """获取当前状态"""
        state = np.zeros(12 + self.num_targets, dtype=np.float32)

        # 机器人位置(归一化)
        state[:2] = self.robot_pos / self.grid_size

        # 四个方向的障碍物距离(归一化)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上,右,下,左
        for i, (dx, dy) in enumerate(directions):
            dist = self._get_obstacle_distance(dx, dy)
            state[2 + i] = dist / self.grid_size

        # 当前目标位置(归一化)
        if self.current_target:
            current_target_pos = self.target_positions[self.current_target]
            state[6:8] = np.array(current_target_pos) / self.grid_size

        # 启发式路径方向
        if len(self.heuristic_path) > 0 and self.path_index < len(self.heuristic_path):
            next_pos = self.heuristic_path[self.path_index]
            dx = next_pos[0] - self.robot_pos[0]
            dy = next_pos[1] - self.robot_pos[1]

            state[8 + self.num_targets] = 1.0 if dy > 0 and abs(dy) > abs(dx) else 0.0  # 上
            state[9 + self.num_targets] = 1.0 if dx > 0 and abs(dx) > abs(dy) else 0.0  # 右
            state[10 + self.num_targets] = 1.0 if dy < 0 and abs(dy) > abs(dx) else 0.0  # 下
            state[11 + self.num_targets] = 1.0 if dx < 0 and abs(dx) > abs(dy) else 0.0  # 左

        return state

    def _get_obstacle_distance(self, dx, dy):
        """计算障碍物距离"""
        x, y = self.robot_pos
        step = 0
        while 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            if self._check_collision((x, y)):
                return step
            x += dx
            y += dy
            step += 1
        return step

    def reset(self, training_mode=False):
        """重置环境"""
        self.robot_pos = np.array([5, 5])  # 固定起始位置
        self.current_target = None
        self.steps = 0
        self.total_reward = 0
        self.heuristic_path = []
        self.path_index = 0

        if training_mode:
            # 训练模式下随机选择目标
            self.current_target = random.choice(list(self.target_positions.keys())[:-1])  # 不包含HOME
            self._update_heuristic_path()

        return self._get_state()

    def _update_heuristic_path(self):
        """更新启发式路径"""
        if self.current_target:
            start = tuple(self.robot_pos.astype(int))
            goal = self.target_positions[self.current_target]

            # 尝试从缓存中获取路径
            cache_key = None
            for (name1, name2), path in self.path_cache.items():
                if (np.array_equal(self.target_positions[name1], start) and
                        np.array_equal(self.target_positions[name2], goal)):
                    cache_key = (name1, name2)
                    break
                elif (np.array_equal(self.target_positions[name2], start) and
                      np.array_equal(self.target_positions[name1], goal)):
                    cache_key = (name2, name1)
                    break

            if cache_key:
                self.heuristic_path = self.path_cache[cache_key]
            else:
                # 如果没有缓存，则实时计算
                self.heuristic_path = self._a_star_search(start, goal)

            self.path_index = 0

    def step(self, action):
        """执行一步动作"""
        assert self.action_space.contains(action), "Invalid action"

        old_pos = self.robot_pos.copy()
        if action == 0:
            self.robot_pos[1] = min(self.robot_pos[1] + 1, self.grid_size)  # 上
        elif action == 1:
            self.robot_pos[1] = max(self.robot_pos[1] - 1, 0)  # 下
        elif action == 2:
            self.robot_pos[0] = max(self.robot_pos[0] - 1, 0)  # 左
        elif action == 3:
            self.robot_pos[0] = min(self.robot_pos[0] + 1, self.grid_size)  # 右

        # 检查碰撞
        collision = self._check_collision(self.robot_pos)
        if collision:
            self.robot_pos = old_pos

        # 更新路径索引
        if len(self.heuristic_path) > 0:
            if (self.path_index < len(self.heuristic_path) and
                    tuple(self.robot_pos) == self.heuristic_path[self.path_index]):
                self.path_index += 1
            elif np.linalg.norm(self.robot_pos - np.array(
                    self.heuristic_path[min(self.path_index, len(self.heuristic_path) - 1)])) > 3:
                self._update_heuristic_path()

        # 计算奖励
        reward = -0.1  # 时间惩罚
        if collision:
            reward -= 5  # 碰撞惩罚

        # 检查是否到达目标
        done = False
        if self.current_target:
            target_pos = np.array(self.target_positions[self.current_target])
            distance = np.linalg.norm(self.robot_pos - target_pos)
            reward += (1 - distance / self.grid_size) * 0.5  # 距离奖励

            if distance < 2:  # 到达目标
                reward += 50
                done = True

        # 路径跟随奖励
        if len(self.heuristic_path) > 0 and self.path_index < len(self.heuristic_path):
            next_pos = self.heuristic_path[self.path_index]
            dx = next_pos[0] - self.robot_pos[0]
            dy = next_pos[1] - self.robot_pos[1]

            # 如果动作与路径方向一致，给予奖励
            if (action == 0 and dy > 0) or (action == 1 and dy < 0) or \
                    (action == 2 and dx < 0) or (action == 3 and dx > 0):
                reward += 1.0

        self.steps += 1
        self.total_reward += reward

        return self._get_state(), reward, done, {}

    def save_path_cache(self, filename):
        """保存路径缓存到文件"""
        with open(filename, 'wb') as f:
            pickle.dump(self.path_cache, f)
        print(f"Path cache saved to {filename}")

    def load_path_cache(self, filename):
        """从文件加载路径缓存"""
        try:
            with open(filename, 'rb') as f:
                self.path_cache = pickle.load(f)
            print(f"Path cache loaded from {filename}")
            return True
        except:
            print(f"Failed to load path cache from {filename}")
            return False


class HybridAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=100000)

        # 超参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005
        self.batch_size = 128
        self.tau = 0.005
        self.heuristic_weight = 1.5
        self.path_follow_bonus = 2.0

        # 网络
        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target_model = DuelingDQN(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()
        self.update_every = 4
        self.learn_step = 0

        # 训练统计
        self.episode_rewards = []
        self.episode_success = []
        self.training_stats = []

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        # 获取启发式方向 (状态向量的最后4个元素)
        heuristic_directions = state[-4:]

        # 创建启发式动作偏好
        heuristic_weights = np.zeros(self.action_size)
        if heuristic_directions[0] > 0.5:  # 上
            heuristic_weights[0] = self.path_follow_bonus
        if heuristic_directions[2] > 0.5:  # 下
            heuristic_weights[1] = self.path_follow_bonus
        if heuristic_directions[3] > 0.5:  # 左
            heuristic_weights[2] = self.path_follow_bonus
        if heuristic_directions[1] > 0.5:  # 右
            heuristic_weights[3] = self.path_follow_bonus

        # 结合DQN输出和启发式信息
        combined_q = q_values + self.heuristic_weight * heuristic_weights
        action = np.argmax(combined_q)

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size or self.learn_step % self.update_every != 0:
            self.learn_step += 1
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        loss = self.criterion(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # 更新目标网络
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.learn_step += 1

        # 记录训练统计
        self.training_stats.append({
            'step': self.learn_step,
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'avg_q': current_q.mean().item()
        })

    def save(self, filename):
        """保存模型和训练状态"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_success': self.episode_success,
            'training_stats': self.training_stats
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """加载模型和训练状态"""
        try:
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_success = checkpoint.get('episode_success', [])
            self.training_stats = checkpoint.get('training_stats', [])
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.action_size = action_size

        # 共享特征层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


def train_agent(episodes=2000, save_interval=100):
    """训练智能体"""
    env = WarehouseEnv()
    agent = HybridAgent(env.observation_space.shape[0], env.action_space.n)

    # 尝试加载预训练模型
    try:
        if agent.load('warehouse_hybrid_agent.pth'):
            print("Resuming training from saved model")
    except:
        print("Starting new training")

    # 训练统计
    best_reward = -np.inf
    start_time = time.time()

    for e in range(episodes):
        state = env.reset(training_mode=True)
        total_reward = 0
        done = False
        success = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

            if done and reward > 45:  # 到达目标的高奖励
                success = True

        # 记录训练统计
        agent.episode_rewards.append(total_reward)
        agent.episode_success.append(1 if success else 0)

        # 定期保存模型
        if (e + 1) % save_interval == 0:
            agent.save(f'warehouse_hybrid_agent_ep{e + 1}.pth')

            # 保存最佳模型
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save('warehouse_hybrid_agent_best.pth')

        # 打印训练进度
        avg_reward = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(
            agent.episode_rewards)
        avg_success = np.mean(agent.episode_success[-100:]) if len(agent.episode_success) >= 100 else np.mean(
            agent.episode_success)

        print(f"Episode: {e + 1}/{episodes}, Reward: {total_reward:.1f}, Avg: {avg_reward:.1f}, "
              f"Success: {success}, Success Rate: {avg_success:.2f}, Epsilon: {agent.epsilon:.3f}, "
              f"Time: {time.time() - start_time:.1f}s")

    # 保存最终模型
    agent.save('warehouse_hybrid_agent_final.pth')

    # 绘制训练曲线
    plot_training_results(agent)

    return agent


def plot_training_results(agent):
    """绘制训练结果图表"""
    plt.figure(figsize=(15, 5))

    # 奖励曲线
    plt.subplot(1, 3, 1)
    plt.plot(agent.episode_rewards)
    plt.plot(np.convolve(agent.episode_rewards, np.ones(100) / 100, mode='valid'))
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # 成功率曲线
    plt.subplot(1, 3, 2)
    plt.plot(np.convolve(agent.episode_success, np.ones(100) / 100, mode='valid'))
    plt.title('Success Rate (100-episode moving avg)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    # 训练损失曲线
    plt.subplot(1, 3, 3)
    if agent.training_stats:
        steps = [s['step'] for s in agent.training_stats]
        losses = [s['loss'] for s in agent.training_stats]
        plt.plot(steps, losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()


if __name__ == "__main__":
    # 训练智能体
    trained_agent = train_agent(episodes=2000)

    # 保存路径缓存
    env = WarehouseEnv()
    env.save_path_cache('warehouse_path_cache.pkl')