import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv
from velodyne_env import GazeboEnv, check_pos
import math


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode
file_name = "TD3_velodyne"  # name of the file to load the policy from


# 顶部：导入与开关
# ... existing code ...

# 切换到A*+DuelingDQN混合寻路模式（替换原策略动作）
USE_HYBRID = True

# 基于router_train.py思想的A*寻路（4邻接、曼哈顿启发式）
class AStarPlanner:
    def __init__(self, world_min=-4.5, world_max=4.5, grid_size=50, cache_file="gazebo_path_cache.pkl"):
        self.world_min = world_min
        self.world_max = world_max
        self.grid_size = grid_size
        self.cache_file = cache_file
        self.occupancy = self._build_occupancy()
        self.cache = {}
        self._load_cache()

    def _build_occupancy(self):
        occ = [[1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        cell = (self.world_max - self.world_min) / self.grid_size
        for ix in range(self.grid_size):
            for iy in range(self.grid_size):
                cx = self.world_min + (ix + 0.5) * cell
                cy = self.world_min + (iy + 0.5) * cell
                occ[ix][iy] = 0 if not check_pos(cx, cy) else 1
        return occ

    def world_to_grid(self, x, y):
        cell = (self.world_max - self.world_min) / self.grid_size
        ix = int((x - self.world_min) / cell)
        iy = int((y - self.world_min) / cell)
        ix = max(0, min(self.grid_size - 1, ix))
        iy = max(0, min(self.grid_size - 1, iy))
        return ix, iy

    def grid_to_world(self, ix, iy):
        cell = (self.world_max - self.world_min) / self.grid_size
        cx = self.world_min + (ix + 0.5) * cell
        cy = self.world_min + (iy + 0.5) * cell
        return cx, cy

    def neighbors(self, ix, iy):
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.occupancy[nx][ny] == 1:
                yield nx, ny

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _cache_key(self, sx, sy, gx, gy):
        return (sx, sy, gx, gy)

    def _load_cache(self):
        try:
            import pickle, os
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"Path cache loaded from {self.cache_file}")
        except Exception as e:
            print(f"Path cache load failed: {e}")

    def save_cache(self):
        try:
            import pickle
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"Path cache saved to {self.cache_file}")
        except Exception as e:
            print(f"Path cache save failed: {e}")

    def plan(self, start_w, goal_w):
        import heapq
        sx, sy = self.world_to_grid(start_w[0], start_w[1])
        gx, gy = self.world_to_grid(goal_w[0], goal_w[1])
        key = self._cache_key(sx, sy, gx, gy)
        if key in self.cache:
            return [self.grid_to_world(ix, iy) for ix, iy in self.cache[key]]
        start = (sx, sy); goal = (gx, gy)
        oheap = []
        heapq.heappush(oheap, (self.heuristic(start, goal), start))
        came_from = {}; gscore = {start: 0}; fscore = {start: self.heuristic(start, goal)}
        closed = set()
        while oheap:
            _, cur = heapq.heappop(oheap)
            if cur == goal:
                path = []
                while cur in came_from:
                    path.append(cur); cur = came_from[cur]
                path.append(start); path.reverse()
                self.cache[key] = path
                return [self.grid_to_world(ix, iy) for ix, iy in path]
            closed.add(cur)
            for nxt in self.neighbors(cur[0], cur[1]):
                tentative = gscore[cur] + 1
                if nxt in closed and tentative >= gscore.get(nxt, 1e9): continue
                if tentative < gscore.get(nxt, 1e9):
                    came_from[nxt] = cur
                    gscore[nxt] = tentative
                    fscore[nxt] = tentative + self.heuristic(nxt, goal)
                    heapq.heappush(oheap, (fscore[nxt], nxt))
        return []

def compute_theta_to_waypoint(px, py, heading, wx, wy):
    skew_x = wx - px
    skew_y = wy - py
    dot = skew_x * 1 + skew_y * 0
    mag1 = math.sqrt(skew_x ** 2 + skew_y ** 2)
    if mag1 < 1e-6:
        return 0.0
    beta = math.acos(dot / mag1)
    if skew_y < 0:
        beta = -beta if skew_x < 0 else -beta
    theta = beta - heading
    if theta > math.pi:
        theta = math.pi - theta
        theta = -math.pi - theta
    if theta < -math.pi:
        theta = -math.pi - theta
        theta = math.pi - theta
    return theta

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.action_size = action_size
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
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

class HybridDQN:
    def __init__(self, state_size, action_size, heuristic_weight=3.0):
        self.model = DuelingDQN(state_size, action_size).to(device)
        self.heuristic_weight = heuristic_weight
        self.action_map = {
            0: (0.6, 0.0),   # 前进
            1: (0.0, 0.8),   # 左转原地
            2: (0.0, -0.8),  # 右转原地
            3: (0.5, 0.5),   # 前进左
            4: (0.5, -0.5),  # 前进右
        }

    def load(self, filename="pytorch_models/hybrid_dqn_best.pth"):
        try:
            state = torch.load(filename, map_location=device)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"Hybrid DQN loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load Hybrid DQN: {e}")
            return False

    def act(self, state, theta_wp):
        state_tensor = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
        heuristic = np.zeros(len(self.action_map))
        if abs(theta_wp) < 0.3:
            heuristic[0] = 1.0
        if theta_wp > 0.2:
            heuristic[1] = 1.0
            heuristic[3] = 1.0
        if theta_wp < -0.2:
            heuristic[2] = 1.0
            heuristic[4] = 1.0
        combined_q = q_values + self.heuristic_weight * heuristic
        action_idx = int(np.argmax(combined_q))
        linear, angular = self.action_map[action_idx]
        a_in = [max(0.0, min(1.0, linear)), max(-1.0, min(1.0, angular))]
        return a_in

# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2  # 原TD3连续动作维度

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

# A* planner与混合DQN初始化
planner = AStarPlanner(world_min=-4.5, world_max=4.5, grid_size=50, cache_file="gazebo_path_cache.pkl")
hybrid = HybridDQN(state_size=state_dim, action_size=5, heuristic_weight=3.0)
# 尝试加载混合模型（若存在）
hybrid.load("pytorch_models/hybrid_dqn_best.pth")

done = False
episode_timesteps = 0
state = env.reset()
path = []
path_idx = 0

# Begin the testing loop
while True:
    if USE_HYBRID:
        px, py, heading = env.get_pose()
        gx, gy = env.goal_x, env.goal_y

        if not path or path_idx >= len(path):
            path = planner.plan((px, py), (gx, gy))
            path_idx = 0

        if not path:
            wx, wy = gx, gy
        else:
            wx, wy = path[min(path_idx, len(path) - 1)]
            if math.hypot(px - wx, py - wy) < 0.2 and path_idx < len(path) - 1:
                path_idx += 1
                wx, wy = path[path_idx]

        theta_wp = compute_theta_to_waypoint(px, py, heading, wx, wy)
        a_in = hybrid.act(state, theta_wp)
    else:
        # 原TD3推理
        action = network.get_action(np.array(state))
        a_in = [(action[0] + 1) / 2, action[1]]

    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    if done:
        state = env.reset()
        done = False
        episode_timesteps = 0
        path = []
        path_idx = 0
        # 可选：持久化路径缓存（测试循环无限，这里仅示例）
        planner.save_cache()
    else:
        state = next_state
        episode_timesteps += 1
    # 可选：保存路径缓存（测试循环是无限的，建议在退出时另存；此处不强制）
