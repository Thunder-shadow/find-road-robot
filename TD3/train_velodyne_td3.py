import os
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from squaternion import Quaternion

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv, check_pos


def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward


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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not

# 顶部插入：开关、A*（带缓存）、角误差计算、DuelingDQN与混合训练器
USE_HYBRID = True

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
    skew_x = wx - px; skew_y = wy - py
    dot = skew_x * 1 + skew_y * 0
    mag1 = math.sqrt(skew_x ** 2 + skew_y ** 2)
    if mag1 < 1e-6: return 0.0
    beta = math.acos(dot / mag1)
    if skew_y < 0: beta = -beta if skew_x < 0 else -beta
    theta = beta - heading
    if theta > math.pi:
        theta = math.pi - theta; theta = -math.pi - theta
    if theta < -math.pi:
        theta = -math.pi - theta; theta = math.pi - theta
    return theta

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(nn.Linear(state_size, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.val = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size))
    def forward(self, s):
        f = self.feature(s)
        v = self.val(f)
        a = self.adv(f)
        return v + (a - a.mean(dim=1, keepdim=True))

class HybridDQNTrainer:
    def __init__(self, state_size, action_size, device, lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, tau=0.005, heuristic_weight=1.5):
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.heuristic_weight = heuristic_weight
        self.batch_size = 128
        self.update_every = 4
        self.learn_step = 0
        self.memory = []
        self.max_mem = 100000
        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target = DuelingDQN(state_size, action_size).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.crit = nn.SmoothL1Loss()
        self.action_map = {
            0: (0.6, 0.0),
            1: (0.0, 0.8),
            2: (0.0, -0.8),
            3: (0.5, 0.5),
            4: (0.5, -0.5),
        }
    def save(self, filename="pytorch_models/hybrid_dqn_latest.pth"):
        try:
            torch.save({"model_state_dict": self.model.state_dict()}, filename)
            print(f"Hybrid DQN saved to {filename}")
        except Exception as e:
            print(f"Hybrid DQN save failed: {e}")
    def load(self, filename="pytorch_models/hybrid_dqn_latest.pth"):
        try:
            state = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self.target.load_state_dict(self.model.state_dict())
            print(f"Hybrid DQN loaded from {filename}")
            return True
        except Exception as e:
            print(f"Hybrid DQN load failed: {e}")
            return False
    def act(self, state, theta_wp):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, len(self.action_map))
        else:
            s = torch.FloatTensor(np.array(state).reshape(1, -1)).to(self.device)
            with torch.no_grad():
                q = self.model(s).cpu().numpy()[0]
            heuristic = np.zeros(len(self.action_map))
            if abs(theta_wp) < 0.3: heuristic[0] = 1.0
            if theta_wp > 0.2: heuristic[1] = 1.0; heuristic[3] = 1.0
            if theta_wp < -0.2: heuristic[2] = 1.0; heuristic[4] = 1.0
            combined = q + self.heuristic_weight * heuristic
            action_idx = int(np.argmax(combined))
        lin, ang = self.action_map[action_idx]
        return [max(0.0, min(1.0, lin)), max(-1.0, min(1.0, ang))], action_idx
    def remember(self, s, a_idx, r, s2, done):
        if len(self.memory) >= self.max_mem: self.memory.pop(0)
        self.memory.append((s, a_idx, r, s2, done))
    def replay(self):
        self.learn_step += 1
        if len(self.memory) < self.batch_size or self.learn_step % self.update_every != 0:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.crit(current_q, target_q)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step()
        for tp, p in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Begin the training loop
while timestep < max_timesteps:

    # On termination of episode
    if done:
        if timestep != 0:
            # 记录上一个episode的奖励并更新计数
            prev_episode_reward = episode_reward
            episodes_completed += 1
            # 定期保存模型
            if (e + 1) % save_interval == 0:
                if USE_HYBRID:
                    hybrid_trainer.save("pytorch_models/hybrid_dqn_latest.pth")
                    try:
                        best_reward
                    except NameError:
                        best_reward = -np.inf
                    if total_reward > best_reward:
                        best_reward = total_reward
                        hybrid_trainer.save("pytorch_models/hybrid_dqn_best.pth")
                        planner.save_cache()
                    else:
                        network.save(f'{"TD3_velodyne"}', directory="./pytorch_models")
            if not USE_HYBRID:
                network.train(
                    replay_buffer,
                    episode_timesteps,
                    batch_size,
                    discount,
                    tau,
                    policy_noise,
                    noise_clip,
                    policy_freq,
                )
        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # add some exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    if USE_HYBRID:
        # 训练主循环中的动作选择与更新（加入混合分支）
        try:
            hybrid_trainer
        except NameError:
            hybrid_trainer = HybridDQNTrainer(state_size=state_dim, action_size=5, device=device, heuristic_weight=1.5)
            planner = AStarPlanner(world_min=-4.5, world_max=4.5, grid_size=50, cache_file="gazebo_path_cache.pkl")
            path = []; path_idx = 0
            hybrid_trainer.load("pytorch_models/hybrid_dqn_latest.pth")

        # 读取位姿与目标
        if env.last_odom is not None:
            q = env.last_odom.pose.pose.orientation
            from squaternion import Quaternion
            heading = round(Quaternion(q.w, q.x, q.y, q.z).to_euler(degrees=False)[2], 4)
        else:
            heading = 0.0
        px, py = env.odom_x, env.odom_y
        gx, gy = env.goal_x, env.goal_y

        # 路径规划推进
        if not path or path_idx >= len(path):
            path = planner.plan((px, py), (gx, gy)); path_idx = 0
        if not path:
            wx, wy = gx, gy
        else:
            wx, wy = path[min(path_idx, len(path) - 1)]
            if math.hypot(px - wx, py - wy) < 0.2 and path_idx < len(path) - 1:
                path_idx += 1; wx, wy = path[path_idx]

        theta_wp = compute_theta_to_waypoint(px, py, heading, wx, wy)
        a_in, a_idx = hybrid_trainer.act(state, theta_wp)
    else:
        action = network.get_action(np.array(state))
        a_in = [(action[0] + 1) / 2, action[1]]

    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    if USE_HYBRID:
        hybrid_trainer.remember(state, a_idx, reward, next_state, done_bool)
        hybrid_trainer.replay()
    else:
        replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# 训练结束后保存模型与路径缓存
if save_model:
    if USE_HYBRID:
        try:
            hybrid_trainer.save("pytorch_models/hybrid_dqn_final.pth")
        except Exception:
            pass
        try:
            planner.save_cache()
        except Exception:
            pass
    else:
        try:
            network.save("%s" % file_name, directory="./pytorch_models")
        except Exception:
            pass
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    if USE_HYBRID:
        hybrid_trainer.save("pytorch_models/hybrid_dqn_final.pth")
        planner.save_cache()
    else:
        network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)
