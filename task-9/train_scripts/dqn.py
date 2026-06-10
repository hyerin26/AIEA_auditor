import os
import json
import random
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Hyperparameters

ENV_NAME = "CarRacing-v3"

TOTAL_STEPS = 200_000
LEARNING_STARTS = 5_000
BUFFER_SIZE = 100_000
BATCH_SIZE = 32

GAMMA = 0.99
LR = 1e-4

TRAIN_FREQ = 4
TARGET_UPDATE_FREQ = 1_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100_000

FRAME_STACK = 4

# Reward clipping for the LEARNING signal (kept consistent across all algorithms).
# Recorded / evaluated episode rewards always use the raw reward.
CLIP_REWARD = True

# Evaluation protocol (shared by all algorithms).
EVAL_FREQ = 20_000
# env steps between evaluations
N_EVAL_EPISODES = 3         # deterministic episodes per evaluation
EVAL_SEED_BASE = 10_000     # fixed -> all algos/seeds see the SAME eval tracks

DEFAULT_SAVE_DIR = "./train_results/dqn_results"


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Preprocessing

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized.astype(np.float32)


class FrameStack:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        processed = preprocess_frame(frame)

        for _ in range(self.num_frames):
            self.frames.append(processed)

        return self.get_state()

    def step(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return self.get_state()

    def get_state(self):
        return np.stack(self.frames, axis=0)


# Replay Buffer

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Q-Network

class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),

            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x)


# Epsilon Schedule

def get_epsilon(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END

    return EPS_START - (EPS_START - EPS_END) * (step / EPS_DECAY_STEPS)


# Action Selection

def select_action(q_network, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = q_network(state_tensor)
        action = q_values.argmax(dim=1).item()

    return action


# Deterministic Evaluation

def make_env():
    return gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        continuous=False,
        lap_complete_percent=0.95,
        domain_randomize=False,
    )


def evaluate(q_network, num_episodes, num_actions):
    """Greedy (argmax) policy on a fixed set of eval tracks. Raw reward."""
    eval_env = make_env()
    frame_stack = FrameStack(FRAME_STACK)

    q_network.eval()
    returns = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset(seed=EVAL_SEED_BASE + ep)
        state = frame_stack.reset(obs)

        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = q_network(state_tensor).argmax(dim=1).item()

            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

            state = frame_stack.step(obs)

        returns.append(total_reward)

    q_network.train()
    eval_env.close()

    return float(np.mean(returns)), float(np.std(returns))


# Training Step

def train_step(q_network, target_network, replay_buffer, optimizer):
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    current_q_values = q_network(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1, keepdim=True)[0]
        # dones here are TERMINATED only -> on time-limit truncation we still bootstrap.
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


# Plotting

def save_plot(x_values, y_values, title, xlabel, ylabel, filename):
    plt.figure()

    if x_values is not None and len(x_values) == len(y_values):
        plt.plot(x_values, y_values)
    else:
        plt.plot(y_values)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Main Training Loop

def main(total_steps=TOTAL_STEPS, save_dir=DEFAULT_SAVE_DIR, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_dir = os.path.join(save_dir, "models")
    plot_dir = os.path.join(save_dir, "plots")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    env = make_env()
    num_actions = env.action_space.n

    print("DQN on CarRacing-v3")
    print("Seed:", seed)
    print("Action space:", env.action_space)
    print("Number of actions:", num_actions)
    print("Total steps:", total_steps)

    q_network = DQN(num_actions).to(device)
    target_network = DQN(num_actions).to(device)

    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    frame_stack = FrameStack(FRAME_STACK)

    episode_rewards = []
    episode_steps = []
    losses = []
    loss_steps = []
    epsilons = []
    epsilon_steps = []

    eval_steps = []
    eval_rewards = []
    eval_rewards_std = []

    obs, info = env.reset(seed=seed)
    state = frame_stack.reset(obs)

    episode_reward = 0.0
    episode_count = 0
    next_eval_step = EVAL_FREQ

    for step in range(1, total_steps + 1):
        epsilon = get_epsilon(step)
        action = select_action(q_network, state, epsilon, num_actions)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = frame_stack.step(next_obs)

        learn_reward = float(np.clip(reward, -1.0, 1.0)) if CLIP_REWARD else float(reward)

        # Store TERMINATED (not done) so time-limit truncations keep bootstrapping.
        replay_buffer.push(state, action, learn_reward, next_state, float(terminated))

        state = next_state
        episode_reward += reward  # raw reward for reporting

        if step > LEARNING_STARTS and step % TRAIN_FREQ == 0:
            loss = train_step(q_network, target_network, replay_buffer, optimizer)
            losses.append(loss)
            loss_steps.append(step)

        if step % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(q_network.state_dict())

        if step % 1000 == 0:
            epsilons.append(epsilon)
            epsilon_steps.append(step)

        if done:
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            episode_count += 1

            print(
                f"[seed {seed}] Step: {step} | "
                f"Episode: {episode_count} | "
                f"Reward: {episode_reward:.2f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Buffer: {len(replay_buffer)}"
            )

            obs, info = env.reset()
            state = frame_stack.reset(obs)
            episode_reward = 0.0

        if step >= next_eval_step:
            eval_mean, eval_std = evaluate(q_network, N_EVAL_EPISODES, num_actions)
            eval_steps.append(step)
            eval_rewards.append(eval_mean)
            eval_rewards_std.append(eval_std)
            next_eval_step += EVAL_FREQ
            print(f"[seed {seed}] EVAL @ {step}: {eval_mean:.2f} +/- {eval_std:.2f}")

        if step % 50_000 == 0:
            model_path = os.path.join(model_dir, f"dqn_step_{step}.pt")
            torch.save(q_network.state_dict(), model_path)
            print(f"Saved checkpoint: {model_path}")

    final_model_path = os.path.join(model_dir, "dqn_car_racing_final.pt")
    torch.save(q_network.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    env.close()

    np.save(os.path.join(save_dir, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(save_dir, "episode_steps.npy"), np.array(episode_steps))
    np.save(os.path.join(save_dir, "losses.npy"), np.array(losses))
    np.save(os.path.join(save_dir, "loss_steps.npy"), np.array(loss_steps))
    np.save(os.path.join(save_dir, "epsilons.npy"), np.array(epsilons))
    np.save(os.path.join(save_dir, "epsilon_steps.npy"), np.array(epsilon_steps))

    np.save(os.path.join(save_dir, "eval_steps.npy"), np.array(eval_steps))
    np.save(os.path.join(save_dir, "eval_rewards.npy"), np.array(eval_rewards))
    np.save(os.path.join(save_dir, "eval_rewards_std.npy"), np.array(eval_rewards_std))

    save_plot(
        episode_steps,
        episode_rewards,
        "DQN Training Episode Rewards (raw)",
        "Training Step",
        "Episode Reward",
        os.path.join(plot_dir, "episode_rewards.png"),
    )

    save_plot(
        eval_steps,
        eval_rewards,
        "DQN Eval Rewards (greedy)",
        "Training Step",
        "Mean Eval Reward",
        os.path.join(plot_dir, "eval_rewards.png"),
    )

    save_plot(
        loss_steps,
        losses,
        "DQN Training Loss",
        "Training Step",
        "Loss",
        os.path.join(plot_dir, "training_loss.png"),
    )

    save_plot(
        epsilon_steps,
        epsilons,
        "DQN Epsilon Decay",
        "Training Step",
        "Epsilon",
        os.path.join(plot_dir, "epsilon_decay.png"),
    )

    summary = {
        "algorithm": "DQN",
        "seed": seed,
        "total_steps": total_steps,
        "num_episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else None,
        "last_10_mean_reward": float(np.mean(episode_rewards[-10:])) if len(episode_rewards) >= 10 else None,
        "best_reward": float(np.max(episode_rewards)) if len(episode_rewards) > 0 else None,
        "final_eval_reward": float(eval_rewards[-1]) if len(eval_rewards) > 0 else None,
        "best_eval_reward": float(np.max(eval_rewards)) if len(eval_rewards) > 0 else None,
        "mean_loss": float(np.mean(losses)) if len(losses) > 0 else None,
        "final_model_path": final_model_path,
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("Training finished.")
    print("Plots saved in:", plot_dir)

    return summary


if __name__ == "__main__":
    main()
