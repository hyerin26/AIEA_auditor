import os
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

SAVE_DIR = "./dqn_results"
MODEL_DIR = os.path.join(SAVE_DIR, "models")
PLOT_DIR = os.path.join(SAVE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Preprocessing

def preprocess_frame(frame):
    """
    Original CarRacing observation: RGB image with shape (96, 96, 3)

    Convert it to grayscale, resize to 84x84, and normalize to [0, 1].
    Output shape: (84, 84)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized.astype(np.float32)


class FrameStack:
    """
    Stores the latest 4 frames.
    """

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
        """
        Return shape: (4, 84, 84)
        """
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

            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x)


# Epsilon Schedule

def get_epsilon(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END

    epsilon = EPS_START - (EPS_START - EPS_END) * (step / EPS_DECAY_STEPS)
    return epsilon


# Action Selection

def select_action(q_network, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = q_network(state_tensor)
        action = q_values.argmax(dim=1).item()

    return action


# Training Step

def train_step(q_network, target_network, replay_buffer, optimizer):
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Current Q value: Q(s, a)
    current_q_values = q_network(states).gather(1, actions)

    # Target Q value:
    # y = r + gamma * max_a' Q_target(s', a')
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1, keepdim=True)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)

    optimizer.step()

    return loss.item()


# Plotting

def save_plot(values, title, ylabel, filename):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def save_step_plot(values, title, ylabel, filename):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Training Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


# Main Training Loop

def main():
    env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        continuous=False,
        lap_complete_percent=0.95,
        domain_randomize=False,
    )

    num_actions = env.action_space.n
    print("Action space:", env.action_space)
    print("Number of actions:", num_actions)

    q_network = DQN(num_actions).to(device)
    target_network = DQN(num_actions).to(device)

    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    frame_stack = FrameStack(FRAME_STACK)

    episode_rewards = []
    losses = []
    epsilons = []

    obs, info = env.reset()
    state = frame_stack.reset(obs)

    episode_reward = 0
    episode_count = 0

    for step in range(1, TOTAL_STEPS + 1):
        epsilon = get_epsilon(step)
        action = select_action(q_network, state, epsilon, num_actions)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = frame_stack.step(next_obs)

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Train Q-network
        if step > LEARNING_STARTS and step % TRAIN_FREQ == 0:
            loss = train_step(q_network, target_network, replay_buffer, optimizer)
            losses.append(loss)

        # Update target network
        if step % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Save epsilon value occasionally
        if step % 1000 == 0:
            epsilons.append(epsilon)

        # Episode ended
        if done:
            episode_rewards.append(episode_reward)
            episode_count += 1

            print(
                f"Step: {step} | "
                f"Episode: {episode_count} | "
                f"Reward: {episode_reward:.2f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Buffer: {len(replay_buffer)}"
            )

            obs, info = env.reset()
            state = frame_stack.reset(obs)
            episode_reward = 0

        # Save checkpoint
        if step % 50_000 == 0:
            model_path = os.path.join(MODEL_DIR, f"dqn_step_{step}.pt")
            torch.save(q_network.state_dict(), model_path)
            print(f"Saved checkpoint: {model_path}")

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, "dqn_car_racing_final.pt")
    torch.save(q_network.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    env.close()

    # Save plots
    save_plot(
        episode_rewards,
        "DQN CarRacing Episode Rewards",
        "Episode Reward",
        os.path.join(PLOT_DIR, "episode_rewards.png")
    )

    save_step_plot(
        losses,
        "DQN Training Loss",
        "Loss",
        os.path.join(PLOT_DIR, "training_loss.png")
    )

    save_step_plot(
        epsilons,
        "Epsilon Decay",
        "Epsilon",
        os.path.join(PLOT_DIR, "epsilon_decay.png")
    )

    # Save raw results
    np.save(os.path.join(SAVE_DIR, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(SAVE_DIR, "losses.npy"), np.array(losses))
    np.save(os.path.join(SAVE_DIR, "epsilons.npy"), np.array(epsilons))

    print("Training finished.")
    print("Plots saved in:", PLOT_DIR)


if __name__ == "__main__":
    main()
