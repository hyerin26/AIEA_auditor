import os
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Settings

ENV_NAME = "CarRacing-v3"
FRAME_STACK = 4
NUM_EVAL_EPISODES = 10

MODEL_PATH = "./dqn_results/models/dqn_car_racing_final.pt"
SAVE_DIR = "./dqn_results/evaluation"

os.makedirs(SAVE_DIR, exist_ok=True)

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


# DQN Model

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


# Greedy Action Selection

def select_greedy_action(q_network, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = q_network(state_tensor)
        action = q_values.argmax(dim=1).item()

    return action


# Evaluation

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

    q_network = DQN(num_actions).to(device)
    q_network.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    q_network.eval()

    frame_stack = FrameStack(FRAME_STACK)

    episode_rewards = []

    for episode in range(1, NUM_EVAL_EPISODES + 1):
        obs, info = env.reset()
        state = frame_stack.reset(obs)

        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            action = select_greedy_action(q_network, state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = frame_stack.step(next_obs)

            state = next_state
            episode_reward += reward
            step_count += 1

        episode_rewards.append(episode_reward)

        print(
            f"Evaluation Episode {episode} | "
            f"Reward: {episode_reward:.2f} | "
            f"Steps: {step_count}"
        )

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("=" * 50)
    print(f"Mean Evaluation Reward: {mean_reward:.2f}")
    print(f"Std Evaluation Reward: {std_reward:.2f}")
    print("=" * 50)

    # Save evaluation results
    result_path = os.path.join(SAVE_DIR, "evaluation_results.txt")
    with open(result_path, "w") as f:
        f.write("DQN CarRacing Evaluation Results\n")
        f.write("=" * 40 + "\n")
        for i, reward in enumerate(episode_rewards, start=1):
            f.write(f"Episode {i}: {reward:.2f}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Mean Reward: {mean_reward:.2f}\n")
        f.write(f"Std Reward: {std_reward:.2f}\n")

    # Save evaluation reward plot
    plt.figure()
    plt.plot(range(1, NUM_EVAL_EPISODES + 1), episode_rewards, marker="o")
    plt.axhline(mean_reward, linestyle="--", label=f"Mean: {mean_reward:.2f}")
    plt.title("DQN Evaluation Rewards")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "evaluation_rewards.png"))
    plt.close()

    np.save(os.path.join(SAVE_DIR, "evaluation_rewards.npy"), np.array(episode_rewards))

    print("Evaluation results saved in:", SAVE_DIR)


if __name__ == "__main__":
    main()
