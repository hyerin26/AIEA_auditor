import os
import cv2
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# Settings

ENV_NAME = "CarRacing-v3"
FRAME_STACK = 4
IMAGE_SIZE = 84

NUM_EVAL_EPISODES = 10

MODEL_PATH = "./a2c_results/models/a2c_car_racing_final.pt"
SAVE_DIR = "./a2c_results/evaluation"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Preprocessing

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
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


# Actor-Critic Network

class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, value


# Environment

def make_env():
    env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        continuous=False,
        lap_complete_percent=0.95,
        domain_randomize=False,
    )
    return env


# Evaluation

def main():
    env = make_env()
    num_actions = env.action_space.n

    print("Action space:", env.action_space)

    model = ActorCritic(num_actions).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    frame_stack = FrameStack(FRAME_STACK)

    episode_rewards = []

    for episode in range(1, NUM_EVAL_EPISODES + 1):
        obs, info = env.reset()
        state = frame_stack.reset(obs)

        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value = model(state_tensor)

                # Greedy evaluation: choose the action with the highest policy logit.
                action = torch.argmax(logits, dim=1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = frame_stack.step(next_obs)

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
    best_reward = np.max(episode_rewards)
    worst_reward = np.min(episode_rewards)

    print("=" * 50)
    print(f"Mean Evaluation Reward: {mean_reward:.2f}")
    print(f"Std Evaluation Reward: {std_reward:.2f}")
    print(f"Best Evaluation Reward: {best_reward:.2f}")
    print(f"Worst Evaluation Reward: {worst_reward:.2f}")
    print("=" * 50)

    result_path = os.path.join(SAVE_DIR, "a2c_evaluation_results.txt")

    with open(result_path, "w") as f:
        f.write("A2C CarRacing Evaluation Results\n")
        f.write("=" * 40 + "\n")

        for i, reward in enumerate(episode_rewards, start=1):
            f.write(f"Episode {i}: {reward:.2f}\n")

        f.write("=" * 40 + "\n")
        f.write(f"Mean Reward: {mean_reward:.2f}\n")
        f.write(f"Std Reward: {std_reward:.2f}\n")
        f.write(f"Best Reward: {best_reward:.2f}\n")
        f.write(f"Worst Reward: {worst_reward:.2f}\n")

    plt.figure()
    plt.plot(range(1, NUM_EVAL_EPISODES + 1), episode_rewards, marker="o")
    plt.axhline(mean_reward, linestyle="--", label=f"Mean: {mean_reward:.2f}")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Reward")
    plt.title("A2C Evaluation Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "a2c_evaluation_rewards.png"))
    plt.close()

    np.save(
        os.path.join(SAVE_DIR, "a2c_evaluation_rewards.npy"),
        np.array(episode_rewards)
    )

    print("Evaluation results saved in:", SAVE_DIR)


if __name__ == "__main__":
    main()
