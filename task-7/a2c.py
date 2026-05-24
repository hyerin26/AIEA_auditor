import os
import cv2
import time
import random
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Hyperparameters

ENV_NAME = "CarRacing-v3"

TOTAL_STEPS = 1_000_000
N_STEPS = 20
NUM_ENVS = 4

GAMMA = 0.99
LR = 5e-5

ENTROPY_COEF = 0.005
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5

FRAME_STACK = 4
IMAGE_SIZE = 84

SAVE_FREQ = 50_000

SAVE_DIR = "./a2c_results"
MODEL_DIR = os.path.join(SAVE_DIR, "models")
PLOT_DIR = os.path.join(SAVE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# Device

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


# Action Selection

def select_actions(model, states):
    state_tensor = torch.tensor(states, dtype=torch.float32).to(device)

    logits, values = model(state_tensor)
    probs = F.softmax(logits, dim=-1)

    dist = Categorical(probs)
    actions = dist.sample()

    log_probs = dist.log_prob(actions)
    entropies = dist.entropy()

    return actions.cpu().numpy(), log_probs, entropies, values


# Moving Average

def moving_average(values, window=10):
    values = np.array(values)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


# Save Plots

def save_training_plots(episode_rewards, losses, policy_losses, value_losses):
    np.save(os.path.join(SAVE_DIR, "a2c_episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(SAVE_DIR, "a2c_losses.npy"), np.array(losses))
    np.save(os.path.join(SAVE_DIR, "a2c_policy_losses.npy"), np.array(policy_losses))
    np.save(os.path.join(SAVE_DIR, "a2c_value_losses.npy"), np.array(value_losses))

    # Reward plot
    plt.figure()
    plt.plot(episode_rewards, alpha=0.5, label="Episode Reward")

    if len(episode_rewards) >= 10:
        ma_rewards = moving_average(episode_rewards, window=10)
        plt.plot(
            range(9, 9 + len(ma_rewards)),
            ma_rewards,
            label="10-Episode Moving Average"
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("A2C CarRacing Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "a2c_training_rewards.png"))
    plt.close()

    # Total loss plot
    if len(losses) > 0:
        plt.figure()
        plt.plot(losses, alpha=0.4, label="Total Loss")

        if len(losses) >= 100:
            ma_losses = moving_average(losses, window=100)
            plt.plot(
                range(99, 99 + len(ma_losses)),
                ma_losses,
                label="100-Update Moving Average"
            )

        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.title("A2C Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, "a2c_training_loss.png"))
        plt.close()

    # Policy loss plot
    if len(policy_losses) > 0:
        plt.figure()
        plt.plot(policy_losses, alpha=0.4, label="Policy Loss")
        plt.xlabel("Update")
        plt.ylabel("Policy Loss")
        plt.title("A2C Policy Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, "a2c_policy_loss.png"))
        plt.close()

    # Value loss plot
    if len(value_losses) > 0:
        plt.figure()
        plt.plot(value_losses, alpha=0.4, label="Value Loss")
        plt.xlabel("Update")
        plt.ylabel("Value Loss")
        plt.title("A2C Value Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, "a2c_value_loss.png"))
        plt.close()


# Main Training

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    envs = [make_env() for _ in range(NUM_ENVS)]
    num_actions = envs[0].action_space.n

    print("A2C from scratch on CarRacing-v3")
    print("Action space:", envs[0].action_space)
    print("Number of actions:", num_actions)
    print("Total steps:", TOTAL_STEPS)
    print("Number of environments:", NUM_ENVS)

    model = ActorCritic(num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    frame_stacks = [FrameStack(FRAME_STACK) for _ in range(NUM_ENVS)]

    states = []

    for env_idx in range(NUM_ENVS):
        obs, info = envs[env_idx].reset()
        state = frame_stacks[env_idx].reset(obs)
        states.append(state)

    states = np.stack(states, axis=0)

    episode_rewards_running = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_count = 0
    global_step = 0
    last_saved_step = 0

    episode_rewards = []
    losses = []
    policy_losses = []
    value_losses = []

    start_time = time.time()

    while global_step < TOTAL_STEPS:
        log_probs = []
        values = []
        rewards = []
        entropies = []
        dones = []

        for _ in range(N_STEPS):
            actions, log_prob, entropy, value = select_actions(model, states)

            next_states = []
            step_rewards = []
            step_dones = []

            for env_idx in range(NUM_ENVS):
                next_obs, reward, terminated, truncated, info = envs[env_idx].step(actions[env_idx])
                done = terminated or truncated

                episode_rewards_running[env_idx] += reward

                clipped_reward = np.clip(reward, -1.0, 1.0)

                if done:
                    episode_count += 1
                    episode_rewards.append(episode_rewards_running[env_idx])

                    print(
                        f"Step: {global_step} | "
                        f"Episode: {episode_count} | "
                        f"Env: {env_idx} | "
                        f"Reward: {episode_rewards_running[env_idx]:.2f}"
                    )

                    obs, info = envs[env_idx].reset()
                    next_state = frame_stacks[env_idx].reset(obs)
                    episode_rewards_running[env_idx] = 0.0
                else:
                    next_state = frame_stacks[env_idx].step(next_obs)

                next_states.append(next_state)
                step_rewards.append(clipped_reward)
                step_dones.append(done)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(step_rewards, dtype=torch.float32).to(device))
            entropies.append(entropy)
            dones.append(torch.tensor(step_dones, dtype=torch.float32).to(device))

            states = np.stack(next_states, axis=0)
            global_step += NUM_ENVS

            if global_step >= TOTAL_STEPS:
                break

        if len(rewards) == 0:
            continue

        # Bootstrap value
        state_tensor = torch.tensor(states, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, next_values = model(state_tensor)

        last_dones = torch.tensor(step_dones, dtype=torch.float32).to(device).unsqueeze(1)
        R = next_values.detach() * (1.0 - last_dones)

        policy_loss = torch.zeros(1).to(device)
        value_loss = torch.zeros(1).to(device)
        entropy_loss = torch.zeros(1).to(device)

        # Compute n-step returns and losses
        for i in reversed(range(len(rewards))):
            reward_i = rewards[i].unsqueeze(1)
            done_i = dones[i].unsqueeze(1)

            R = reward_i + GAMMA * R * (1.0 - done_i)

            advantage = R - values[i]

            policy_loss = policy_loss - (log_probs[i].unsqueeze(1) * advantage.detach()).mean()
            value_loss = value_loss + advantage.pow(2).mean()
            entropy_loss = entropy_loss + entropies[i].mean()

        # Average losses over actual rollout length
        rollout_len = len(rewards)

        policy_loss = policy_loss / rollout_len
        value_loss = value_loss / rollout_len
        entropy_loss = entropy_loss / rollout_len

        loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()

        losses.append(loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        if global_step - last_saved_step >= SAVE_FREQ:
            checkpoint_path = os.path.join(MODEL_DIR, f"a2c_step_{global_step}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            last_saved_step = global_step

    end_time = time.time()

    for env in envs:
        env.close()

    final_model_path = os.path.join(MODEL_DIR, "a2c_car_racing_final.pt")
    torch.save(model.state_dict(), final_model_path)

    print("Training finished.")
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")
    print("Saved final model:", final_model_path)

    save_training_plots(episode_rewards, losses, policy_losses, value_losses)
    print("Saved plots in:", PLOT_DIR)


if __name__ == "__main__":
    main()
