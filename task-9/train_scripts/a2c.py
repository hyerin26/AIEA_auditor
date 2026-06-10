import os
import cv2
import time
import json
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

TOTAL_STEPS = 200_000
N_STEPS = 20
NUM_ENVS = 4

GAMMA = 0.99
LR = 5e-5

ENTROPY_COEF = 0.005
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5

FRAME_STACK = 4
IMAGE_SIZE = 84

CLIP_REWARD = True

EVAL_FREQ = 20_000
N_EVAL_EPISODES = 3
EVAL_SEED_BASE = 10_000

SAVE_FREQ = 50_000

DEFAULT_SAVE_DIR = "./train_results/a2c_results"


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

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
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


# Deterministic Evaluation

def evaluate(model, num_episodes):
    """Greedy (argmax logits) policy on a fixed set of eval tracks. Raw reward."""
    eval_env = make_env()
    frame_stack = FrameStack(FRAME_STACK)

    model.eval()
    returns = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset(seed=EVAL_SEED_BASE + ep)
        state = frame_stack.reset(obs)

        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(state_tensor)
                action = logits.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated

            state = frame_stack.step(obs)

        returns.append(total_reward)

    model.train()
    eval_env.close()

    return float(np.mean(returns)), float(np.std(returns))


# Moving Average

def moving_average(values, window=10):
    values = np.array(values)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


# Save Plots

def save_training_plots(
    episode_rewards,
    episode_steps,
    eval_steps,
    eval_rewards,
    eval_rewards_std,
    losses,
    policy_losses,
    value_losses,
    update_steps,
    save_dir,
    plot_dir,
):
    np.save(os.path.join(save_dir, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(save_dir, "episode_steps.npy"), np.array(episode_steps))
    np.save(os.path.join(save_dir, "losses.npy"), np.array(losses))
    np.save(os.path.join(save_dir, "policy_losses.npy"), np.array(policy_losses))
    np.save(os.path.join(save_dir, "value_losses.npy"), np.array(value_losses))
    np.save(os.path.join(save_dir, "update_steps.npy"), np.array(update_steps))

    np.save(os.path.join(save_dir, "eval_steps.npy"), np.array(eval_steps))
    np.save(os.path.join(save_dir, "eval_rewards.npy"), np.array(eval_rewards))
    np.save(os.path.join(save_dir, "eval_rewards_std.npy"), np.array(eval_rewards_std))

    plt.figure()
    plt.plot(episode_steps, episode_rewards, alpha=0.5, label="Train Episode Reward (raw)")

    if len(episode_rewards) >= 10:
        ma_rewards = moving_average(episode_rewards, window=10)
        plt.plot(episode_steps[9:], ma_rewards, label="10-Episode Moving Average")

    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("A2C Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "a2c_training_rewards.png"))
    plt.close()

    if len(eval_rewards) > 0:
        plt.figure()
        plt.plot(eval_steps, eval_rewards, label="Eval Reward (greedy)")
        plt.xlabel("Training Step")
        plt.ylabel("Mean Eval Reward")
        plt.title("A2C Eval Rewards")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "a2c_eval_rewards.png"))
        plt.close()

    if len(losses) > 0:
        plt.figure()
        plt.plot(update_steps, losses, alpha=0.5, label="Total Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("A2C Training Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "a2c_training_loss.png"))
        plt.close()

    if len(policy_losses) > 0:
        plt.figure()
        plt.plot(update_steps, policy_losses, alpha=0.5, label="Policy Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Policy Loss")
        plt.title("A2C Policy Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "a2c_policy_loss.png"))
        plt.close()

    if len(value_losses) > 0:
        plt.figure()
        plt.plot(update_steps, value_losses, alpha=0.5, label="Value Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Value Loss")
        plt.title("A2C Value Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "a2c_value_loss.png"))
        plt.close()


# Main Training

def main(total_steps=TOTAL_STEPS, save_dir=DEFAULT_SAVE_DIR, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_dir = os.path.join(save_dir, "models")
    plot_dir = os.path.join(save_dir, "plots")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    envs = [make_env() for _ in range(NUM_ENVS)]
    num_actions = envs[0].action_space.n

    print("A2C from scratch on CarRacing-v3")
    print("Seed:", seed)
    print("Action space:", envs[0].action_space)
    print("Number of actions:", num_actions)
    print("Total steps:", total_steps)
    print("Number of environments:", NUM_ENVS)

    model = ActorCritic(num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    frame_stacks = [FrameStack(FRAME_STACK) for _ in range(NUM_ENVS)]

    states = []

    for env_idx in range(NUM_ENVS):
        obs, info = envs[env_idx].reset(seed=seed * 100 + env_idx)
        state = frame_stacks[env_idx].reset(obs)
        states.append(state)

    states = np.stack(states, axis=0)

    episode_rewards_running = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_count = 0
    global_step = 0
    last_saved_step = 0
    next_eval_step = EVAL_FREQ

    episode_rewards = []
    episode_steps = []
    losses = []
    policy_losses = []
    value_losses = []
    update_steps = []

    eval_steps = []
    eval_rewards = []
    eval_rewards_std = []

    start_time = time.time()

    while global_step < total_steps:
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

                episode_rewards_running[env_idx] += reward  # raw for reporting

                learn_reward = float(np.clip(reward, -1.0, 1.0)) if CLIP_REWARD else float(reward)

                if done:
                    episode_count += 1
                    episode_rewards.append(episode_rewards_running[env_idx])
                    episode_steps.append(global_step)

                    print(
                        f"[seed {seed}] Step: {global_step} | "
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
                step_rewards.append(learn_reward)
                step_dones.append(done)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(step_rewards, dtype=torch.float32).to(device))
            entropies.append(entropy)
            dones.append(torch.tensor(step_dones, dtype=torch.float32).to(device))

            states = np.stack(next_states, axis=0)
            global_step += NUM_ENVS

            if global_step >= total_steps:
                break

        if len(rewards) == 0:
            continue

        state_tensor = torch.tensor(states, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, next_values = model(state_tensor)

        last_dones = torch.tensor(step_dones, dtype=torch.float32).to(device).unsqueeze(1)
        R = next_values.detach() * (1.0 - last_dones)

        policy_loss = torch.zeros(1).to(device)
        value_loss = torch.zeros(1).to(device)
        entropy_loss = torch.zeros(1).to(device)

        for i in reversed(range(len(rewards))):
            reward_i = rewards[i].unsqueeze(1)
            done_i = dones[i].unsqueeze(1)

            R = reward_i + GAMMA * R * (1.0 - done_i)
            advantage = R - values[i]

            policy_loss = policy_loss - (log_probs[i].unsqueeze(1) * advantage.detach()).mean()
            value_loss = value_loss + advantage.pow(2).mean()
            entropy_loss = entropy_loss + entropies[i].mean()

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
        update_steps.append(global_step)

        if global_step >= next_eval_step:
            eval_mean, eval_std = evaluate(model, N_EVAL_EPISODES)
            eval_steps.append(global_step)
            eval_rewards.append(eval_mean)
            eval_rewards_std.append(eval_std)
            next_eval_step += EVAL_FREQ
            print(f"[seed {seed}] EVAL @ {global_step}: {eval_mean:.2f} +/- {eval_std:.2f}")

        if global_step - last_saved_step >= SAVE_FREQ:
            checkpoint_path = os.path.join(model_dir, f"a2c_step_{global_step}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            last_saved_step = global_step

    end_time = time.time()

    for env in envs:
        env.close()

    final_model_path = os.path.join(model_dir, "a2c_car_racing_final.pt")
    torch.save(model.state_dict(), final_model_path)

    save_training_plots(
        episode_rewards,
        episode_steps,
        eval_steps,
        eval_rewards,
        eval_rewards_std,
        losses,
        policy_losses,
        value_losses,
        update_steps,
        save_dir,
        plot_dir,
    )

    summary = {
        "algorithm": "A2C",
        "seed": seed,
        "total_steps": total_steps,
        "num_envs": NUM_ENVS,
        "n_steps": N_STEPS,
        "num_episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else None,
        "last_10_mean_reward": float(np.mean(episode_rewards[-10:])) if len(episode_rewards) >= 10 else None,
        "best_reward": float(np.max(episode_rewards)) if len(episode_rewards) > 0 else None,
        "final_eval_reward": float(eval_rewards[-1]) if len(eval_rewards) > 0 else None,
        "best_eval_reward": float(np.max(eval_rewards)) if len(eval_rewards) > 0 else None,
        "mean_loss": float(np.mean(losses)) if len(losses) > 0 else None,
        "mean_policy_loss": float(np.mean(policy_losses)) if len(policy_losses) > 0 else None,
        "mean_value_loss": float(np.mean(value_losses)) if len(value_losses) > 0 else None,
        "training_time_minutes": float((end_time - start_time) / 60),
        "final_model_path": final_model_path,
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("Training finished.")
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")
    print("Saved final model:", final_model_path)
    print("Saved plots in:", plot_dir)

    return summary


if __name__ == "__main__":
    main()
