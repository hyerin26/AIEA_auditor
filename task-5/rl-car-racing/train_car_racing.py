import os

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


def main():
    log_dir = "./logs/car_racing_ppo/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        continuous=True,
        lap_complete_percent=0.95,
        domain_randomize=False,
    )

    env = Monitor(env, log_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix="ppo_car_racing",
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu"
        )

    model.learn(
        total_timesteps=200_000,
        callback=checkpoint_callback,
        tb_log_name="PPO_CarRacing",
    )

    model.save("./models/ppo_car_racing_final")
    env.close()


if __name__ == "__main__":
    main()
