# AIEA Auditor

This repository stores the work I completed as a CMPM 118 AIEA Auditor.

## Repository Structure

```text
AIEA_auditor/
├── README.md
├── task-5/
│   ├── rl-car-racing/
│   └── rl-carla-gym/
├── task-6/
│   ├── dqn.py
│   ├── evaluate_dqn.py
│   └── dqn_results/
└── task-7/
    ├── a2c.py
    ├── evaluate_a2c.py
    └── a2c_results/
````

## Task 5: Reinforcement Learning on Nautilus

The `task-5` folder contains files for running reinforcement learning experiments on the Nautilus deployment.

### `rl-car-racing/`

This folder contains the files for running one baseline reinforcement learning algorithm from Stable-Baselines3 on the Car Racing Gymnasium environment.

Contents include:

* `train_car_racing.py`: Training script for the CarRacing-v3 Gymnasium environment using a Stable-Baselines3 baseline algorithm.
* `logs/`: Training logs generated during the Car Racing experiment, including TensorBoard-related outputs.

### `rl-carla-gym/`

This folder contains the files for running CARLA Gym and plotting the training results with TensorBoard.

Contents include:

* `run.py`: Script used to run the CARLA Gym environment.
* `train.py`: Training script used for the CARLA Gym experiment.


## Task 6: DQN from Scratch on CarRacing-v3

The `task-6` folder contains my from-scratch implementation of Deep Q-Network (DQN) for the CarRacing-v3 Gymnasium environment.

### Files

* `dqn.py`: Training script for the DQN agent.
* `evaluate_dqn.py`: Evaluation script for the trained model.
* `dqn_results/`: Saved plots, model outputs, and evaluation results.


## Task 7: A2C from Scratch on CarRacing-v3

The task-7 folder contains my from-scratch implementation of Advantage Actor-Critic (A2C) for the CarRacing-v3 Gymnasium environment.

### Files
* 'a2c.py': Training script for the A2C agent.
* 'evaluate_a2c.py': Evaluation script for the trained model.
* 'a2c_results/': Saved plots, model outputs, and evaluation results.
