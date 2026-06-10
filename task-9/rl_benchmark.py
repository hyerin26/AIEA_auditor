import os
import json
import time
import importlib

import numpy as np
import csv
import matplotlib.pyplot as plt


# Benchmark settings

TOTAL_STEPS = 1_000_000
SEEDS = [0, 1]
EVAL_FREQ = 20_000

FINAL_FRACTION = 0.2

RESULT_ROOT = "./train_results"
BENCHMARK_DIR = "./benchmark_results"

os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)


ALGORITHMS = [
    {"name": "DQN", "key": "dqn", "module": "train_scripts.dqn"},
    {"name": "A2C", "key": "a2c", "module": "train_scripts.a2c"},
    {"name": "Fine-tuned A2C", "key": "finetuned_a2c", "module": "train_scripts.finetuned_a2c"},
]


def load_array(path):
    if os.path.exists(path):
        return np.load(path)
    return None


def seed_dir(algo_key, seed):
    return os.path.join(RESULT_ROOT, algo_key, f"seed_{seed}")


def run_algorithm_seed(algo, seed):
    name = algo["name"]
    module_name = algo["module"]
    save_dir = seed_dir(algo["key"], seed)

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print(f"Running {name} | seed {seed}")
    print(f"Module: {module_name}")
    print(f"Save directory: {save_dir}")
    print("=" * 80)

    start_time = time.time()

    module = importlib.import_module(module_name)
    summary = module.main(total_steps=TOTAL_STEPS, save_dir=save_dir, seed=seed)

    elapsed_minutes = (time.time() - start_time) / 60

    if summary is None:
        summary = {}

    summary["algorithm"] = name
    summary["seed"] = seed
    summary["benchmark_total_steps"] = TOTAL_STEPS
    summary["benchmark_time_minutes"] = elapsed_minutes
    summary["save_dir"] = save_dir

    with open(os.path.join(save_dir, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Finished {name} | seed {seed} in {elapsed_minutes:.2f} min\n")

    return summary


def common_grid():
    return np.arange(EVAL_FREQ, TOTAL_STEPS + 1, EVAL_FREQ)


def collect_eval_curves():
    """For each algorithm, return per-seed eval curves interpolated onto a common grid."""
    grid = common_grid()
    curves = {}

    for algo in ALGORITHMS:
        name = algo["name"]
        per_seed = []

        for seed in SEEDS:
            sd = seed_dir(algo["key"], seed)
            es = load_array(os.path.join(sd, "eval_steps.npy"))
            er = load_array(os.path.join(sd, "eval_rewards.npy"))

            if es is None or er is None or len(es) == 0 or len(es) != len(er):
                continue

            # np.interp needs increasing x; eval_steps are monotonically increasing.
            interp = np.interp(grid, es, er)
            per_seed.append(interp)

        if len(per_seed) > 0:
            curves[name] = np.array(per_seed)  # shape: [n_seeds, n_grid]

    return grid, curves


def save_eval_comparison(grid, curves):
    plt.figure()

    for name, c in curves.items():
        mean = c.mean(axis=0)
        std = c.std(axis=0)

        plt.plot(grid, mean, label=f"{name} (n={c.shape[0]})")
        if c.shape[0] > 1:
            plt.fill_between(grid, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Training Step (env interactions)")
    plt.ylabel("Mean Eval Reward (greedy, fixed tracks)")
    plt.title(f"Evaluation Reward Comparison (mean +/- std over {len(SEEDS)} seeds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BENCHMARK_DIR, "eval_reward_comparison.png"))
    plt.close()


def save_finetuned_extra_plots():
    """Schedules are deterministic; just plot from seed 0 if present."""
    sd = seed_dir("finetuned_a2c", SEEDS[0])
    update_steps = load_array(os.path.join(sd, "update_steps.npy"))
    entropy_coefs = load_array(os.path.join(sd, "entropy_coefs.npy"))
    learning_rates = load_array(os.path.join(sd, "learning_rates.npy"))

    if entropy_coefs is not None and len(entropy_coefs) > 0:
        plt.figure()
        if update_steps is not None and len(update_steps) == len(entropy_coefs):
            plt.plot(update_steps, entropy_coefs)
        else:
            plt.plot(entropy_coefs)
        plt.xlabel("Training Step")
        plt.ylabel("Entropy Coefficient")
        plt.title("Fine-tuned A2C Entropy Decay")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(BENCHMARK_DIR, "finetuned_entropy_decay.png"))
        plt.close()

    if learning_rates is not None and len(learning_rates) > 0:
        plt.figure()
        if update_steps is not None and len(update_steps) == len(learning_rates):
            plt.plot(update_steps, learning_rates)
        else:
            plt.plot(learning_rates)
        plt.xlabel("Training Step")
        plt.ylabel("Learning Rate")
        plt.title("Fine-tuned A2C Learning Rate Schedule")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(BENCHMARK_DIR, "finetuned_lr_schedule.png"))
        plt.close()


def save_summary_csv(grid, curves):
    n_last = max(1, int(round(len(grid) * FINAL_FRACTION)))
    rows = []

    for algo in ALGORITHMS:
        name = algo["name"]

        if name not in curves:
            rows.append({
                "algorithm": name, "n_seeds": 0,
                "final_eval_mean": None, "final_eval_std": None,
                "best_eval_mean": None, "auc_mean": None,
            })
            continue

        c = curves[name]
        per_seed_final = c[:, -n_last:].mean(axis=1)
        per_seed_best = c.max(axis=1)
        per_seed_auc = c.mean(axis=1)

        rows.append({
            "algorithm": name,
            "n_seeds": int(c.shape[0]),
            "final_eval_mean": float(per_seed_final.mean()),
            "final_eval_std": float(per_seed_final.std()),
            "best_eval_mean": float(per_seed_best.mean()),
            "auc_mean": float(per_seed_auc.mean()),
        })

    csv_path = os.path.join(BENCHMARK_DIR, "benchmark_summary.csv")
    fieldnames = ["algorithm", "n_seeds", "final_eval_mean",
                  "final_eval_std", "best_eval_mean", "auc_mean"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved summary CSV:", csv_path)
    for row in rows:
        print(row)


def generate_all_plots_and_summary():
    grid, curves = collect_eval_curves()

    if len(curves) == 0:
        print("WARNING: no eval curves found. Did the training scripts save eval_*.npy?")
        return

    save_eval_comparison(grid, curves)
    save_finetuned_extra_plots()
    save_summary_csv(grid, curves)

    print("Saved benchmark plots in:", BENCHMARK_DIR)


def main():
    for algo in ALGORITHMS:
        for seed in SEEDS:
            run_algorithm_seed(algo, seed)

    generate_all_plots_and_summary()

    print("=" * 80)
    print("Benchmark finished.")
    print("Results saved in:", BENCHMARK_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
