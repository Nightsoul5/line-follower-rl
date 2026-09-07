import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from line_follower_env import LineFollowerEnv

# ------------------------------------------------------------ configuration
ALGO = "DQN"                  # "PPO" or "DQN"
REWARD_MODE = "progress"      # "progress" (v2) or "presence" (v1)
TOTAL_TIMESTEPS = 200_000
# ---------------------------------------------------------------------------

RUN_NAME = f"{ALGO.lower()}_{REWARD_MODE}"
RESULTS_DIR = "results"


def plot_training(monitor_csv, out_path):
    """Reward per episode + moving average — the plot for the documentation."""
    data = np.genfromtxt(monitor_csv, delimiter=",", skip_header=2, usecols=(0,))
    episodes = np.arange(1, len(data) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, data, alpha=0.3, label="Reward pro Episode")
    window = max(1, len(data) // 50)
    if len(data) >= window:
        moving = np.convolve(data, np.ones(window) / window, mode="valid")
        plt.plot(episodes[window - 1:], moving, linewidth=2,
                 label=f"Gleitender Durchschnitt ({window} Episoden)")
    plt.xlabel("Episode")
    plt.ylabel("Gesamt-Reward")
    plt.title(f"Trainingsverlauf: {ALGO} Line Follower (Reward: {REWARD_MODE})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Trainingsplot gespeichert: {out_path}")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = Monitor(LineFollowerEnv(reward_mode=REWARD_MODE),
                  filename=os.path.join(RESULTS_DIR, RUN_NAME))

    if ALGO == "PPO":
        # SB3 defaults are solid here (lr=3e-4, gamma=0.99, n_steps=2048, batch=64)
        model = PPO("MlpPolicy", env, verbose=1, seed=0)
    elif ALGO == "DQN":
        # exploration_fraction stretched: default explores only the first 2% of
        # training, too little to ever reach the far side of the track
        model = DQN("MlpPolicy", env, verbose=1, seed=0,
                    exploration_fraction=0.3, learning_starts=1000)
    else:
        raise ValueError(f"unknown ALGO: {ALGO}")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(f"models/{RUN_NAME}")
    print(f"Modell gespeichert: models/{RUN_NAME}.zip")

    env.close()
    plot_training(os.path.join(RESULTS_DIR, f"{RUN_NAME}.monitor.csv"),
                  os.path.join(RESULTS_DIR, f"{RUN_NAME}_training_plot.png"))


if __name__ == "__main__":
    main()
