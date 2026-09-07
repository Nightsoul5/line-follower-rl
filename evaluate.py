import os
import numpy as np
from stable_baselines3 import PPO, DQN
from line_follower_env import LineFollowerEnv

EPISODES = 50
SEED = 0

# track shapes never seen in training (training track: radius=200, amp=50, freq=3)
UNSEEN_TRACKS = {
    "sanfter (amp=30, freq=2)": {"radius": 200, "wave_amp": 30, "wave_freq": 2},
    "welliger (amp=45, freq=5)": {"radius": 210, "wave_amp": 45, "wave_freq": 5},
}


def random_policy(obs, rng):
    return int(rng.integers(0, 3))


def rule_based_policy(obs, rng):
    """Steer toward the weighted center of the sensors that see the line."""
    if obs.sum() == 0:
        return 1  # blind: go straight and hope
    offset = sum((i - 2) * v for i, v in enumerate(obs)) / obs.sum()
    if offset < -0.3:
        return 0
    if offset > 0.3:
        return 2
    return 1


def model_policy(model):
    def policy(obs, rng):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy


def evaluate(policy, episodes=EPISODES, seed=SEED, track_kwargs=None):
    env = LineFollowerEnv(random_start=True, reward_mode="progress",
                          track_kwargs=track_kwargs)
    rng = np.random.default_rng(seed)
    rewards, steps, wins, laps = [], [], 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)   # same start sequence for every policy
        total, n = 0.0, 0
        while True:
            obs, r, terminated, truncated, info = env.step(policy(obs, rng))
            total += r
            n += 1
            if terminated or truncated:
                if info.get("result") == "win":
                    wins += 1
                laps.append(info.get("laps", 0))
                break
        rewards.append(total)
        steps.append(n)
    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "success_rate": wins / episodes,
        "avg_laps": float(np.mean(laps)),
    }


def print_table(rows):
    header = f"{'Policy':<22}{'Ø Reward':>12}{'± Std':>10}{'Ø Steps':>10}{'Erfolg':>9}{'Ø Laps':>9}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<22}{m['avg_reward']:>12.1f}{m['std_reward']:>10.1f}"
              f"{m['avg_steps']:>10.1f}{m['success_rate']:>9.0%}{m['avg_laps']:>9.2f}")
    print()


if __name__ == "__main__":
    policies = [("Random", random_policy), ("Rule-Based", rule_based_policy)]
    trained = []
    for algo, cls in [("PPO", PPO), ("DQN", DQN)]:
        path = f"models/{algo.lower()}_progress.zip"
        if os.path.exists(path):
            model = cls.load(path.removesuffix(".zip"))
            policies.append((f"{algo} (trainiert)", model_policy(model)))
            trained.append((algo, model))
        else:
            print(f"[Hinweis] {path} nicht gefunden — {algo} wird übersprungen. "
                  f"Trainieren mit ALGO=\"{algo}\" in train.py.")

    print(f"\n=== Teil 1: Trainingsstrecke, {EPISODES} Episoden, zufällige Starts (Seed {SEED}) ===\n")
    print_table([(name, evaluate(pol)) for name, pol in policies])

    if trained:
        print(f"=== Teil 2: Generalisierung auf unbekannte Strecken ===\n")
        for track_name, kwargs in UNSEEN_TRACKS.items():
            print(f"Strecke: {track_name}")
            rows = [("Rule-Based", evaluate(rule_based_policy, track_kwargs=kwargs))]
            for algo, model in trained:
                rows.append((f"{algo} (trainiert)", evaluate(model_policy(model), track_kwargs=kwargs)))
            print_table(rows)
