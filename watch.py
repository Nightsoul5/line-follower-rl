import sys
from stable_baselines3 import PPO, DQN
from line_follower_env import LineFollowerEnv


def main():
    algo = sys.argv[1].upper() if len(sys.argv) > 1 else "PPO"
    cls = {"PPO": PPO, "DQN": DQN}[algo]
    model = cls.load(f"models/{algo.lower()}_progress")

    env = LineFollowerEnv(render_mode="human", reward_mode="progress")
    for episode in range(3):
        obs, _ = env.reset()
        total = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                print(f"Episode {episode + 1}: reward={total:.1f}, "
                      f"laps={info.get('laps', 0)}, result={info.get('result')}")
                break
    env.close()


if __name__ == "__main__":
    main()
