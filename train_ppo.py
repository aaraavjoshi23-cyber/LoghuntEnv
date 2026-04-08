from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import LogHuntEnv

DATA_PATH = "data/CICIDS2017_sample.csv"

# Sanity check
env = LogHuntEnv(DATA_PATH, curriculum="easy")
check_env(env)  # gymnasium compliance check
print("Env passed check.")

# Train PPO
model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=64)
model.learn(total_timesteps=100_000)
model.save("loghunt_ppo")

# Evaluate
obs, _ = env.reset()
total_reward = 0
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(int(action))
    total_reward += reward
    if done:
        print(f"Episode done. Reward: {total_reward:.1f} | Stats: {info}")
        break
