from stable_baselines3 import PPO
from env import BotEnv

env = BotEnv(num_bots=3)

print("Final cooperative navigation training...")

model = PPO.load("swarm_model", env=env)

model.learning_rate = 0.0001

model.learn(total_timesteps=150000, reset_num_timesteps=False)

model.save("swarm_model")