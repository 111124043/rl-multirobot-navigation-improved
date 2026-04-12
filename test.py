import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import BotEnv

env = BotEnv(num_bots=3)
model = PPO.load("swarm_model")

runs = 20
success_count = 0
collision_runs = 0

plt.ion()

for run in range(runs):
    obs = env.reset()
    trajectories = [[] for _ in range(env.num_bots)]
    collision_flag = False

    fig, ax = plt.subplots()

    for step in range(400):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        for i in range(env.num_bots):
            trajectories[i].append(env.positions[i].copy())

        # collision check
        for i in range(env.num_bots):
            for j in range(env.num_bots):
                if i != j:
                    if np.linalg.norm(env.positions[i] - env.positions[j]) < 0.3:
                        collision_flag = True

        ax.clear()

        pos = env.positions

        # bots (fine points)
        ax.scatter(pos[:, 0], pos[:, 1], s=20, label="Bots")

        # humans (thicker)
        ax.scatter(env.humans[:, 0], env.humans[:, 1], s=100, c='red', label="Humans")

        # trajectories
        for traj in trajectories:
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1)

        circle = plt.Circle(env.goal, env.goal_radius, fill=False)
        ax.add_patch(circle)
        ax.scatter(env.goal[0], env.goal[1], marker='X', s=150)

        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title(f"Run {run+1}")
        ax.legend()

        plt.pause(0.01)

        if done:
            break

    if all(env.reached):
        success_count += 1
    if collision_flag:
        collision_runs += 1

    plt.close(fig)

print("\nRESULTS")
print("Success Rate:", success_count / runs)
print("Collision Rate:", collision_runs / runs)