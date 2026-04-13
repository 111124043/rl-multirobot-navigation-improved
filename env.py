import gym
from gym import spaces
import numpy as np

class BotEnv(gym.Env):
    def __init__(self, num_bots=3):
        super().__init__()

        self.num_bots = num_bots
        self.num_humans = 2
        self.goal_radius = 1.0
        self.world_size = 6
        self.max_steps = 300

        self.action_space = spaces.Box(low=-1, high=1, shape=(2 * self.num_bots,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 * self.num_bots,), dtype=np.float32)

    def reset(self):
        self.goal = np.random.uniform(-3, 3, 2)

        self.positions = []
        for _ in range(self.num_bots):
            while True:
                pos = np.random.uniform(-5, 5, 2)
                if np.linalg.norm(pos - self.goal) > 2.0:
                    self.positions.append(pos)
                    break
        self.positions = np.array(self.positions)

        self.humans = np.random.uniform(-4, 4, (self.num_humans, 2))

        self.velocities = np.zeros((self.num_bots, 2))
        self.reached = [False] * self.num_bots
        self.steps = 0

        self.prev_distances = [
            np.linalg.norm(self.positions[i] - self.goal)
            for i in range(self.num_bots)
        ]

        return self._get_state()

    def step(self, action):
        action = action.reshape(self.num_bots, 2)

        self.humans += np.random.uniform(-0.03, 0.03, self.humans.shape)

        for i in range(self.num_bots):
            if not self.reached[i]:
                v = action[i]
                repulsion = np.array([0.0, 0.0])

                for j in range(self.num_bots):
                    if i != j:
                        diff = self.positions[i] - self.positions[j]
                        d = np.linalg.norm(diff)

                        if d < 1.0:
                            direction = diff / (d + 1e-5)

                            # stronger separation
                            repulsion += 1.8 * direction * (1.0 - d)

                            # sideways motion for going around
                            perp = np.array([-direction[1], direction[0]])
                            repulsion += 1.0 * perp * (1.0 - d)

                # human avoidance 
                for h in self.humans:
                    diff = self.positions[i] - h
                    d = np.linalg.norm(diff)
                    if d < 1.8:
                        repulsion += (diff / (d + 1e-5)) * (1.8 - d)

                self.velocities[i] = v + 0.5 * repulsion

                # slow near bots
                for j in range(self.num_bots):
                    if i != j:
                        d = np.linalg.norm(self.positions[i] - self.positions[j])
                        if d < 0.7:
                            self.velocities[i] *= 0.5

                self.velocities[i] *= 0.9
                self.positions[i] += self.velocities[i] * 0.07

            else:
                self.velocities[i] = np.array([0.0, 0.0])

        self.steps += 1

        rewards = 0
        distances = []

        for i in range(self.num_bots):
            curr_dist = np.linalg.norm(self.positions[i] - self.goal)
            prev_dist = self.prev_distances[i]

            distances.append(curr_dist)

            rewards += 5 * (prev_dist - curr_dist)

            # reduced distance importance
            if curr_dist > 2.0:
                rewards += -0.02 * curr_dist

            self.prev_distances[i] = curr_dist

            # freeze inside goal
            if curr_dist < self.goal_radius:
                if not self.reached[i]:
                    rewards += 40

                self.reached[i] = True
                self.velocities[i] = np.array([0.0, 0.0])
                continue

            # collision shaping
            for j in range(self.num_bots):
                if i != j:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])

                    if d < 0.3:
                        rewards -= 120
                    elif d < 0.6:
                        rewards -= 40 * (0.6 - d)

            # human safety
            for h in self.humans:
                d_h = np.linalg.norm(self.positions[i] - h)
                if d_h < 1.8:
                    rewards -= 40 * (1.8 - d_h)

            # discourage hovering near goal
            if curr_dist < 1.5:
                rewards -= 2

        rewards -= 0.3 * max(distances)

        done = False

        if all(self.reached):
            rewards += 100
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), rewards, done, {}

    def _get_state(self):
        state = []

        for i in range(self.num_bots):
            direction = self.goal - self.positions[i]
            dist = np.linalg.norm(direction)

            if dist > self.goal_radius:
                dx, dy = direction
            else:
                dx, dy = direction * 0.3

            vx, vy = self.velocities[i]

            min_bot = min(
                [np.linalg.norm(self.positions[i] - self.positions[j]) 
                 for j in range(self.num_bots) if j != i],
                default=10
            )

            min_human = min(
                [np.linalg.norm(self.positions[i] - h) for h in self.humans],
                default=10
            )

            state.extend([dx, dy, vx, vy, min_bot, min_human])

        return np.array(state, dtype=np.float32)
