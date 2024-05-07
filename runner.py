from Agent import RLAgent

from env import CustomEnv

import gymnasium as gym

env = CustomEnv(
    demand_type = 'Poisson',
)

agent = RLAgent(env = env)

agent.train(num_episodes = 3000)