from Agent import RLAgent

from env import CustomEnv

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt 

env = CustomEnv(
    demand_type = 'Poisson',
    max_days = 100,
    max_price = 100,
    num_seats = 100
)

agent = RLAgent(env = env)

agent.train(num_episodes = 1000)

avg = []
for i in range(15):
    avg.append(agent.simulateEpisode())
    print(f"Episode #{i + 1}: Reward : {avg[-1]}")
print("--------------")
print(f"Average Reward: {sum(avg)/len(avg)}")
print("---------------") 


action = np.zeros(shape = (env.num_seats + 1, env.max_days + 1))
for day in range(env.num_seats + 1):
    for t in range(env.max_days + 1):
        print(day, t, end = '\r')
        action[day, t] = agent.select_action(np.array([day,t])) 


# action = np.flip(action, axis=1)
# Print the actions array
plt.imshow(action, cmap="viridis")
plt.xlabel("seats_left")
plt.ylabel("Days")
plt.title("Policy")
plt.colorbar()
plt.savefig("policy.png")
plt.show()

