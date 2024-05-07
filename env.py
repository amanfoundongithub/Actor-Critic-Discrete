import numpy as np
from scipy.special import factorial
import gymnasium as gym
from gymnasium import spaces


# Assumption, people will buy independent of number of seats available
# State is (day, seats_left)
# action is continuous in range [min_price, max_price]
class CustomEnv(gym.Env):
    def __init__(
            self, 
            demand_type='Poisson',
            min_price=0, 
            max_price=10,
            num_seats=10,
            max_days=10) -> None:
        
        super(CustomEnv, self).__init__()

        # np.random.seed(rng_seed)
        # self.rng_seed = rng_seed
        self.demand_type = demand_type
        self.min_price = min_price
        self.max_price = max_price
        self.num_seats = num_seats
        self.max_days = max_days

       
        # self.action_space = spaces.Box(low=min_price, high=max_price, shape=(1,), dtype=np.float32)  # Continuous action space from A to B
        self.action_space = spaces.Discrete(n = max_price - min_price + 1, start = min_price)
        self.observation_space = spaces.MultiDiscrete([max_days+1, num_seats+1])
        # 0: day 1: seats_left

        # Initialize state
        self.state = np.zeros(2)
        self.state[1] = num_seats # initially all seats are left
        # Set maximum number of steps
        self.max_steps = max_days - 1
        self.current_step = 0


        pass

    def poisson_demand(self, price):
        lam = self.num_seats * ((self.max_price - price + 1) 
                    / (self.max_price - self.min_price + 1)) 
        return np.random.poisson(lam=lam)
    
    def bernoulli_demand(self, price):
        p = ((self.max_price - price + 1)
                    / (self.max_price - self.min_price + 1))
        return np.random.binomial(n=1, p=p)
    

    def demand(self, price):
        if self.demand_type == 'Poisson':
            return self.poisson_demand(price)
        elif self.demand_type == 'Bernoulli':
            return self.bernoulli_demand(price)
        
    def poisson_demand_pmf(self, s_new, s_old, price):
        lam = self.num_seats * ((self.max_price - price + 1) 
                    / (self.max_price - self.min_price + 1))
        x = s_old - s_new
        if s_new == 0:
            p = 1
            for i in range(s_old):
                p -= (lam ** i) * np.exp(-lam) / factorial(i)
            return p

        # calculate poisson PMF of X=x with lambda=lam without using scipy
        return (lam ** x) * np.exp(-lam) / factorial(x)

    def bernoulli_demand_pmf(self, s_new, s_old, price):
        if s_new < s_old - 1:
            return 0

        p = ((self.max_price - price + 1)
             / (self.max_price - self.min_price + 1))
        if s_new == s_old - 1:
            return p
        return 1 - p
    
    # P(s'/s, a)
    def pmf(self, s_new, s_old, price):

        # because they can't cancel seats
        if s_old < s_new: 
            return 0
        
        if self.demand_type == 'Poisson':
            return self.poisson_demand_pmf(s_new, s_old, price)
        elif self.demand_type == 'Bernoulli':
            return self.bernoulli_demand_pmf(s_new, s_old, price)
        
    def reward(self, s_new, s_old, price):

        if s_old < s_new: 
            return 0

        return (s_old - s_new) * price

    def reset(
            self) -> None:

        self.state = np.zeros(2, dtype=int)
        self.state[1] = self.num_seats
        self.current_step = 0
        
        return self.state, {}
        pass

    def step(self, action):
        # clipping anything outside, recommend keeping min = 0, max = inf, 
        # and having actions in a reasonable range
        action = np.clip(action, self.min_price, self.max_price)     

        demand = self.demand(action)
#         print(demand)
        reward = min(self.state[1], demand) * action
        self.state[0] += 1
        self.state[1] = max(0, self.state[1] - demand)

        terminated = self.state[1] == 0 or self.state[0] == self.max_days

        return self.state, reward, terminated, terminated, {}

    def render(self, mode='human'):
        # Rendering not implemented
        pass


    