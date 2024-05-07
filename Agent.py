# from Policy import ActorCritic


# import gymnasium as gym
# import torch 
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np 
# import collections 

# class RLAgent:
#     def __init__(self, 
#                  env : gym.Env,
#                  discount_factor = 0.99):
        
#         self.__gamma = discount_factor
        
#         self.__state_dim = env.observation_space.shape[0]
#         self.__num_action = env.action_space.n

#         self.__env = env 

#         # Network initialization
#         self.__network = ActorCritic(
#             state_dim = self.__state_dim,
#             num_actions = self.__num_action,
#             hid_dim = 64
#         )

#         # Saved actions, 
#         self.__saved_action = collections.namedtuple("SavedAction",['log_prob', 'value'])

#         self.__list_of_actions = [] 
#         self.__list_of_rewards = [] 
    

#     def select_action(self, state):
#         state = torch.from_numpy(state).float()

#         critic_val, action_probs = self.__network(state) 


#         # Categorical distribution
#         categorical_distributor = torch.distributions.Categorical(action_probs)

#         # Now sample action
#         action = categorical_distributor.sample()

#         return action.item() 
    
#     def train(self, num_episodes = 10):
#         # Initialize a trainer 
#         network_trainer = optim.Adam(params = self.__network.parameters(),
#                                        lr = 1e-3)
        
#         # TRAINING LOOP BEGINS HERE 
#         vals = [] 
#         for episode in range(1, num_episodes + 1):
#             # Initialize an episode 
#             state, _ = self.__env.reset() 

#             done = False
#             # Generate an episode 
#             total_reward = 0
#             for _ in range(10000):
#                 action = self.__select_action_for_training(state)
#                 next_state, reward, terminated, truncated, info = self.__env.step(action)
#                 self.__list_of_rewards.append(reward) 

#                 done = terminated or truncated

#                 total_reward += reward

#                 state = next_state
#                 if done: 
#                     break 
            
#             # print(f"Total reward at #{episode} : {total_reward}")
#             vals.append(total_reward)
#             # Training phase begins here 
#             self.__finish_training(trainer = network_trainer)
            
#             print(f"Average of Last 100 episodes: {np.mean(vals[-100:])}")
#         # TRAINING LOOP ENDS HERE 
        
#         pass 

#     # --------------- USER HIDDEN FUNCTIONS ---------------------------------
#     def __select_action_for_training(self, state):
#         state = torch.from_numpy(state).float()

#         critic_val, action_probs = self.__network(state) 

#         # Categorical distribution
#         categorical_distributor = torch.distributions.Categorical(action_probs)

#         # Now sample action
#         action = categorical_distributor.sample() 

#         # Now append it 
#         self.__list_of_actions.append(
#             self.__saved_action(log_prob = categorical_distributor.log_prob(action),
#                                 value = critic_val)
#         )

#         # Return value 
#         return action.item() 

#     def __finish_training(self, trainer : torch.optim.Optimizer):
#         # eps to avoid NaN values 
#         eps = np.finfo(np.float32).eps.item() 

#         R = 0
#         saved_actions = self.__list_of_actions
#         policy_losses = [] 
#         value_losses = [] 
#         returns = [] 


#         for r in self.__list_of_rewards[::-1]:
#             R = r + self.__gamma * R
#             returns.insert(0, R)
        
        
#         returns = torch.tensor(returns)

#         returns = (returns - returns.mean()) / (returns.std(correction = 0) + eps)


#         for (log_prob, value), R in zip(saved_actions, returns):
#             advantage = R - value.item()

#             # calculate actor (policy) loss
#             policy_losses.append(-log_prob * advantage)

#             # calculate critic (value) loss using L1 smooth loss
#             value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

#         # reset gradients
#         trainer.zero_grad()

#         # sum up all the values of policy_losses and value_losses
#         loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

#         # perform backprop
#         loss.backward()
#         trainer.step()

#         self.__list_of_actions.clear()
#         self.__list_of_rewards.clear()


from Policy import ActorCritic
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import matplotlib.pyplot as plt 


class RLAgent:
    def __init__(self,
                 env: gym.Env,
                 discount_factor=0.99,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.__gamma = discount_factor
        self.device = device

        self.__state_dim = env.observation_space.shape[0]
        self.__num_action = env.action_space.n

        self.__env = env

        # Network initialization
        self.__network = ActorCritic(
            state_dim = self.__state_dim,
            num_actions = self.__num_action,
            hid_dim = 128
        ).to(device)  

        # Saved actions,
        self.__saved_action = collections.namedtuple("SavedAction", ['log_prob', 'value'])

        self.__list_of_actions = []
        self.__list_of_rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)  

        critic_val, action_probs = self.__network(state)

        # Categorical distribution
        categorical_distributor = torch.distributions.Categorical(action_probs)

        # Now sample action
        action = categorical_distributor.sample()

        return action.item()

    def train(self, num_episodes=10):
        # Initialize a trainer
        network_trainer = optim.AdamW(params=self.__network.parameters(),
                                     lr=5e-4)

        # TRAINING LOOP BEGINS HERE
        vals = []
        for episode in range(1, num_episodes + 1):
            # Initialize an episode
            state, _ = self.__env.reset()

            done = False
            # Generate an episode
            total_reward = 0
            for _ in range(10000):
                action = self.__select_action_for_training(state)
                next_state, reward, terminated, truncated, info = self.__env.step(action)
                self.__list_of_rewards.append(reward)

                done = terminated or truncated

                total_reward += reward

                state = next_state
                if done:
                    break

            # print(f"Total reward at #{episode} : {total_reward}")
            vals.append(total_reward)
            # Training phase begins here
            self.__finish_training(trainer=network_trainer)
            

            print(f"Average of Last 100 episodes: {round(np.mean(vals[-100:]),2)}", end = '\r') 

            if np.mean(vals[-100:]) > 91:
                break 
        # TRAINING LOOP ENDS HERE

        # RESULT ANALYSIS TIME !!!  
        val_tensor = torch.tensor(vals, dtype = torch.float)

        plt.title(f'Result of Training Average of Last 100 = {round(np.mean(vals[-100:]),2)}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(val_tensor.numpy())

        means = val_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
            
        plt.plot(means.numpy())

        plt.show()

        pass

    # --------------- USER HIDDEN FUNCTIONS ---------------------------------
    def __select_action_for_training(self, state):
        state = torch.from_numpy(state).float().to(self.device)  # Move state to CUDA or CPU based on device

        critic_val, action_probs = self.__network(state)

        # Categorical distribution
        categorical_distributor = torch.distributions.Categorical(action_probs)

        # Now sample action
        action = categorical_distributor.sample()
        
        # Now append it
        self.__list_of_actions.append(
            self.__saved_action(log_prob=categorical_distributor.log_prob(action),
                                value=critic_val)
        )

        # Return value
        return action.item()

    def __finish_training(self, trainer: torch.optim.Optimizer):
        # eps to avoid NaN values
        eps = np.finfo(np.float32).eps.item()

        R = 0
        saved_actions = self.__list_of_actions
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.__list_of_rewards[::-1]:
            R = r + self.__gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)  # Move returns to CUDA or CPU based on device

        returns = (returns - returns.mean()) / (returns.std(correction=0) + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))  # Move tensor to CUDA or CPU based on device

        # reset gradients
        trainer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        trainer.step()

        self.__list_of_actions.clear()
        self.__list_of_rewards.clear()


