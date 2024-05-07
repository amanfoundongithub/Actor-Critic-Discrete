import torch 
import torch.nn as nn




class ActorCritic(nn.Module):

    def __init__(self,
                 state_dim : int, 
                num_actions : int,
                hid_dim = 128):
        
        super().__init__() 

        self.__actor_network = nn.Sequential(
            nn.Linear(state_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, num_actions),
            nn.Softmax(dim = -1),
        )

        self.__critic_network = nn.Sequential(
            nn.Linear(state_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )
    
    def forward(self, x):
        action_probs = self.__actor_network(x) 

        critic_val   = self.__critic_network(x)

        return critic_val, action_probs
        

