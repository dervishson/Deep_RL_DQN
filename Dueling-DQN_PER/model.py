import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):   
    def __init__(self,state_size,action_size):
        super(DuelingDQN, self).__init__()
        self.input_dim=state_size
        self.output_dim=action_size
        
        self.feature_layer=nn.Sequential(
            nn.Linear(state_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU())
        
        self.value_stream=nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 1))
        
        self.advantage_stream=nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_size))
    
    def forward(self,state):
        features=self.feature_layer(state)
        values=self.value_stream(features)
        advantages=self.advantage_stream(features)
        qvals=values+(advantages-advantages.mean())
        return qvals