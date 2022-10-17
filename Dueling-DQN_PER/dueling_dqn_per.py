import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import random
import numpy as np
from itertools import count
from collections import deque, namedtuple

from per import ProportionalPrioritizedMemory
from model import DuelingDQN
import matplotlib.pyplot as plt


"""
# Exploration-related Parameters
EPSILON_START = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY = 1000

# Memory Related Parameters
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

# QLearning Parameters
GAMMA = 0.99

# Prioritized Replay Paramteres
ALPHA = 0.6
BETA = 0.4

# Network Parameters
HIDDEN_UNITS = 128
LEARNING_RATE = 0.0001
NETWORK_UPDATE_FREQUENCY = 1000
"""
#Namedtuple for experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
#Device assignment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


class Agent():
    Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    """Interacts with and learns from the environment."""
    def __init__(self,state_size,action_size,seed,memory_capacity=BUFFER_SIZE,batch_size=BATCH_SIZE):
        """Initialize an Agent object.

            Params
            ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                seed (int): random seed
        """
        self.step=0
        self.state_size=state_size
        self.action_size=action_size
        self.batch_size=batch_size
        self.seed=random.seed(seed)
        self.memory_capacity=memory_capacity
        self.memory=ProportionalPrioritizedMemory(memory_capacity)
        
        
        #Q-Network
        self.qnetwork_local=DuelingDQN(state_size,action_size).to(device)
        self.qnetwork_target=DuelingDQN(state_size,action_size).to(device)
        self.optimizer=optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
    def one_step(self,state,action,reward,next_state,done):
        #Push and save experience in replay memory
        self.memory.push(Experience(state,action,reward,next_state,done))
        
        
        #Learn every UPDATE_EVERY time steps
        self.step = (self.step+1) % UPDATE_EVERY


        if self.step==0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory)>self.batch_size:
                idxs,wghs,exps=self.memory.sample(self.batch_size)
                self.learn(idxs,wghs,exps,GAMMA)

        
    def act(self,state,eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state=torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values=self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        #Epsilon-greedy action selection
        if random.random()>eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self,idx,wgh,exp,GAMMA):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            gamma (float): discount factor
        """
        batch=Experience(*zip(*exp))
        weights=torch.tensor(wgh).to(device)
        
        done=torch.FloatTensor(batch.done).to(device)
        action=torch.LongTensor(batch.action).to(device)
        reward=torch.FloatTensor(batch.reward).to(device)
        state=torch.FloatTensor(np.float32(batch.state)).to(device)
        next_state=torch.FloatTensor(np.float32(batch.next_state)).to(device)
        
        q_values=self.qnetwork_local(state).gather(1,action.unsqueeze(1)).squeeze(1)
        next_q_values=self.qnetwork_target(next_state).max(1)[0]
        expected_q_values=reward+GAMMA*next_q_values*(1-done)
        
        delta=q_values-expected_q_values.detach()
        self.memory.update(deltas=delta.tolist(),indexes=idx)
        
        loss=(delta.pow(2)*weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
        
        
                