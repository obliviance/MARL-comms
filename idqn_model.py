import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

# small feedforward neural network
# input size: obs_dim (length of the listener's observation vector)
# output size: n_actions (number of discrete actions)
# given a state it outputs a vector of Q-values and takes the argmax to chhose the best action
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        return self.net(x)

# FIFO memory that stores transitions
# transition entry = (state, action, reward, next_state, done)
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, s, a, r, s2, d):
        self.buffer.append((s,a,r,s2, d))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d
    
    def __len__(self):
        return len(self.buffer)
    
 
