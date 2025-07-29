import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,32)
        self.ln1 = nn.LayerNorm(32)
        #relu
        self.fc2 = nn.Linear(32,32)
        self.ln2 = nn.LayerNorm(32)
        #relu
        self.fc3 = nn.Linear(32,action_dim)

    def forward(self,state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return F.tanh(x)

class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim,32)
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32,32)
        self.ln2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32,1)

    def forward(self,state,action):
        x = self.fc1(torch.cat([state,action],dim = 1))
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    



