import torch.nn as nn
import torch
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,input_dim:int,output_dim:int,hidden_dim = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.tanh(self.fc2(x))

class Critic(nn.Module):
    def __init__(self,s_dim:int,a_dim:int,hidden_dim = 128):
        super().__init__()
        self.fc1 = nn.Linear(s_dim+a_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

