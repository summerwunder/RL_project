import torch.nn as nn
import torch.nn.functional
from django.db.models import F


class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,64)
        self.fc2 = nn.Linear(64,action_dim)

    def forward(self,x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)

        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim = 1)

class Critic(nn.Module):
    def __init__(self,state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,64)
        self.fc2 = nn.Linear(64,1)

    def forward(self,x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32)

        x = F.relu(self.fc1(x))
        return self.fc2(x)

