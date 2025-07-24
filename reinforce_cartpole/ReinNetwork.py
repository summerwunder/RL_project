import torch
import torch.nn as nn
import torch.nn.functional as F

class ReinNetwork(nn.Module):
    def __init__(self,state_dim:int,action_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,256)
        self.fc2 = nn.Linear(256,action_dim)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)