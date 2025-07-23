import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self,state_n:int,action_n:int,hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_n,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_n)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
