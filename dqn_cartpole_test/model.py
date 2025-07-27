import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x