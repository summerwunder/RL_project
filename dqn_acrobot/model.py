import torch.nn as nn
import typing

class DQN(nn.Module):
    def __init__(self,state_dim:int,action_dim:int,hidden_layer:typing.List[int]):
        super().__init__()
        self.features = nn.ModuleList()
        for idx,hidden in enumerate(hidden_layer):
            self.features.append(nn.ModuleDict({
                'linear':nn.Linear(hidden_layer[idx-1] if idx else state_dim,hidden),
                'linear_action':nn.ReLU(inplace=True)
            }))
        self.fc_out = nn.Linear(hidden_layer[-1],action_dim)

    def forward(self,x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))
        return self.fc_out(x)