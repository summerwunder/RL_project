import torch.nn as nn
import torch.nn.functional as F
import typing
class Model(nn.Module):
    def __init__(self,state_dim:int,action_dim:int,hidden_layer:typing.List[int]):
        super().__init__()
        self.features = nn.ModuleList()
        for idx,layer in enumerate(hidden_layer):
            self.features.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_layer[idx - 1] if idx else state_dim, layer),
                'linear_action':nn.ReLU()
            }))
        self.out = nn.Linear(hidden_layer[-1] , action_dim)

    def forward(self,x):
        for layer in self.features:
            x = layer['linear'](layer['linear_action'](x))
        return F.softmax(self.out(x),dim=1)