import torch.nn as nn
import torch.nn.functional as F
import typing

# 输出连续动作的概率分布参数,适用于连续动作空间（输出服从正态分布）
class Actor(nn.Module):
    def __init__(self,state_dim:int, action_dim:int, hidden_layer_list:typing.List[int]):
        super().__init__()
        self.features = nn.ModuleList()
        for idx,h in enumerate(hidden_layer_list):
            self.features.append(nn.ModuleDict({
                'linear':nn.Linear(hidden_layer_list[idx - 1] if idx else state_dim, h),
                'linear_action':nn.ReLU(inplace=True)
            }))
        self.fc_mu = nn.Linear(hidden_layer_list[-1], action_dim)
        self.fc_std = nn.Linear(hidden_layer_list[-1], action_dim)

    def forward(self,x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))
        mean_ = 2 * F.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))   # 标准差>0
        return mean_, std


class Critic(nn.Module):
    def __init__(self,state_dim:int,  hidden_layer_list:typing.List[int]):
        super().__init__()
        self.features = nn.ModuleList()
        for idx,h in enumerate(hidden_layer_list):
            self.features.append(nn.ModuleDict({
                'linear':nn.Linear(hidden_layer_list[idx - 1] if idx else state_dim, h),
                'linear_action':nn.ReLU(inplace=True)
            }))
        self.head = nn.Linear(hidden_layer_list[-1], 1)

    def forward(self,x):
        for layer in self.features:
            x = layer['linear_action'](layer['linear'](x))
        return self.head(x)