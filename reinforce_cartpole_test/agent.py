import torch
from model import Model
import torch.optim as optim
import numpy as np
from icecream import ic
from torch.distributions import Categorical
class Agent(object):
    def __init__(self,state_dim,action_dim,hidden_layer,lr,gamma):
        self.net = Model(state_dim,action_dim,hidden_layer)
        self.optimizer = optim.Adam(self.net.parameters(),lr=lr)
        self.gamma = gamma
        self.saved_log_probs = []
        self.saved_reward = []
        self.q_values = []

    def sample(self,state):
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        action_probs = self.net(state)   # 1*2 tensor
        categorical = Categorical(action_probs)
        action = categorical.sample()
        log_prob = categorical.log_prob(action)
        return action.item() , log_prob

    def learn(self):
        # 根据轨迹计算q_value
        policy_loss = []
        reward_tmp = 0
        for reward in reversed(self.saved_reward):
            reward_tmp = self.gamma * reward_tmp + reward
            self.q_values.append(reward_tmp)
        self.q_values.reverse()
        # 梯度下降
        for log_prob , q_values in zip(self.saved_log_probs, self.q_values):
            policy_loss.append(-log_prob * q_values)
        self.optimizer.zero_grad()
        loss = torch.sum(torch.stack(policy_loss),dim = 0).squeeze()
        # loss = torch.stack(policy_loss).sum()
        # ic(loss.shape)
        loss.backward()
        self.optimizer.step()
        self.saved_log_probs.clear()
        self.saved_reward.clear()
        self.q_values.clear()





