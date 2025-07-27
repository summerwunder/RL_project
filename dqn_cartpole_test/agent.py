from model import *
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import *
import numpy as np
import torch

class Agent(object):
    def __init__(self,state_dim,action_dim,gamma=0.99,lr=1e-3,epsilion=0.9,epsilon_decay=0.98,eps_min=0.01,batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.eps = epsilion
        self.eps_decay = epsilon_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.steps = 0
        # 定义Agent
        self.target_net = Net(state_dim,action_dim)
        self.policy_net = Net(state_dim,action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=lr)
        self.buffer = ReplayMemory(500)

    def predict(self,state):
        state = torch.FloatTensor(state).unsqueeze(0)  #1*4
        with torch.no_grad():
            out = self.policy_net.forward(state)  # 1*2
            return out.max(1).indices.item()

    def select_action(self,state):
        if np.random.uniform(0,1) < self.eps:
            # explore
            action = np.random.choice(self.action_dim)
            return action
        else:
            # exploit
            action = self.predict(state)
        return action

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.buffer.sample(self.batch_size)
        state_tensor = torch.FloatTensor(state_batch)
        next_state_tensor = torch.FloatTensor(next_state_batch)         # 64*5
        action_tensor = torch.LongTensor(action_batch).unsqueeze(1)    # 64*1
        reward_tensor = torch.FloatTensor(reward_batch).unsqueeze(1)
        done_tensor = torch.FloatTensor(done_batch).unsqueeze(1)

        td_target = (1-done_tensor)*self.gamma*self.target_net(next_state_tensor).max(1).values.unsqueeze(1) +reward_tensor
        value = self.policy_net(state_tensor).gather(1,action_tensor)
        loss = F.mse_loss(value, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(),1)
        self.optimizer.step()

        self.steps += 1

    def sync(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())





