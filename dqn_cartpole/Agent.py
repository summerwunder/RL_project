from DQN import *
from ReplayMemory import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class DQNAgent(object):
    def __init__(self,obs_n,act_n,eps=0.9,eps_min=0.01,eps_decay=0.95,gamma=0.99,batch=50,lr=3e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_n = obs_n
        self.act_n = act_n
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.batch = batch
        self.lr = lr
        self.buffer = ReplayMemory(300)
        self.target_net = DQN(obs_n,act_n).to(self.device)
        self.policy_net = DQN(obs_n,act_n).to(self.device)
        self.optim = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def predict(self,obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.policy_net(state)  #  --> 1*2 (tensor)
            return out.max(1).indices.item()  # 1 / 2

    def sample(self,obs):
        if np.random.uniform(0,1) < self.eps:
            # explore
            act = np.random.choice(self.act_n)
        else:
            # exploit
            act = self.predict(obs)  # 选择最优动作
        return act

    def learn(self):
        if len(self.buffer)< self.batch:
            return
        states, actions, rewards, next_states, terminated = self.buffer.sample(self.batch)
        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)    # 64 * 1
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(terminated).unsqueeze(1).to(self.device)      # 64 * 1

        q = self.policy_net(s).gather(1, a)  #dim = 1 -> a固定行,选择某一列   [64,2] --> [64,1]
        with torch.no_grad():
            tgt = r + (1-d)*self.gamma *self.target_net(ns).max(1).values.unsqueeze(1)   # if terminal = 1  TD target = R
        loss = nn.MSELoss()(q,tgt)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optim.step()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())