import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN
from replay_buffer import ReplayBuffer

class Agent(object):
    def __init__(self, config ):
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.eps = config.eps_max
        self.gamma = config.gamma
        self.batch_size = config.batch_size

        self.buffer = ReplayBuffer(config.capacity)

        self.policy_net = DQN(self.state_dim, self.action_dim, config.hidden_layer)
        self.target_net = DQN(self.state_dim, self.action_dim, config.hidden_layer)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr = config.lr)

    def predict(self, state) -> int:
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        with torch.no_grad():
            q_action = self.policy_net(state)
            action = q_action.max(1).indices.item()
            return action

    def select_action(self, state) -> int:
        if np.random.uniform(0,1) < self.eps:
            # explore
            action = np.random.choice(self.action_dim)
        else:
            action = self.predict(state)
        return action

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        s ,a ,r, n_s ,d = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).view(-1,1)
        r = torch.FloatTensor(r).view(-1,1)
        n_s = torch.FloatTensor(n_s)
        d = torch.FloatTensor(d).view(-1,1)

        td_target = r + self.gamma* (1-d) * self.target_net(n_s).detach().max(1).values.unsqueeze(1)
        q = self.policy_net(s).gather(1, a)
        loss = nn.MSELoss()(q,td_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self,path = "./policy.pth"):
        torch.save(self.policy_net.state_dict(),path)

    def load(self, path = './policy.pth'):
        self.policy_net.load_state_dict(torch.load(path,weights_only = True))
        self.target_net.load_state_dict(torch.load(path,weights_only = True))
