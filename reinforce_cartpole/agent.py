import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ReinNetwork import ReinNetwork

class Agent(object):
    def __init__(self, state_size, action_size,lr=1e-3,gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.net = ReinNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def sample(self,state)->int:
        state = torch.FloatTensor(state).unsqueeze(0)     # 1*4
        action_logits = self.net(state)         # 1*2
        probs = F.softmax(action_logits, dim=1)
        m = Categorical(probs)
        action = m.sample()     # tensor(1)
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def store_reward(self,reward:float)->None:
        self.rewards.append(reward)

    def update_policy(self):
        """根据一个episode的数据更新策略网络."""
        R = 0
        policy_loss = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        # 计算损失函数 L = - sum(log_prob * return)
        for log_prob, returns in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * returns)

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        # #stack必然会拼接增加维度，这里只是将tensor向量求和才用到，如果直接用sum是不行的
        # list -->  torch --> num

        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]

