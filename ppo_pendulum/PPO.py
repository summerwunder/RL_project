from model import *
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import typing
from replay_buffer import ReplayBuffer
class PPO(object):
    """
    PPO算法, 采用截断方式
    """

    def __init__(self ,state_dim, action_dim, hidden_layers_dim, lr_a, lr_c, gamma, capacity, ppo_kwargs:typing.Dict, device):
        self.actor = Actor(state_dim, action_dim, hidden_layers_dim).to(device)
        self.critic = Critic(state_dim, hidden_layers_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.buffer = ReplayBuffer(capacity)

        self.gamma = gamma
        self.lamb = ppo_kwargs['lambda']
        self.eps = ppo_kwargs['eps']
        self.ppo_epochs = ppo_kwargs['ppo_epochs']
        self.device = device


    def select_action(self, state):
        state = torch.FloatTensor(np.array(state)).to(self.device)
        mu , std = self.actor(state)
        action_list = torch.distributions.Normal(mu, std)
        action = action_list.sample()
        return [action.item()]        # 期望输入是数组action[0]提取

    def compute_advantage(self,td_error):
        td_error = td_error.detach().numpy()
        adv_list = []
        adv = 0
        for delta in td_error[::-1]:
            adv = self.gamma * self.lamb * adv +delta
            adv_list.append(adv)
        adv_list.reverse()
        return torch.FloatTensor(np.array(adv_list, dtype=np.float32))

    #   reward = -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
    def learn(self):
        s,a,r,s_,d = self.buffer.sample()

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).view(-1,1).to(self.device)
        r = torch.FloatTensor(r).view(-1,1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).view(-1,1).to(self.device)
        r = (r + 8.0) / 8.0  # 奖励归一化

        td_target = r + self.gamma * (1-d) * self.critic(s_)
        td_error = td_target - self.critic(s)
        advantage = self.compute_advantage(td_error.cpu()).to(self.device)  #GAE

        mu , std = self.actor(s)
        action_dist = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dist.log_prob(a)

        # 对同一批样本多次梯度更新
        # PPO 是一种 on-policy 算法，意味着每次策略更新后，旧的样本（基于旧策略采集的数据）会变得“过时”（因为新策略可能生成不同的动作分布）。
        # 为了最大化利用这些样本，PPO 在每次更新中对同一批数据进行多次优化（通常 ppo_epochs 设置为 3 到 10 次），以在策略发生显著变化之前充分学习。
        for _ in range(self.ppo_epochs):
            mu , std = self.actor(s)
            action_dist = torch.distributions.Normal(mu, std)
            log_probs = action_dist.log_prob(a)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio , 1-self.eps, 1+self.eps) * advantage

            actor_loss = -torch.mean(torch.min(surr1,surr2)).float()
            critic_loss = torch.mean(F.mse_loss(self.critic(s).float(), td_target.detach())).float()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()







