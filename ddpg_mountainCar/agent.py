import os
from os.path import exists
import torch
import torch.nn.functional as F
from model import *
import torch.optim as optim
from replay_buffer import *
from noise import *
class Agent(object):
     def __init__(self, state_size, action_size, device,batch_size = 64, capacity = 10000,gamma = 0.99,tau = 1e-3,lr_a = 1e-3,lr_c = 2e-3, dir= "./net/"):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.capacity = capacity
        self.device = device
        self.gamma = gamma
        self.dir = dir
        self.tau = tau
        # 定义网络
        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size, action_size).to(self.device)
        self.target_actor = Actor(state_size, action_size).to(self.device)
        self.target_critic = Critic(state_size, action_size).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c)
        # 定义经验回放
        self.replay_buffer = ReplayBuffer(capacity)

        self.noise = Noise(action_size, mu = 0,theta = 0.1 , sigma = 0.2)

     def select_action(self, state ,add_noise=True):
         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
         # action = self.actor(state).flatten().item()   # 适合单个维度
         with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
         if add_noise:
            action += self.noise.sample()
         return np.clip(action,-1,1)

     def learn(self):
         if len(self.replay_buffer) < self.batch_size:
             return
         state, action, reward, next_state, done = self.replay_buffer.sample(batch_size=self.batch_size)
         state = torch.FloatTensor(state).to(self.device)
         next_state = torch.FloatTensor(next_state).to(self.device)
         action = torch.FloatTensor(action).to(self.device)
         reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
         done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

         target_Q = self.target_critic(next_state,self.target_actor(next_state).detach())   # target_Q。require_grad = False  #不会计算梯度
         # 此处detach阻断目标actor的梯度，目标网络参数的更新应该是软更新而不是梯度直接反向优化
         target_Q = reward + self.gamma * target_Q * (1 - done)
         current_Q = self.critic(state,action)

         critic_loss = F.mse_loss(current_Q, target_Q)
         self.critic_optimizer.zero_grad()
         critic_loss.backward()
         torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
         self.critic_optimizer.step()

         actor_loss = -self.critic(state,self.actor(state)).mean()  #计算actor的损失梯度只需要用到critic的输出值而不要用到他的参数梯度
         self.actor_optimizer.zero_grad()
         actor_loss.backward()
         self.actor_optimizer.step()

         for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

         for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

     def save(self):
         if not exists(self.dir):
             os.makedirs(self.dir)
         torch.save(self.actor.state_dict(), self.dir + 'actor.pth', weights_only=True)
         torch.save(self.critic.state_dict(), self.dir + 'critic.pth', weights_only=True)

     def load(self):
         self.actor.load_state_dict(torch.load(self.dir + 'actor.pth', weights_only=True))
         self.critic.load_state_dict(torch.load(self.dir + 'critic.pth', weights_only=True))
