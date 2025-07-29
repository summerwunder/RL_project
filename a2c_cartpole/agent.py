from model import *
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
class Agent(object):
    def __init__(self,state_dim,action_dim,gamma=0.99,lr_a=1e-3,lr_c=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.actor = Actor(state_dim,action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(),lr=self.lr_a)
        self.critic_optim = optim.Adam(self.critic.parameters(),lr=self.lr_c)

    def select_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0)
        a = self.actor(state)
        a_list = Categorical(a)      #TENSOR LIST
        action = a_list.sample()      #  tensor
        log_prob = a_list.log_prob(action)

        return action.item(),log_prob

    def learn(self,log_prob,state,next_state,reward,done):
        # 训练critic，用于逼近V_pi(s)
        v = self.critic(state)
        next_v = self.critic(next_state)
        td_target = reward + (1-done) * self.gamma * next_v
        loss = F.mse_loss(v,td_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        # 得到error后直接更新actor的参数
        td_error = (td_target - v).detach()  # 必须去除梯度，否则会更新到Critic
        loss_actor = -log_prob * td_error
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
