import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import Actor, Critic      # 你的两个网络

class Agent:
    def __init__(self, s_dim, a_dim,
                 lr_a=1e-3, lr_c=1e-3, gamma=0.99, action_scale=2.0):
        self.actor  = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim)
        self.opt_a  = optim.Adam(self.actor.parameters(),  lr=lr_a)
        self.opt_c  = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma  = gamma
        self.scale  = action_scale      # Pendulum 需要 [-2,2]

    # ---------- 采样 ----------
    def select_action(self, state: np.ndarray) -> float:
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s) * self.scale   # (-scale, scale)
        return a.item()

    # ---------- 单步学习 ----------
    def learn(self, state, action, reward, next_state, done):
        s  = torch.tensor(state,  dtype=torch.float32).unsqueeze(0)
        a  = torch.tensor([[action]], dtype=torch.float32)
        r  = torch.tensor([reward], dtype=torch.float32)
        s2 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # ---- Critic (TD) ----
        q = self.critic(s, a)
        with torch.no_grad():
            a2 = self.actor(s2) * self.scale
            q_next = self.critic(s2, a2)
            target = r + self.gamma * q_next * (1 - done)
        loss_c = F.mse_loss(q, target)
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        # ---- Actor (Deterministic Policy Gradient) ----
        a_pred = self.actor(s) * self.scale
        loss_a = -self.critic(s, a_pred).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()