import numpy as np
import gymnasium as gym
from agent import Agent    # 新 Agent 已返回 float

env = gym.make('Pendulum-v1')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

agent = Agent(s_dim, a_dim, action_scale=2.0)

for ep in range(500):
    s, _ = env.reset()
    ep_r = 0
    while True:
        a = agent.select_action(s)
        s2, r, done1, done2, _ = env.step(np.array([a], dtype=np.float32))
        done = done1 or done2
        agent.learn(s, a, r, s2, done)
        s = s2
        ep_r += r
        if done:
            break
    print(f'Ep {ep+1}: {ep_r:.2f}')
env.close()



