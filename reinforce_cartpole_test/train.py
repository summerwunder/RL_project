import gymnasium as gym
from icecream import ic
from agent import Agent
import numpy as np
from loguru import logger
class Config:
    state_dim = None
    action_dim = None
    max_episodes = 1200
    max_steps = 1000
    gamma = 0.99
    hidden_layer = [128,64]
    lr = 1e-3

    def __init__(self,env: gym.Env):
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.n
        except IndexError:
            self.action_dim = env.action_space.shape[0]

def train(env, agent, config: Config):
    rewards = []
    for episode in range(config.max_episodes):
        episode_reward = 0
        s, _ = env.reset()
        for _ in range(config.max_steps):
            action , log_prob = agent.sample(s)
            n_s , r , done ,_,_ = env.step(action)
            agent.saved_log_probs.append(log_prob)
            agent.saved_reward.append(r)
            s = n_s
            episode_reward += r
            if done:
                break
        agent.learn()
        rewards.append(episode_reward)
        if (episode + 1) % 25 == 0:
            logger.info(f"Episode: {episode + 1}, Reward: {np.mean(rewards[-15:]):.2f}")
        if  np.mean(rewards[-30:]) >= 450:
            ic(f"Solved! Episode {episode + 1}. Average reward over last 100 episodes: {np.mean(rewards[-30:]):.2f}")
            break
    env.close()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    config = Config(env)
    agent = Agent(config.state_dim, config.action_dim,config.hidden_layer,config.lr,config.gamma)
    train(env, agent, config)
    # ic(config.action_dim)