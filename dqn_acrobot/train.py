import gymnasium as gym
from agent import Agent
from tqdm import tqdm
import numpy as np
import torch
from loguru import logger
from icecream import ic
class Config:
    max_episodes = 800
    sync_episode = 20
    action_dim = None
    state_dim = None
    hidden_layer = [64,64]
    eps_max = 0.9
    eps_min = 0.01
    eps_decay = 0.99
    gamma = 0.95
    batch_size = 128
    capacity = 10000
    test_max_episodes = 30
    test_max_steps = 500
    lr = 2e-3
    def __init__(self,env):
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.n
        except Exception as e:
            self.action_dim = env.action_space.shape[0]

def train(env, agent, config):
    rewards = []
    best_reward = -np.inf
    now_reward = 0
    tq_bar = tqdm(range(config.max_episodes))
    for episode in tq_bar:
        episode_reward = 0
        tq_bar.set_description(f"episode: {episode + 1}/{config.max_episodes}")
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state,reward,done,_,_ = env.step(action)
            episode_reward += reward
            agent.buffer.push(state,action,reward,next_state,done)
            state = next_state
            agent.learn()

        agent.eps = max(agent.eps * config.eps_decay, config.eps_min)
        rewards.append(episode_reward)
        now_reward = np.mean(rewards[-10:]) if len(rewards) > 10 else np.mean(rewards)
        if now_reward > best_reward:
            best_reward = now_reward
            agent.save()
        tq_bar.set_postfix({"reward of the last 10 episodes": f"{now_reward:.2f}",
                            "the best reward" : f"{best_reward:.2f}"})
        if episode % config.sync_episode == 0:
            agent.sync_target()
    env.close()

def test(env, agent, config):
    '''
    验证模型是否完成挑战。
    :params: targs 用于运行模型的环境和关键参数
    :params: epoch 正在验证的回合数
    '''
    finish_list = []
    for episode in range(config.test_max_episodes):
        state, _ = env.reset()
        step = 0
        for _ in range(config.test_max_steps):
            # env.render()
            action = agent.predict(state)
            next_state,reward,done,_,_ = env.step(action)
            state = next_state
            step += 1
            if done:
                break
        if step > config.test_max_steps:
            finish_list.append(0)
            logger.warning(f"[第 {episode + 1} 回合] 智能体未能在 {config.test_max_steps} 步内完成 Acrobot 挑战")
        else:
            finish_list.append(1)
            logger.info(f"[第 {episode} 回合] 智能体在第 {step} 步完成 Acrobot 挑战")
    victory_rate = np.sum(finish_list)/config.test_max_episodes
    ic(victory_rate)


if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    config = Config(env)
    agent = Agent(config)
    agent.load()
    test(env, agent, config)
    # train(env, agent,config)

