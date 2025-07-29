import gymnasium as gym
import torch
from agent import Agent
import numpy as np
def train(env,agent,max_episode = 500):
    rewards = []
    for episode in range(max_episode):
        total_reward = 0
        agent.noise.reset()
        state , _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state,add_noise=True)
            next_state, reward, term, trun, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, term)
            agent.learn()
            done = term or trun
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
        if episode % 5 == 0 :
            print("Episode: {}, Total reward: {}".format(episode, total_reward))
            agent.save()
        # if np.mean(rewards[-20:])> 70:
        #     print("Episode: {}, Total reward: {}".format(episode, total_reward))
        #     print("train done")
        #     agent.save()
        #     break
    env.close()

def play(env,agent):
    test_episode = 10
    agent.load()
    for episode in range(test_episode):
        state , _ = env.reset()
        ep_reward = 0
        for i in range(1200):
            action = agent.select_action(state,add_noise=True)
            next_state, reward, term, trun, _ = env.step(action)
            ep_reward += reward
            state = next_state
            env.render()
            if term or trun:
                print("Episode: {}, Total reward: {}".format(i, ep_reward))
                break
    env.close()

if __name__ == '__main__':
  #   env = gym.make('MountainCarContinuous-v0')
    env = gym.make('MountainCarContinuous-v0',render_mode = "human")
    state_dim = env.observation_space.shape[0]  # 2
    action_dim = env.action_space.shape[0]   # 1
    # action_high = env.action_space.high[0]  # 动作上限 (1.0)
    # action_low = env.action_space.low[0]  # 动作下限 (-1.0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(state_dim, action_dim, device)
   #  train(env,agent)


    play(env,agent)



