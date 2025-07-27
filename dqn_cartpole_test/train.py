import torch
import gymnasium as gym
from agent import Agent
import numpy as np
def train(env,agent,max_episodes=500):
    rewards = []
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        agent.eps = max(agent.eps*agent.eps_decay,agent.eps_min)

        if np.mean(rewards[-30:]) > 400:
            print(f"Solved in {episode} episodes!")
            break
        while True:
            action = agent.select_action(state)
            next_state, reward, term, trun , _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, term)
            agent.learn()
            state = next_state
            episode_reward += reward
            if term or trun:
                break
        print(f"episode:{episode} , reward:{episode_reward:.3f}")
        rewards.append(episode_reward)
        if episode % 20:
            agent.sync()
    env.close()

def play(env,agent,max_steps=500):
    s,_ = env.reset()
    for _ in range(max_steps):
        env.render()
        a = agent.predict(s)
        s,_,term,trunc,_ = env.step(a)
        if term or trunc:
            break
    env.close()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent =Agent(
       env.observation_space.shape[0],
       env.action_space.n
    )
    train(env,agent)
    env = gym.make('CartPole-v1',render_mode ="human")
    play(env,agent)