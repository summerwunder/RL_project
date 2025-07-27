import gymnasium as gym
from agent import *


def train(env, agent, episodes = 300):
    rewards = []
    for episode in range(episodes):
        done = False
        reward_episode = 0
        state, _ = env.reset()
        while not done:
            action , log_prob = agent.select_action(state)
            next_state, reward, term, trun, _ = env.step(action)
            agent.learn(log_prob, state,  next_state, reward, term)
            done = term or trun
            state = next_state
            reward_episode += reward
        rewards.append(reward_episode)
        print(f"episode:{episode} ep_r:{reward_episode}")
        if rewards[-40:] > 400:
            break
    env.close()

def play(env,agent,max_steps=500):
    s,_ = env.reset()
    for _ in range(max_steps):
        env.render()
        a = agent.select_action(s)
        s,_,term,trunc,_ = env.step(a)
        if term or trunc:
            break
    env.close()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    agent = Agent(
        env.observation_space.shape[0],
        env.action_space.n
    )
    train(env, agent)
    play(env, agent)