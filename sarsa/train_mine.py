import gym
import numpy as np
from agent_mine import Agent
from gridworld import CliffWalkingWapper
import time
assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

def run_episode(env,agent,render=False):
    total_step = 0
    total_reward = 0

    obs = env.reset()
    action = agent.sample(obs)

    while True:
        next_obs, reward, done, info=env.step(action)
        next_action = agent.sample(next_obs)
        agent.learn(obs, action, reward, next_obs, next_action,done)

        action = next_action
        obs = next_obs
        total_reward += reward
        total_step += 1
        if render:
            env.render()  #渲染新的一帧图形
        if done:
            break
    return total_reward, total_step

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)

    agent = Agent(env.observation_space.n,
                  env.action_space.n,
                  lr=0.1,
                  gamma=0.9,
                  e_greed=0.1)

    is_render = False
    num_episodes = 500
    for i in range(num_episodes):
        total_reward, total_step = run_episode(env,agent,render=is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (i, total_step,
                                                          total_reward))

        # if i % 40 == 0:
        #     is_render = True
        # else:
        #     is_render = False
    # 训练结束，查看算法效果
    test_episode(env, agent)