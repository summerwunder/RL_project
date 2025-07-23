from gridworld import CliffWalkingWapper
from agent_mine import Agent
import gym

def run_one_episode(env,agent,is_render=False):
    obs = env.reset()
    total_step = 0
    total_reward = 0

    while True:
        action = agent.sample(obs)
        next_obs,reward,done,info = env.step(action)
        agent.learn(obs,action,reward,next_obs,done)
        obs = next_obs
        total_step += 1
        total_reward += reward
        if is_render:
            env.render()
        if done:
            break
    return total_reward, total_step

def test_episode(env,agent):
    obs = env.reset()
    total_reward = 0
    total_step = 0
    while True:
        action = agent.predict(obs)
        next_obs,reward,done,_ = env.step(action)
        total_reward += reward
        total_step += 1
        obs = next_obs
        env.render()
        if done:
            print('test reward = %.1f' % (total_reward))
            break

if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    env = CliffWalkingWapper(env)


    agent = Agent(env.observation_space.n,
                  env.action_space.n)
    is_render = False
    for episode in range(500):
        ep_reward, ep_steps = run_one_episode(env, agent, is_render)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps,
                                                          ep_reward))

        # 每隔20个episode渲染一下看看效果
        if episode % 20 == 0:
            is_render = True
        else:
            is_render = False
        # 训练结束，查看算法效果
    test_episode(env, agent)