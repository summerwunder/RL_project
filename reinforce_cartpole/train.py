import gymnasium as gym
from agent import Agent
import numpy as np
def train(env,agent,episodes=500):
    total_rewards = []
    for episode in range(episodes):
        state,_ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.sample(state)
            next_state,reward,terminated,truncated,_ = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward
            state = next_state
            done = terminated or truncated

        # update the policy
        agent.update_policy()
        total_rewards.append(episode_reward)

        # 打印训练信息
        if episode % 20 == 0:
            avg_reward = np.mean(total_rewards[-20:])
            print(f"Episode {episode}\tAverage Reward (last 20): {avg_reward:.2f}")

        # 提前终止条件
        if len(total_rewards) >= 100 and np.mean(total_rewards[-30:]) >= 450:
            print(
                f"Solved! Episode {episode}. Average reward over last 100 episodes: {np.mean(total_rewards[-30:])}")
            break
    env.close()

def play(env,agent):
    state,_ = env.reset()
    done = False
    rewards = 0
    steps = 0
    while not done:
        env.render()
        steps += 1
        action= agent.sample(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards += reward
        state = next_state
        done = terminated or truncated
    print(f"step {steps}\t Reward : {rewards:.2f}")

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(
        env.observation_space.shape[0],
        env.action_space.n,
    )
    train(env,agent)
    env = gym.make('CartPole-v1',render_mode="human")
    play(env,agent)