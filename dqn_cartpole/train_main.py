import numpy as np
import gymnasium as gym
from Agent import DQNAgent
import matplotlib.pyplot as plt
def train(env,agent,episodes=500,max_steps=500,sync_every=20):
    rewards = []
    for episode in range(episodes):
        s,_ = env.reset()
        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
        total_reward = 0
        for _ in range(max_steps):
            a = agent.sample(s)
            ns , r , terminal ,truncated ,_= env.step(a)
            agent.buffer.push(s,a,r,ns,terminal)
            agent.learn()
            s = ns
            total_reward += r
            if terminal or truncated:
                break
        rewards.append(total_reward)
        if episode % sync_every == 0:
            agent.sync_target()
        print(f"Ep {episode:3d} | Reward: {total_reward:5.1f} | ε: {agent.eps:.3f}")
        if np.mean(rewards[-100:])>160:
            print(f"Solved in {episode} episodes!")
            break
    return rewards

def play(env,agent,max_steps=500):
    s,_ = env.reset()
    for _ in range(max_steps):
        env.render()
        a = agent.predict(s)
        s,_,term,trunc,_ = env.step(a)
        if term or trunc:
            break
    env.close()

if __name__ == "__main__":
    env_train = gym.make("CartPole-v1")
    agent = DQNAgent(env_train.observation_space.shape[0],
                     env_train.action_space.n)
    rewards = train(env_train, agent)

    env_train.close()

    plt.plot(rewards, alpha=0.5, label="episode reward")
    plt.legend();
    plt.show()

    env_play = gym.make("CartPole-v1", render_mode="human")
    play(env_play, agent)