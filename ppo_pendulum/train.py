import  os
import gymnasium as gym
import torch
from PPO import PPO
from tqdm import tqdm
import numpy as np

class Config:
    num_episodes = 800
    state_dim = None
    action_dim = None
    hidden_layer_dim = [128,128]
    actor_lr = 1e-4
    critic_lr = 5e-3
    gamma = 0.9
    ppo_kwargs = {
        'lambda' : 0.9,
        'eps' : 0.2,
        'ppo_epochs' : 10
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    buffer_capacity = 20000
    batch_size = 32
    save_dir = './net/'
    max_episode_reward = 260
    max_episode_steps = 260

    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]  #可以覆盖类变量
        try:
            self.action_dim = env.action_space.n
        except Exception as e:
            self.action_dim = env.action_space.shape[0]
        print(f"device: {self.device}")


def train(env,config):
    agent = PPO(config.state_dim,
                config.action_dim,
                config.hidden_layer_dim,
                config.actor_lr,
                config.critic_lr,
                config.gamma,
                config.buffer_capacity,
                config.ppo_kwargs,
                config.device)

    tq_bar = tqdm(range(config.num_episodes))
    rewards = []
    best_reward = -np.inf

    for episode in tq_bar:
        agent.buffer.reset()
        tq_bar.set_description(f"episode: {episode+1}/{config.num_episodes}")
        s , _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done :
            a = agent.select_action(s)
            n_s , r, done , _ ,_ = env.step(a)
            agent.buffer.push(s, a, r, n_s, done)
            episode_reward += r
            s = n_s
            steps += 1
            if (episode_reward > config.max_episode_reward) or (steps > config.max_episode_steps):
                break
        agent.learn()
        rewards.append(episode_reward)
        now_reward = np.mean(rewards[-10:])
        if best_reward < now_reward:
            torch.save(agent.actor.state_dict(), config.save_dir + 'actor.pth')
            best_reward = now_reward
        tq_bar.set_postfix({'lastMeanRewards': f'{now_reward:.2f}', 'BEST': f'{best_reward:.2f}'})
    env.close()
    return agent


def play(env, agent,  episode_count=3):
    for e in range(episode_count):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        episode_cnt = 0
        while not done:
            env.render()
            a = agent.select_action(s)
            n_state, reward, done, _, _ = env.step(a)
            episode_reward += reward
            episode_cnt += 1
            s = n_state

        print(f'Get reward {episode_reward}. Last {episode_cnt} times')
    env.close()

if __name__ == '__main__':
    print("Train begin...")
    env = gym.make('Pendulum-v1')
    config = Config(env)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    agent = train(env,config)

    agent.actor.load_state_dict(torch.load(config.save_dir + 'actor.pth',weights_only=True))
    play(gym.make('Pendulum-v1', render_mode="human"), agent)