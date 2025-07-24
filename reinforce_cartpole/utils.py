import matplotlib.pyplot as plt
from typing import List

def plot_rewards(rewards: List[float], window: int = 50, filename: str = 'training_rewards.png'):
    """绘制训练过程中的总奖励变化图。"""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward')
    if len(rewards) >= window:
        moving_avg = [sum(rewards[i-window:i])/window for i in range(window, len(rewards))]
        plt.plot(range(window, len(rewards)), moving_avg, label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()