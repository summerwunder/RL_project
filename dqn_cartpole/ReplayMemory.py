from collections import deque
import numpy as np
import random

class ReplayMemory(object):
    def __init__(self, capacity:int)->None:
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size:int):
        batch = random.sample(self.memory, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.memory)


