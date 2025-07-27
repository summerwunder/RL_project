from collections import deque
import random
import torch
import numpy as np
class ReplayMemory(object):
    def __init__(self, capacity=500):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        # 此处传入的是元组，一次push对应着一个元组
        self.memory.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return np.array(state_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
            np.array(next_state_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    a = torch.tensor([[1,2],[2,2],[3,3]])
    print(a.shape)
    print(a.max(1).values.shape)   # 降为了  3*2 --》 3
    memory = ReplayMemory(20)
    memory.push((1,2),3,4,5,6)              #(1,3,4)
    memory.push((1,5),3,4,3,4)
    memory.sample(2)             #[(1,3,4)] ----> zip(*batch)--->  ((1, 5), (1, 2))
