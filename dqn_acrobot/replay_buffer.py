import random
from collections import deque
import numpy as np
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self,*args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch , action_batch , state_next_batch , reward_batch, done_mask_batch = zip(*batch)
        return np.array(state_batch).astype('float32'),np.array(action_batch).astype('float32'),\
            np.array(state_next_batch).astype('float32'),np.array(reward_batch).astype('float32'),np.array(done_mask_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)