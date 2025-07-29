from collections import deque
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity : int):
        self.buffer = deque(maxlen=capacity)

    # state, action, reward, next_state, done
    def push(self,*args):
        self.buffer.append(args)

    def sample(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch = zip(*self.buffer)
        return (np.array(state_batch).astype('float32'),np.array(action_batch).astype('float32')
                ,np.array(reward_batch).astype('float32'),np.array(next_state_batch).astype('float32')
                ,np.array(done_mask_batch).astype('float32'))

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer.clear()


