import numpy as np
class Agent(object):
    def __init__(self,obs_n,act_n,lr=0.01,gamma=0.9,e_greed=0.1):
        self.obs_n = obs_n
        self.act_n = act_n
        self.lr = lr
        self.gamma = gamma
        self.e_greed = e_greed
        self.q_table = np.zeros((self.obs_n,self.act_n))

    def sample(self,obs):
        if np.random.uniform(0,1) < (1- self.e_greed):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    def predict(self,obs):
        Q_list = self.q_table[obs,:]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        return np.random.choice(action_list)

    def learn(self,obs,action,reward,next_obs,next_action,done):
        predict_Q = self.q_table[obs,action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * self.q_table[next_obs,next_action]
        self.q_table[obs,action] += self.lr * (target_Q - predict_Q)

    def save(self):
        npy_file = "./q_table.npy"
        np.save(npy_file,self.q_table)
        print("saved q_table")

    def load(self):
        npy_file = "./q_table.npy"
        self.q_table = np.load(npy_file)