import numpy as np

class Agent(object):
    def __init__(self,obs_n,act_n,gamma=0.9,epsilon=0.1,alpha=0.1):
        self.obs_n = obs_n
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_table = np.zeros((self.obs_n,self.act_n))

    def predict(self,obs):
        list_to_select = self.q_table[obs,:]
        q_max =  np.max(list_to_select)
        action_list = np.where(q_max == list_to_select)[0]
        return np.random.choice(action_list)

    def sample(self,obs):
        if np.random.uniform(0,1) < 1 - self.epsilon:
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    def learn(self,obs,action,reward,next_obs,done):
        td_predict = self.q_table[obs,action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_obs,:])
        self.q_table[obs,action] = td_predict + self.alpha * (td_target - td_predict)

    def save(self):
        np.save("q_table",self.q_table)
        print("save q_table")

    def load(self):
        self.q_table = np.load("q_table")
        print("load q_table")