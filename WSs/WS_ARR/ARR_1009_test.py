import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import collections, random
import matplotlib.pyplot as plt
import os
import pandas as pd
import openpyxl
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

Path = "C:/Users/owner/Desktop/Workspace_paper/"

# Hyperparameters
buffer_limit = 50000
batch_size = 32
lr_pi = 0.0005
lr_q = 0.0001
lr_alpha = 0.01
init_alpha = 0.01
target_min_entropy = -1.0
gamma = 0.98
tau = 0.01

class PolicyNet(nn.Module):
    def __init__(self, len_s, len_a):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(len_s, 256)
        self.fc_mu = nn.Linear(256, len_a)
        self.fc_std = nn.Linear(256, len_a)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_pi)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_opimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, state):

        # start_time = time.time()

        h1 = F.relu(self.fc1(state))
        mu = self.fc_mu(h1)
        std = F.softplus(self.fc_std(h1))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob - torch.log(1 - torch.tanh(action)**2 + 1e-7)
        action = torch.tanh(action)

        # print("pi forward time :", time.time() - start_time)

        return action, log_prob

    def train_net(self, q1, q2, mini_batch):

        # start_time = time.time()

        s, _, _, _, _ = mini_batch
        a, log_prob =  self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -(min_q + entropy).cpu()
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_opimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp())*(log_prob + target_min_entropy).detach()
        alpha_loss.mean().backward()
        self.log_alpha_opimizer.step()

        # print("pi update time :", time.time() - start_time)

def main():
    env = gym.make('InvertedDoublePendulum-v2')
    len_s = len(env.reset())
    len_a = len(env.action_space.sample())

    pi = PolicyNet(len_s, len_a)
    pi.load_state_dict(torch.load(Path + "results/20211021-221613/policy/policy"))

    score = 0.0
    time_data = []
    score_data = []
    state_data = []
    print_interval = 10

    for n_epi in range(10):
        start_time = time.time()
        s = env.reset()
        done = False

        while not done:
            a, _ = pi.forward(torch.from_numpy(s).float())
            a = a.cpu().detach().numpy()
            # print("action : ", a)
            s_prime, r, done, _ = env.step(a)
            env.render()
            score += r
            s = s_prime
            state_data.append(s)

        print("score : ", score)
        run_time = time.time() - start_time
        print("run time : ", run_time)
        # plt.plot(score_data)
        # plt.show(block=False)
        # plt.pause(0.00001)
        # plt.cla()

        score = 0.0

        # print(n_epi, "episode terminal time :", time.time() - start_time)

    env.close()

    plt.plot(score_data)
    plt.show()

    # df = pd.DataFrame(state_data)
    # df.to_csv(Path + "results/score.csv", index=False)

if __name__ == '__main__':
    main()




##############################################################
# Acrobot의 경우, Reward를 받기 힘든 환경이기 때문에
# 일반적인 RL Algorithm으로는 학습을 시키기 힘든 환경.
# 그래서, Hindsight Experience Replay와 같이, 리워드를 못 받은 상황에서도
# Value Fcn을 Update할 수 있는 보조적인 Algorithm이 필요.
##############################################################
