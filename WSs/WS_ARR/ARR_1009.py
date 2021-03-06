import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import collections, random
import matplotlib.pyplot as plt
import os, time
import openpyxl

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#########################################################
# USE_CUDA = torch.cuda.is_available()
# print(USE_CUDA)
#
# device = torch.device('cuda:0' if USE_CUDA else 'cpu')
# cuda = torch.device('cuda')
#
# print(cuda)
############################################################

path = "C:/Users/owner/Desktop/Workspace_paper/"

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
arr = 4

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, trasition):
        self.buffer.append(trasition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask=0.0 if done else 1.0
            done_lst.append([done_mask])

            return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                   torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                   torch.tensor(done_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


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

class Qnet(nn.Module):
    def __init__(self, len_s, len_a):
        super(Qnet, self).__init__()

        self.fc_s = nn.Linear(len_s, 128)
        self.fc_a = nn.Linear(len_a, 128)
        self.fc_cat = nn.Linear(256, 32)
        self.fc_out = nn.Linear(32, len_a)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_q)

    def forward(self, state, action):

        # start_time = time.time()

        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))
        cat = torch.cat([h1, h2], dim=1)
        h3 = F.relu(self.fc_cat(cat))
        Q_val = self.fc_out(h3)

        # print("Qnet forward time :", time.time() - start_time)

        return Q_val

    def train_net(self, target, mini_batch):

        # start_time = time.time()

        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # print("Qnet update time :", time.time() - start_time)


    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch):

    # start_time = time.time()

    s, a, r, s_prime, done =  mini_batch

    with torch.no_grad():
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2,1,keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

        # print("clac_target time :", time.time() - start_time)

    return target

def create_log_directories(root):
    """Create log directory.
    :arg (str) root    :  Root for log directory
    :returns (str) log_dir         : Log directory
             (str) reward_dir  : Log directory for reward files.
             (str) policy_dir  : Log directory for policy files.
    """
    if not os.path.isdir(root): os.mkdir(root)
    log_dir = root+"/%s" % time.strftime("%Y%m%d-%H%M%S")
    reward_dir = log_dir + "/reward/"
    policy_dir = log_dir + "/policy/"
    if not os.path.isdir(log_dir): os.mkdir(log_dir)
    if not os.path.isdir(reward_dir): os.mkdir(reward_dir)
    if not os.path.isdir(policy_dir): os.mkdir(policy_dir)
    return log_dir, reward_dir, policy_dir


def main():
    env = gym.make('InvertedPendulum-v2')
    len_s = len(env.reset())
    len_a = len(env.action_space.sample())

    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = Qnet(len_s, len_a), Qnet(len_s, len_a), Qnet(len_s, len_a), Qnet(len_s, len_a)
    pi = PolicyNet(len_s, len_a)

    q1_target.load_state_dict(q1.state_dict())
    q1_target.load_state_dict(q2.state_dict())

    score = 0.0
    time_data = []
    score_data = []
    print_interval = 10

    for n_epi in range(10000):
        start_time = time.time()
        s = env.reset()
        done = False

        while not done:
            a, _ = pi.forward(torch.from_numpy(s).float())

            a = a.cpu().detach().numpy()
            for _ in range(arr):
                s_prime, r, done, _ = env.step(a)
                env.render()
                # print("{0}?????? \n action : {1}".format(n_epi, a.item()))
                memory.put((s, a, r / 10.0, s_prime, done))
                score += r
                s = s_prime


        if memory.size() > 100:

            for _ in range(10):

                minibatch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, minibatch)
                q1.train_net(td_target, minibatch)
                q2.train_net(td_target, minibatch)
                pi.train_net(q1, q2, minibatch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        if n_epi % print_interval == 0 and n_epi != 0:

            score_data.append(score)
            run_time = time.time() - start_time
            time_data.append(run_time)
            # plt.plot(score_data)
            # plt.show(block=False)
            # plt.pause(0.00001)
            # plt.cla()

            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score / print_interval,
                                                                             pi.log_alpha.exp()))
            score = 0.0

        # print(n_epi, "episode terminal time :", time.time() - start_time)

    env.close()

    plt.plot(score_data)
    plt.show()

    score_data = np.array(score_data)
    time_data = np.array(time_data)
    # score_data = score_data.reshape(-1, 1)
    # time_data = time_data.reshape(-1, 1)
    df = np.stack((time_data, score_data), axis=1)
    df = pd.DataFrame(df, columns=['runtime', 'score'])

    log_dir, reward_dir, policy_dir = create_log_directories(path + "results")

    df.to_csv(reward_dir + "score.csv", index=False)

    torch.save(pi.state_dict(), policy_dir + "policy")

if __name__ == '__main__':
    main()




##############################################################
# Acrobot??? ??????, Reward??? ?????? ?????? ???????????? ?????????
# ???????????? RL Algorithm????????? ????????? ????????? ?????? ??????.
# ?????????, Hindsight Experience Replay??? ??????, ???????????? ??? ?????? ???????????????
# Value Fcn??? Update??? ??? ?????? ???????????? Algorithm??? ??????.
##############################################################
