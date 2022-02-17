import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os, time

from WSs.WS_FIP.fip_config import FIP_config
from tools.buffer import *


#########################################################
""" related to cuda """

os.environ['KMP_DUPLICATE_LIB_OK']='True'

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
cuda = torch.device('cuda')

print(cuda)
###########################################################

config = FIP_config()

MIN_LOG_SIG = -10. # default = -20.
MAX_LOG_SIG = 2.   # default = 2.

class PolicyNet(nn.Module):
    def __init__(self, len_s, len_a):
        super(PolicyNet, self).__init__()

        self.lr_pi = config.learning_rate_pi
        self.init_alpha = config.initial_alpha
        self.lr_alpha = config.learning_rate_alpha
        self.target_min_entropy = config.target_min_entropy

        self.fc1 = nn.Linear(len_s, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.fc_mu = nn.Linear(256, len_a, bias=True)
        self.fc_std = nn.Linear(256, len_a, bias=True)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_pi)

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).cuda()
        self.log_alpha.requires_grad = True
        self.log_alpha_opimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

    def forward(self, state):

        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        log_std = self.fc_std(h2)
        log_std = torch.clamp(log_std, min=MIN_LOG_SIG, max=MAX_LOG_SIG)
        std = log_std.exp()
        dist = Normal(mu, std, validate_args=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = (log_prob - torch.log(1 - torch.tanh(action)**2 + 1e-7)).sum(dim=-1, keepdim=True)
        action = torch.tanh(action)
        eval_action = torch.tanh(mu)
        return action, eval_action, log_prob

    def train_net(self, q1, q2, mini_batch):

        s, _, _, _, _ = mini_batch
        a, _, log_prob = self.forward(s.cuda())
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s.cuda(), a), q2(s.cuda(), a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -(min_q + entropy).cpu()
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_min_entropy).detach()).mean()
        self.log_alpha_opimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opimizer.step()

class Qnet(nn.Module):
    def __init__(self, len_s, len_a):
        super(Qnet, self).__init__()
        self.lr_q = config.learning_rate_q
        self.tau = config.tau

        self.fc_1 = nn.Linear(len_s+len_a, 256, bias=True)
        self.fc_2 = nn.Linear(256, 256, bias=True)
        self.fc_3 = nn.Linear(256, 1, bias=True)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_q)

    def forward(self, state, action):

        inp = torch.cat([state, action],dim=1)
        h1 = F.relu(self.fc_1(inp))
        h2 = F.relu(self.fc_2(h1))
        Q_val = self.fc_3(h2)
        return Q_val

    def train_net(self, target, mini_batch):

        # start_time = time.time()
        s, a, r, s_prime, done = mini_batch
        loss = F.mse_loss(self.forward(s.cuda(), a.cuda()), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

def calc_target(pi, q1, q2, mini_batch):

    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, _, log_prob = pi(s_prime.cuda())
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime.cuda(), a_prime), q2(s_prime.cuda(), a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = (r.cuda() + config.gamma * done.cuda() * (min_q + entropy)).detach()
    return target

def main():
    env = gym.make('InvertedDoublePendulum-v2')
    len_s = len(env.reset())
    len_a = len(env.action_space.sample())

    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = Qnet(len_s, len_a), Qnet(len_s, len_a), Qnet(len_s, len_a), Qnet(len_s, len_a)
    q1.cuda(), q2.cuda(), q1_target.cuda(), q2_target.cuda()
    pi = PolicyNet(len_s, len_a)
    pi.cuda()

    q1_target.load_state_dict(q1.state_dict())
    q1_target.load_state_dict(q2.state_dict())

    score = 0.0
    print_interval = 10

    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            a, _ = pi.forward(torch.from_numpy(s).cuda().float())
            a = a.cpu().detach().numpy()
            s_prime, r, done, success = env.step(a)
            env.render()
            # print("{0}번째 \n action : {1}".format(n_epi, a))
            memory.put((s, a, r/10, s_prime, done))
            score += r
            s = s_prime

        if memory.size() > 100:

            for _ in range(10):

                minibatch = memory.sample(config.batch_size)
                td_target = calc_target(pi, q1_target, q2_target, minibatch)
                q1.train_net(td_target, minibatch)
                q2.train_net(td_target, minibatch)
                pi.train_net(q1, q2, minibatch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, alpha : {:.4f}".format(n_epi, score / print_interval,
                                                                                pi.log_alpha.exp()))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()