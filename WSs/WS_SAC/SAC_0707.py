import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

#################################################################################
# 2nd practice

# HyperParameters
learning_rate = 0.0001
gamma = 0.98

class AC_algorithm(nn.Module):
    def __init__(self):
        super(AC_algorithm, self).__init__()

        self.Data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_vFcn = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def for_pi(self, state, softmax_dim=0):

        h1 = F.relu(self.fc1(state))
        pi = self.fc_pi(h1)
        prob_act = F.softmax(pi, softmax_dim)

        return prob_act

    def for_vFcn(self, state):

        h1 = F.relu(self.fc1(state))
        vFcn = self.fc_vFcn(h1)

        return vFcn

    def put_data(self, Transition):
        self.Data.append(Transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for trasiton in self.Data:
            s, a, r, s_prime, done = trasiton

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

            s_batch, a_batch, r_batch, s_prime_batch, done_batch = \
            torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
            torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype = torch.float), \
            torch.tensor(done_lst, dtype=torch.float)

        self.Data = []

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = self.make_batch()

        pi = self.for_pi(s_batch, softmax_dim=1)
        pi_a = torch.gather(pi, 1, a_batch)

        td_target = r_batch + gamma*self.for_vFcn(s_prime_batch)*done_batch
        delta = td_target - self.for_vFcn(s_batch)

        loss =  -torch.log(pi_a)*delta.detach() + F.smooth_l1_loss(self.for_vFcn(s_batch), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def main():

    env = gym.make('CartPole_v1')
    AC = AC_algorithm()
    print_interval = 20
    score = 0.0

    for epi in range(10000):
        done = False
        s = env.reset()
        env.render()

        while not done:

            prob_act = AC.for_pi(s)
            m = Categorical(prob_act)
            a = m.sample().item()

            s_prime, r, done, info = env.step(a)
            AC.put_data((s, a, r, s_prime, done))

            s = s_prime
            score += r

            if done :
                break

        AC.train_net()

        if (epi % print_interval == 0) and (epi != 0):
            print("# of episode : {}, avg score : {:.1f}".format(epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()








##################################################################################
# 1st practice

# # HyperParameters
# Learning_rate = 0.0001
# gamma = 0.98
# batch_size = 10
#
# class AC_algorithm(nn.Module):
#     def __init__(self):
#         super(AC_algorithm, self).__init__()
#
#         self.data = []
#
#         self.fc1 = nn.Linear(4, 256)
#         self.fc_pi = nn.Linear(256, 2)
#         self.fc_val_fcn = nn.Linear(256, 1)
#         self.optimizer = optim.Adam(self.parameters(), lr=Learning_rate)
#
#     def forward_pi(self, state, softmax_dim=0):
#
#         h1 = F.relu(self.fc1(state))
#         pi = self.fc_pi(h1)
#         prob_act = F.softmax(pi, softmax_dim)
#
#         return prob_act
#
#
#     def forward_val_fcn(self, state):
#         h1 = F.relu(self.fc1(state))
#         val_fcn = self.fc_val_fcn(h1)
#
#         return val_fcn
#
#     def put_data(self, transition):
#         self.data.append(transition)
#
#     def make_batch(self):
#         s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
#         for transition in self.data:
#             s, a, r, s_prime, done = transition
#             s_lst.append(s)     # s는 애초부터 list형태라 [] 필요 x
#             a_lst.append([a])
#             r_lst.append([r])
#             s_prime_lst.append(s_prime)
#             done_mask = 0 if done else 1.0
#             done_lst.append([done_mask])
#
#         s_batch, a_batch, r_batch, s_prime_batch, done_batch = \
#             torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst),\
#             torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float),\
#             torch.tensor(done_lst, dtype=torch.float)
#
#         self.data = []
#
#         return s_batch, a_batch, r_batch, s_prime_batch, done_batch
#
#     def train_net(self):
#
#         s, a, r, s_prime, done_msk = self.make_batch()
#         td_target = r + gamma*self.forward_val_fcn(s_prime) * done_msk
#         delta = td_target - self.forward_val_fcn(s)
#
#         pi = self.forward_pi(s, softmax_dim=1)
#         pi_a =torch.gather(pi, 1, a)
#         loss = -torch.log(pi_a)*delta.detach() + F.smooth_l1_loss(self.forward_val_fcn(s), td_target.detach())
#
#         self.optimizer.zero_grad()
#         loss.mean().backward()
#         self.optimizer.step()
#
# def main():
#     env = gym.make('CartPole-v1')
#     AC = AC_algorithm()
#     print_interval = 20
#     score = 0.0
#     score_data = []
#
#     for n_epi in range(1000):
#         done = False
#         s = env.reset()
#         env.render()
#
#         while not done:
#             for _ in range(batch_size):
#                 prob_act = AC.forward_pi(torch.from_numpy(s).float())
#                 m = Categorical(prob_act)
#                 a = m.sample().item()
#                 s_prime, r, done, _ = env.step(a)
#                 AC.put_data((s, a, r, s_prime, done))
#
#                 s = s_prime
#                 score += r
#                 score_data.append(score)
#
#                 if done:
#                     break
#
#
#             AC.train_net()
#
#         if n_epi % print_interval == 0 and n_epi != 0:
#             print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
#             # score = 0.0
#
#     env.close()
#
#     print(score_data.__sizeof__())
#     plt.plot(score_data, label="score")
#     plt.show()
#
# if __name__ == '__main__':
#     main()