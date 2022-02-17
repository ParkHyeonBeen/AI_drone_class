import numpy as np
import torch
import collections, random

from WSs.WS_FIP.fip_config import FIP_config

config = FIP_config()

class ReplayBuffer():
    def __init__(self):
        self.buffer_limit = config.buffer_limit
        self.buffer = collections.deque(maxlen=self.buffer_limit)

        self.history_state = np.array([])

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

    def sample_history(self, n, history_num):
        idx = np.arange(self.size())
        np.random.shuffle(idx)
        batch_idx = idx[:n]

        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for i in batch_idx:
            s, a, r, s_prime, done = self.buffer[i]

            for j in range(1, history_num):
                if i < (history_num - 1):
                    s_before = self.buffer[i][0]
                    s_prime_before = self.buffer[i][3]
                else:
                    s_before = self.buffer[i - j][0]
                    s_prime_before = self.buffer[i - j][3]
                s = np.hstack((s_before, s))
                s_prime = np.hstack((s_prime_before, s_prime))
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_lst, dtype=torch.float)

    def make_history(self, s, history_num, first_step):

        if first_step:
            self.history_state = np.array([])
            for _ in range(history_num):
                self.history_state = np.hstack((s, self.history_state))
        else:
            self.history_state = np.hstack((self.history_state, s))
            self.history_state = np.delete(self.history_state, np.arange(len(s)))

        return self.history_state

    def size(self):
        return len(self.buffer)