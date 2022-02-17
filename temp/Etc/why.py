import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import time

def main():
    env = gym.make('InvertedPendulum-v2')
    s = env.reset()
    print("state : ", s)
    print("len of state : ", len(s))

    for _ in range(5000):
        # start_time = time.time()
        a = env.action_space.sample()
        print("action : ", a)
        print("len of action : ", len(env.action_space))
        env.step(a)
        env.render()
        # print("time :", time.time() - start_time)  # 현재시각 - 시작시간 = 실행 시간

if __name__ == '__main__':
    main()