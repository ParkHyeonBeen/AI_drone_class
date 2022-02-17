import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.distributions import categorical
#softmax 활성함수를 통해 나온 각 행동에 대한 확률을 확률밀도함수로 만들어줘야 합니다.
#이를 쉽고 빠르게 처리할 수 있도록 torch가 제공하는 Categorical 패키지를 import


# Hyperparameters
learning_rate   = 0.0002
minibatch = 1000
gamma           = 0.98

class SoftActorCritic(nn.Module):
    def __init__(self):
        super(SoftActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        y = self.fc3(h2)
        return y

    def train_net(self, x_batch, y_batch):
        y_hat = self.forward(x_batch)
        mse_loss = F.mse_loss(y_hat, y_batch)
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

# Data generation
source = np.linspace(-10,10,1000).reshape([-1,1])
X = np.hstack([source, source**2, source**3])
Y = np.sum(X,axis=1,keepdims=True)

# Define network
net = SoftActorCritic()

for i in range(5000):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    batch_idx = idx[:minibatch]
    print(batch_idx)
    x_batch = torch.tensor(X[batch_idx], dtype=torch.float)
    y_batch = torch.tensor(Y[batch_idx], dtype=torch.float)
    net.train_net(x_batch,y_batch)

    tensor_X = torch.tensor(X, dtype=torch.float)
    y_pred = net.forward(tensor_X)
    y_pred = y_pred.detach().numpy()
    plt.plot(X[:,0], Y, label='Label')
    plt.plot(X[:,0], y_pred, label='Pred')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)
    plt.cla()

#############################################################################################

# my version

# import torch
# import torch.nn  as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
#
# # HyperParameters
# Learning_rate = 0.0001
# batch_size = 100
#
# class simple_NN(nn.Module):
#     def __init__(self):
#         super(simple_NN, self).__init__()
#
#         self.fc1 = nn.Linear(3,256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 1)
#         self.optimizer = optim.Adam(self.parameters(), lr=Learning_rate)
#
#     def forward(self, x):
#
#         hidden1 = F.relu(self.fc1(x))
#         hidden2 = F.relu(self.fc2(hidden1))
#         y = self.fc3(hidden2)
#
#         return y
#
#     def train_net(self, x_batch, y_batch):
#
#         y_hat = self.forward(x_batch)
#         square = (y_hat - y_batch)**2
#         mse_loss = torch.sum(square)/len(y_hat)
#
#         #mse_loss = F.mse_loss(y_hat, y_batch)
#
#         self.optimizer.zero_grad()
#         mse_loss.backward()
#         self.optimizer.step()
#
# # source generation
#
# source = np.linspace(-10, 10, 1000).reshape(-1, 1)
# X = np.hstack([source, source**2, source**3])
# Y = np.sum(X,axis=1,keepdims=True)  # numpy 연산을 하면 원형이 무너지기 때문에, keepdims 또는 reshape 을 해줘야 함
#
# # train data
#
# net = simple_NN()
#
# for _ in range(1000):
#
#     idx = np.arange(len(X))
#     np.random.shuffle(idx)
#     idx_batch =  idx[:batch_size]
#     x_batch = torch.tensor(X[idx_batch], dtype=torch.float)
#     y_batch = torch.tensor(Y[idx_batch], dtype=torch.float)
#     net.train_net(x_batch, y_batch)
#
#     x_tensor = torch.tensor(X, dtype=torch.float)
#     y_pred = net.forward(x_tensor)
#     y_pred = y_pred.detach().numpy()
#     plt.plot(X[:, 0], Y, label="origin")
#     plt.plot(X[:, 0], y_pred, label="trained")
#     plt.legend()
#     plt.show(block=False)
#     plt.pause(0.001)
#     plt.cla()