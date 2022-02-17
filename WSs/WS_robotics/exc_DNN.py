import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyperparameters
Learning_rete = 0.0001
batch_size =100

class simple_NN(nn.Module):
    def __init__(self):
        super(simple_NN, self).__init__()

        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=Learning_rete)

    def forward(self, x):

        hidden1 = F.relu(self.fc1(x))
        hidden2 = F.relu(self.fc2(hidden1))
        y = self.fc3(hidden2)

        return y

    def train_net(self, x_batch, y_batch):

        y_hat = self.forward(x_batch)
        mse_loss = F.mse_loss(y_hat, y_batch)
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

# Data generator
source =  np.linspace(-10, 10, 1000).reshape(-1, 1)
X =  np.hstack([source, source**2, source**3])
Y =  np.sum(X, axis=1, keepdims=True)

# Define Network

net = simple_NN()

for i in range(1000):

    idx =  np.arange(len(X))
    np.random.shuffle(idx)
    batch_idx = idx[:batch_size]
    X_batch = torch.tensor(X[batch_idx], dtype=torch.float)
    Y_batch = torch.tensor(Y[batch_idx], dtype=torch.float)
    net.train_net(X_batch, Y_batch)

    X_tensor = torch.tensor(X, dtype=torch.float)
    Y_pred = net.forward(X_tensor)
    Y_pred = Y_pred.detach().numpy()
    plt.plot(X[:, 0], Y, label)