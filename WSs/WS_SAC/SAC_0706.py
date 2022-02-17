import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Data load
X = np.load("X_data.npy")
Y = np.load("Y_data.npy")

#hyperparameters
Learning_rate = 0.0003
batch_size = 100

class simple_NN(nn.Module):
    def __init__(self):
        super(simple_NN, self).__init__()

        self.fc1 = nn.Linear(3,256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=Learning_rate)

    def forward(self, x):

        hidden1 = F.relu(self.fc1(x))
        hidden2 = F.relu(self.fc2(hidden1))
        y = self.fc3(hidden2)

        return y

    def train_net(self, x_batch, y_batch):

        y_hat = self.forward(x_batch)

        square = torch.zeros(100, 1)
        for i in range(2):
            tmp = (y_hat[:, i] - y_batch[:, i])**2
            tmp = tmp.reshape(100, 1)
            square += tmp

        mse_loss = torch.sum(square)/len(y_hat)
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()


net = simple_NN()

for _ in range(1000):

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    idx_batch = idx[:batch_size]
    x_batch = torch.tensor(X[idx_batch], dtype=torch.float)
    y_batch = torch.tensor(Y[idx_batch], dtype=torch.float)
    net.train_net(x_batch, y_batch)

    # x_tensor = torch.tensor(X, dtype=torch.float)
    # y_pred = net.forward(x_tensor)
    # y_pred = y_pred.detach().numpy()
    # plt.plot(X[:, 0], Y, label="origin")
    # plt.plot(X[:, 0], y_pred, label="trained")
    # plt.legend()
    # plt.show(block=False)
    # plt.pause(0.001)
    # plt.cla()

#######################################################################################
## Model 내부 출력

# print("Model`s state_dict : ")
# for param_tensor in net.state_dict():
#     print(param_tensor, "/t", net.state_dict()[param_tensor].size())
#
# print("Optimizer's state_dict:")
# for var_name in net.optimizer.state_dict():
#     print(var_name, "\t", net.optimizer.state_dict()[var_name])

path = "C:/Users/owner/Desktop/Workspace_paper/" + "saved_torch"
torch.save(net.state_dict(), path)

new_net = simple_NN()
new_net.load_state_dict(torch.load(path))
new_net.eval()
# 추론을 실행하기 전에는 반드시 model.eval() 을 호출하여 드롭아웃 및 배치 정규화를 평가 모드로 설정
# 하지 않으면 추론 결과가 일관성 없게 출력

new_net_entire = torch.load(path)
new_net_entire.eval()

x_tensor = torch.tensor(X, dtype=torch.float)
y_pred = new_net.forward(x_tensor)
y_pred = y_pred.detach().numpy()
plt.plot(X[:, 0], Y, label="origin")
plt.plot(X[:, 0], y_pred, label="trained")
plt.legend()
plt.show()
