import torch
import IPython as display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim
from collections import OrderedDict

num_inputs = 2
num_example = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_example, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

net = torch.nn.Sequential(OrderedDict([
    ('linear', torch.nn.Linear(num_inputs, 1))
]))

print(net)  # 使用print可以打印出网络的结构
for param in net.parameters():
    print(param)

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

loss = torch.nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
