import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# generate datasets
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# load dataset: using API in torch
def load_array(data_arrays, batch_size, is_train=True):
    # construct pytorch data iterator 
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels), batch_size)
print(next(iter(data_iter)))

# using nn layer in pytorch
net = nn.Sequential(nn.Linear(2,1))

# initial model parameters
net[0].weight.data.normal_(mean=0, std=0.01)
net[0].bias.data.fill_(0)

# loss function: L2
loss = nn.MSELoss()

# SGD
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# train
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        # .step: parameter update
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch: {epoch + 1}, loss {l:f}')