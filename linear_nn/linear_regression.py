# linear regression can be considered as single layer neural network
# matplotlib inline
import random
from statistics import mean
import numpy
import torch
from d2l import torch as d2l

# generate handcraft datasets with denoise
# y = Xw + b + noise
def synthetic_data(w, b, num_examples):
    # torch.normal(mean, std, size)
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))
    # torch.matmul()-> matrix multiple
    y = torch.matmul(X,w) + b
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features: ', features[0], '\nlabels: ', labels[0])
# visualization
d2l.set_figsize()
# .detach() : torch -> numpy
d2l.plt.scatter(features[:,1].detach().numpy(), labels.detach().numpy(),1)
d2l.plt.show()