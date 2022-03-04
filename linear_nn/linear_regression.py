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

# inputs: batch_size, features, labels
# outputs: xn, yn whose size is batch_size
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    # from 0 to num_examples, skip nums of indices is batch_size
    for i in range(0, num_examples, batch_size):
        # min(i+batch_size, num_examples): to prohibt over num_examples
        batch_indices =  torch.tensor(
            indices[i:min(i+batch_size, num_examples)])
        # yield: iterator in python
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X,y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# initial model parameters
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# define model: y = Xw + b
def linreg(X, w, b):
    return torch.matmul(X,w) + b

# define loss function: squared loss function
# no means
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

# define optimization algorithm: SGD: minibatch stochastic gradient descent
# inputs: params including w and b; learning rate; batch size
# goals: update params including w and b
def sgd(params, lr, batch_size):
    # when update params, no gradient calculation
    with torch.no_grad():
        for param in params:
            # update parameters, then calculate means
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# train
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

# for each epoch, for each xi, yi in each batch
# first build loss function, then backward to calculating each gradient
# finally using optimization function to update paramters(w and b)
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b),y)
        # l.shape is ('batch_size', 1)
        l.sum().backward()
        sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_1 = loss(net(features,w,b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_1.mean()):f}')

# comparing with true w and b, to evaluate model
print(f'error of w: {true_w - w.reshape(true_w.shape)}')
print(f'error of b: {true_b - b}')