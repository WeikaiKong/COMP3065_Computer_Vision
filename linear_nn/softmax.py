import torch 
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# transfer each image to vector whose length is 784(28*28)
num_inputs = 784
# there are 10 categories in the datasets
num_outputs = 10

# initial parameters
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    # broadcasting
    return X_exp/partition

# test for softmax
# X = torch.normal(0, 1, size=(2,5))
# X_prob = softmax(X)
# print(X_prob)
# print(X_prob.sum(1))

# softmax regression model
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.reshape[0])), W) + b)

# example of seeking percentages in a category
# y: samples, y_hat: predictions
# find predictions of y[0], y[1]
y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.5], [0.2,0.5,0.7]])
print(y_hat[[0,1], y])

# cross entropy loss function
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y])
print(cross_entropy(y_hat, y))