from numpy import dtype
import torch

# tensor
x = torch.arange(12)
print(x)
# size of tensor
print(x.shape)
# number of elements
print(x.numel())
# reshape
print(x.reshape(3,4))
# zeros tensor
print(torch.zeros((2,3,4)))
# ones tensor
print(torch.ones((2,3,4)))
# initialisaion tensor list
print(torch.tensor([[1,2,3,4],[2,3,4,5]]).shape)

# sames as basic mathematic operations
y = torch.tensor([1.0, 2, 3, 5])
print(torch.exp(y))

# multi-tensors
a = torch.arange(12, dtype=torch.float32).reshape(3,4)
b = torch.tensor([[2.0,1,3,4], [11,23,43,22], [2,3,4,5.0]])
# dimention = 0
print(torch.cat((a,b), dim=0))
# dimention = 1
print(torch.cat((a,b), dim=1))
# logic operator to construct tensor 
print(a==b)
# sum of elements
print(a.sum())

# read elements
# last raw
print(a[-1])
# second and third raw
print(a[1:3])

# changes elements
# change one element
a[1,2] = 9
print(a)
# change multi elements
a[0:2,:] = 12
print(a)

# broadcasting mechanism
c = torch.arange(3).reshape(3,1)
d = torch.arange(2).reshape(1,2)
print(c+d)

# memory restore
before = id(b)
b = b+a
print(id(b)==before)
# operations in the same memory
z = torch.zeros_like(b)
print('id(z):', + id(z))
z[:] = a+b
print('id(z):', + id(z))
# Or
before = id(b)
# b[:] = a+b
b += a
print(id(b)==before)

# numpy tensor
d = a.numpy()
e = torch.tensor(d)
print(type(d))
print(type(e))