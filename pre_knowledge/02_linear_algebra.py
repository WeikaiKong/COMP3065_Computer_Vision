from cgi import test
from matplotlib.pyplot import axis
import torch

# scalar
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x+y,x*y,x/y,x**y)

# vector = list of scalar's values
x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

# matrix
A = torch.arange(20).reshape((5,4))
print(A)
# transform
print(A.T)
# symmetric matrix, A = A.T
B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B)
print(B == B.T)
# more dimensions
x = torch.arange(24).reshape(2,3,4)
print(x)

# operations
A = torch.arange(20, dtype=torch.float32).reshape(5,4)
# restore memory
B = A.clone() 
print(A,A+B)
# hadamard product: each element in two arraies production
print(A*B)

# dimention reduction
x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
print(A.sum(), A.shape)
# dimention reduction in axis = 0
A_sum_axix0 = A.sum(axis=0)
print(A_sum_axix0,A_sum_axix0.shape)
# dimention reduction in axis = [0,1]
# same as A.sum()
# eg. shape[2,5,4], when axis = 0, then [5,4]; when axis = 1, then [2,4]; when axis = 2, then [2,5]
# when axis = [1,2], then [2]
# so for axis = [a1, a2..an], just delete an in shape[1..m]
A.sum(axis=[0,1])
# mean
print(A.mean())
print(A.sum()/A.numel())
# mean in axis = 0
print(A.mean(axis=0))
print(A.sum(axis=0)/A.shape[0]) 
# usually we need to keep the dimention
# calculating means value without dimention reduction
# eg. shape[2,5,4], when axis = 0, then [1,5,4]
# so just when axis = 1..n, then [1]..[n] = 1
print(A.sum(axis=1))
sum_A = A.sum(axis=1, keepdim=True)
print(sum_A)
# broadcasting
print(A/sum_A)
# Cumulative sum
print(A.cumsum(axis=0))

# dot
y = torch.ones(4, dtype=torch.float32)
print(x,y,torch.dot(x,y))
print(torch.sum(x*y))
# matrix verctor multiplication
print(A.shape,x.shape,torch.mv(A,x))
# matrix matrix muliplication
B = torch.ones(4,3)
print(torch.mm(A,B))

# normal
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
# L1 normal
# sum of abs
print(torch.abs(u).sum())
# Frobenius normal
torch.normal(torch.ones(4,9))