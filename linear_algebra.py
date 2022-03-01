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