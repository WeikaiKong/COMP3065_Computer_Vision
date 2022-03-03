import torch

x = torch.arange(4.0)
print(x)
# store gradient
x.requires_grad_(True)
# default value is none
print(x.grad)

y = 2 * torch.dot(x,x)
print(y)

# using backward to calculate grad
y.backward()
print(x.grad)
print(x.grad == 4*x)

# pytorch will sum gradience, so clear values first
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# in dl, our goal is not calculating differential matrix, is sum of derivation for each sample
# therefore, instead of outputing matrix, but scalar
x.grad.zero_()
y = x*x
# the same as y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

# move some calculation out computation graph
x.grad.zero_()
y = x*x
# consider y as a tensor, not a function
u = y.detach()
z = u*x
z.sum().backward()
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad ==2*x)

# still automatic derivation in python control flow
def f(a):
    b = a*2
    while b.norm() < 1000:
        b = b*2
    if b.sum() > 0:
        c = b
    else:
        c = 100*b
    return c
a = torch.rand(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad==d/a)