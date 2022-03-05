from matplotlib.pyplot import get
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# use svg to visualize 
d2l.use_svg_display()

# load train and test datasets in Fashion MNIST
# To.Tensor(): for image data, PIL -> float 32
# then float32/255 to make pixel between (0,1)
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
# size of datasets
print(len(mnist_train), len(mnist_test))
# size of image
print(mnist_train[0][0].shape)

# return labels in datasets
def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

# draw images list
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# load batch size
batch_size = 256
# 
def get_dataloader_workers():
    return 4
trian_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
# time of trainning
timer = d2l.Timer()
for X, y in trian_iter:
    continue
print(f'{timer.stop():2f} sec')

# make them together
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), 
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

trian_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in trian_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break