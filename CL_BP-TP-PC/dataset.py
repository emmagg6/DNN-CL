import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms


class MyClassification(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return feature, label



def make_MNIST(dim=None, test=False, pc = False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([60000, 784]), torch.empty([60000], dtype=torch.long)
    for i, t in enumerate(list(mnist_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]

    mnist_test = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 784]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(mnist_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]

    if test:
            trainset = MyClassification(train_x, train_y)
            validset = MyClassification(test_x, test_y)
    else:
            trainset = MyClassification(train_x[:55000], train_y[:55000])
            validset = MyClassification(train_x[55000:], train_y[55000:])

    testset = MyClassification(test_x, test_y)

    if pc :
        return mnist_train, mnist_test
    else:
        return trainset, validset, testset


def make_FashionMNIST(dim=None, test=False, pc = False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    fashion_train = tv.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
    train_x, train_y = torch.empty([60000, 784]), torch.empty([60000], dtype=torch.long)
    for i, t in enumerate(list(fashion_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]

    fashion_test = tv.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 784]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(fashion_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]

    if test:
            trainset = MyClassification(train_x, train_y)
            validset = MyClassification(test_x, test_y)
    else:
            trainset = MyClassification(train_x[:55000], train_y[:55000])
            validset = MyClassification(train_x[55000:], train_y[55000:])

    testset = MyClassification(test_x, test_y)

    if pc :
        return fashion_train, fashion_test
    else:
        return trainset, validset, testset


def make_CIFAR10(dim=None, test=False, pc = False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    cifar_train = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([50000, 3072]), torch.empty([50000], dtype=torch.long)
    for i, t in enumerate(list(cifar_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]

    cifar_test = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 3072]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(cifar_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]

    if test:
            trainset = MyClassification(train_x, train_y)
            validset = MyClassification(test_x, test_y)
    else:
            trainset = MyClassification(train_x[:45000], train_y[:45000])
            validset = MyClassification(train_x[45000:], train_y[45000:])

    testset = MyClassification(test_x, test_y)

    if pc :
        return cifar_train, cifar_test
    else:
        return trainset, validset, testset
    


def make_STL10(dim=None, test=False):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize images to 28x28 -- matching cifar10
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalization values for STL-10
    ])

    stl_train = tv.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    num_train = len(stl_train)
    train_x, train_y = torch.empty([num_train, 2352]), torch.empty([num_train], dtype=torch.long)
    for i, t in enumerate(list(stl_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]

    stl_test = tv.datasets.STL10(root='./data', split='test', download=True, transform=transform)
    num_test = len(stl_test)
    test_x, test_y = torch.empty([num_test, 2352]), torch.empty([num_test], dtype=torch.long)
    for i, t in enumerate(list(stl_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]

    if test:
            trainset = MyClassification(train_x, train_y)
            validset = MyClassification(test_x, test_y)
    else:
            trainset = MyClassification(train_x[:int(0.9*num_train)], train_y[:int(0.9*num_train)])
            validset = MyClassification(train_x[int(0.9*num_train):], train_y[int(0.9*num_train):])

    testset = MyClassification(test_x, test_y)

    if pc :
        return stl_train, stl_test
    else:
        return trainset, validset, testset




def make_CIFAR100(dim=None, test=False, pc = False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    cifar_train = tv.datasets.CIFAR100(root='./data', train=True,
                                       download=True, transform=transform)
    train_x, train_y = torch.empty([50000, 3072]), torch.empty([50000], dtype=torch.long)
    for i, t in enumerate(list(cifar_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]

    cifar_test = tv.datasets.CIFAR100(root='./data', train=False,
                                      download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 3072]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(cifar_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]

    if test:
            trainset = MyClassification(train_x, train_y)
            validset = MyClassification(test_x, test_y)
    else:
            trainset = MyClassification(train_x[:45000], train_y[:45000])
            validset = MyClassification(train_x[45000:], train_y[45000:])

    testset = MyClassification(test_x, test_y)

    if pc :
        return cifar_train, cifar_test
    else:
        return trainset, validset, testset
