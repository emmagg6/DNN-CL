import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

# Note - you must have torchvision installed for this example
# from torchvision.datasets import MNIST
from torchvision import transforms
import torch


class CopyMem(Dataset):
    def __init__(self, data_dir,delay, rand, T):
        self.x = torch.load(data_dir+f'copymem_{delay}_{rand}_{T}_x')
        self.y = torch.load(data_dir+f'copymem_{delay}_{rand}_{T}_y')
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

class CopyMemDataModule(LightningDataModule):
    def __init__(self, data_dir: str = 'data/', delay, rand, T):
        super().__init__()
        self.data_dir = data_dir
        self.delay = delay
        self.rand = rand
        self.T =T
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        
        self.dims = (T*2+delay)

    # def prepare_data(self):
    #     # fit into dataloader
    #     return CopyMem(self.data_dir,self.delay, self.rand, self.T)
    #     # MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        copymem_full = CopyMem(self.data_dir,self.delay, self.rand, self.T)
        self.copymem_train, self.copymem_val, self.copymem_test = random_split(copymem_full, [1400000,300000,300000])
        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        #     copymem_full = CopyMem(self.data_dir,self.delay, self.rand, self.T)
        #     # MNIST(self.data_dir, train=True, transform=self.transform)
        #     self.copymem_train, self.copymem_val = random_split(copymem_full, [55000, 5000])

        #     # Optionally...
        #     # self.dims = tuple(self.mnist_train[0][0].shape)

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.copymem_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.copymem_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.copymem_test, batch_size=32)




class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)