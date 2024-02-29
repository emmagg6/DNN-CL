import math
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import torch
from torch._C import dtype
# from torch.utils.data import dataloader
# from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split



#########################################################
################ Sequence expansion dataset #############
#########################################################


# dataset
# input X:  [a, b, c, d, *, *, *, *]
# output Y: [a, a, b, b, c, c, d, d]

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

# distractors in input
# def make_data_points(vocab_size, seq_len, copy_num, n_samples, device):
#     # return
#     X = torch.randint(0, vocab_size, size=[n_samples, seq_len], device=device)
#     Y = tile(X, dim=1, n_tile=copy_num, device=device)[:,:seq_len]
#     return X.squeeze_(), Y.squeeze_()

def make_data_points(vocab_size, seq_len, copy_num, n_samples, rng):
    # return
    
    num_fill = math.ceil(seq_len/copy_num)
    X1 = rng.integers(0, vocab_size-1, size=[n_samples, num_fill], dtype=np.int64)
    # X1 = torch.randint(0, vocab_size-1, size=[n_samples, num_fill], dtype=torch.long)
    X1 = torch.from_numpy(X1)
    X2 = torch.ones(size=[n_samples, seq_len - num_fill], dtype=torch.long)*(vocab_size-1)
    X = torch.cat([X1, X2], dim=1)
    Y = tile(X, dim=1, n_tile=copy_num)[:, :seq_len]
    return X.squeeze_(), Y.squeeze_()


class ExpandSeqTrainDataset(IterableDataset):
    def __init__(self, vocab_size, seq_len, copy_num, n_samples, rand_seed):
        super(ExpandSeqTrainDataset).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.copy_num = copy_num
        self.n_samples = n_samples
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)

    def __iter__(self):
        for i in range(self.n_samples):
            yield make_data_points(self.vocab_size, self.seq_len, self.copy_num, 1, self.rng)


class ExpandSeqValDataset(Dataset):
    def __init__(self, vocab_size, seq_len, copy_num, n_samples, rand_seed):
        super(ExpandSeqValDataset).__init__()
        self.n_samples = n_samples
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
        self.X, self.Y = make_data_points(vocab_size, seq_len, copy_num, n_samples, self.rng)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]


class ExpandSeqTestDataset(Dataset):
    def __init__(self, vocab_size, seq_len, copy_num, n_samples, rand_seed):
        super(ExpandSeqValDataset).__init__()
        self.n_samples = n_samples
        self.rand_seed = rand_seed
        self.rng = np.random.default_rng(self.rand_seed)
        self.X, self.Y = make_data_points(vocab_size, seq_len, copy_num, n_samples, self.rng)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]


class ExpandSequenceDataModule(pl.LightningDataModule):

    def __init__(self, seq_len: int, vocab_size: int, batch_size: int, n_train: int, n_val: int, n_test: int, random_seeds: dict):
        super().__init__()
        self.copy_num=2 # default 
        self.seq_len = seq_len
        self.vocab_size = vocab_size 
        self.batch_size = batch_size 
        self.random_seeds = random_seeds # "train", "val", "test"
        self.n_train=n_train
        self.n_val= n_val
        self.n_test = n_test 

    def setup(self, stage=None):
        ## Download and prepare data
        ## using self.random_seed 
        if stage == 'fit':
            self.train_set = ExpandSeqTrainDataset(self.vocab_size, self.seq_len, self.copy_num, self.n_train, self.random_seeds['train'])
            self.val_set = ExpandSeqValDataset(self.vocab_size, self.seq_len, self.copy_num, self.n_val, self.random_seeds['val'])
        if stage == 'test':
            self.test_set = ExpandSeqTestDataset(self.vocab_size, self.seq_len, self.copy_num, self.n_test, self.random_seeds['test'])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=1)


def test_expandsequence():
    dm = ExpandSequenceDataModule(
        seq_len=10,
        vocab_size=5,
        batch_size=20,
        n_train=100,
        n_val=100,
        n_test=100,
        random_seeds={'train':10, 'val':1, 'test':2})

    dm.setup('fit')
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    dm.setup('test')
    test_loader = dm.test_dataloader()

    for X, Y in train_loader:
        assert (X.size() == torch.Size([20, 10]))
        print(X[1,:])
        assert (Y.size() == torch.Size([20, 10]))
        # print(Y[1,:])
        break 

    for X, Y in val_loader:
        print(X[1,:])
        assert (X.size() == torch.Size([20, 10]))
        # print(Y[1,:])
        assert (Y.size() == torch.Size([20, 10]))
        break 
         
    for X, Y in test_loader:
        print(X[1,:])
        assert (X.size() == torch.Size([20, 10]))
        # print(Y[1,:])
        assert (Y.size() == torch.Size([20, 10]))



#########################################################
################ Copy memory dataset ####################
##########################################################


# def gen_a_copy_sample(delay, rng):
#     # X: {0,...,7} x 10 + {8} x T + {9} x 1 + {8} x 10
#     # Y: {8} x (T+10) + {0...7} x 10
#     # torch.randint(0, 8, size=[10], dtype=torch.long)
#     rand_seq_np = rng.integers(0, 8, size=[10], dtype=np.int64)
#     x = torch.cat([torch.from_numpy(rand_seq_np),
#                    torch.ones(delay-1, dtype=torch.long)*8,
#                    torch.tensor([9]),
#                    torch.ones(10, dtype=torch.long)*8])
#     y = torch.ones(delay+20, dtype=torch.long)*8
#     y[-10:] = x[:10]
#     return x, y

# def gen_n_copy_samples(delay: int, n: int, rng):
#     # torch.randint(0, 8, size=[n, 10], dtype=torch.long)
#     rand_seq_np = rng.integers(0, 8, size=[n, 10], dtype=np.int64)
#     x = torch.cat([torch.from_numpy(rand_seq_np),
#                    torch.ones([n, delay-1], dtype=torch.long)*8,
#                    torch.ones([n, 1], dtype=torch.long)*9,
#                    torch.ones([n, 10], dtype=torch.long)*8], dim=1)
#     y = torch.ones([n, delay+20], dtype=torch.long)*8
#     y[:, -10:] = x[:, :10]
#     #torch.save(x, f'data/copymem_{delay}_{rng}_10_x')
#     #torch.save(y, 'data/copymem_{delay}_{rng}_10_y')
#     return x, y


# class CopyTrainDataset(IterableDataset):
#     def __init__(self, delay, n_samples, rand_seed):
#         super(CopyTrainDataset).__init__()
#         self.delay = delay
#         self.n_samples = n_samples
#         self.rand_seed = rand_seed
#         self.rng = np.random.default_rng(self.rand_seed)

#     def __iter__(self):
#         for i in range(self.n_samples):
#             yield gen_a_copy_sample(self.delay, self.rng)

# class CopyValTestDataset(Dataset):
#     def __init__(self, delay, n_samples, rand_seed):
#         super(CopyValTestDataset).__init__()
#         self.delay = delay
#         self.n_samples = n_samples
#         self.rand_seed = rand_seed
#         self.rng = np.random.default_rng(self.rand_seed)
#         self.X, self.Y = gen_n_copy_samples(self.delay, self.n_samples, self.rng)

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         return self.X[idx, :], self.Y[idx, :]




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

class CopyMemoryDataModule(LightningDataModule):
    def __init__(self,  delay, rand, T, data_dir: str = 'data/'):
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

    def setup(self):
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




class NextChar(Dataset):
    def __init__(self, data_dir,file, seq_length):
        self.data  = open(data_dir+file, 'r').read()   #shakespare data
        self.chars = list(set(self.data))
        self.seq_length = seq_length
        self.data_size, self.vocab_size = len(self.data), len(self.chars)
        self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) } 
        self.ix_to_char = { i:ch for i,ch in enumerate(self.chars) }
        
    def __getitem__(self, index):
        x = [self.char_to_ix[ch] for ch in self.data[index:index+self.seq_length]] 
        y = [self.char_to_ix[ch] for ch in self.data[index+1:index+self.seq_length+1]] 
        return x, y

    def __len__(self):
        x = [self.char_to_ix[ch] for ch in self.data] 
        avail_idx = len(x)-self.seq_length +1
        return avail_idx


class NextCharDataModule(LightningDataModule):
    def __init__(self, seq_length, batch_size, data_dir: str = 'data/', file: str = 'input.txt'):
        super().__init__()
        self.seq_length = seq_length
        self.data_dir = data_dir
        self.file = file
        self.batch_size = batch_size
  
    def setup(self):
        nextchar_full = NextChar(self.data_dir,self.file, self.seq_length)
        train_size = int(len(nextchar_full)*0.8)
        valid_size = int(len(nextchar_full)*0.1)
        test_size = len(nextchar_full)-train_size-valid_size
        self.nextchar_train, self.nextchar_val, self.nextchar_test = random_split(nextchar_full, [train_size,valid_size,test_size])

    def train_dataloader(self):
        return DataLoader(self.nextchar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.nextchar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.nextchar_test, batch_size=self.batch_size)




# class CopyMemoryDataModule(pl.LightningDataModule):

#     def __init__(self, delay: int, batch_size: int, n_train: int, n_val: int, n_test: int, random_seeds: dict):
#         super().__init__()
#         self.delay = delay
#         self.batch_size = batch_size
#         self.n_train = n_train 
#         self.n_val = n_val
#         self.n_test = n_test 
#         self.random_seeds = random_seeds # 'train', 'val', 'test'

#     def setup(self, stage=None):
#         ## Download and prepare data
#         ## using self.random_seed 
#         if stage == 'fit':
#             self.train_set = CopyTrainDataset(delay=self.delay, n_samples=self.n_train, rand_seed=self.random_seeds['train'])
#             self.val_set = CopyValTestDataset(delay=self.delay, n_samples=self.n_val, rand_seed=self.random_seeds['val'])
#         if stage == 'test':
#             self.test_set = CopyValTestDataset(delay=self.delay, n_samples=self.n_test, rand_seed=self.random_seeds['test'])

#     def train_dataloader(self):
#         return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=1)

#     def val_dataloader(self):
#         return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=1)

#     def test_dataloader(self):
#         return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=1)





def test_copyemmory_dm():

    dm = CopyMemoryDataModule(delay=5, # total length is 20 + 5 = 25 
                              batch_size=20, 
                              n_train=100, 
                              n_val=100, 
                              n_test=100, 
                              random_seeds={
                                  'train': 123,
                                  'val': 123,
                                  'test': 3
                                  }
                              )
    dm.setup('fit') # prepare train and val dataset
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    dm.setup('test') # prepare test dataset 
    test_loader = dm.test_dataloader()

    for idx, (X, Y) in enumerate(train_loader):
        if idx == 2:
            break 
        assert (X.size() == torch.Size([20, 25]))
        assert (Y.size() == torch.Size([20, 25]))
        print(X[1,:])
        print()
        # print(Y[1,:])


    for idx, (X, Y) in enumerate(val_loader):
        if idx == 2:
            break 
        assert (X.size() == torch.Size([20, 25]))
        assert (Y.size() == torch.Size([20, 25]))
        print(X[1,:])
        print()
        # print(Y[1,:])
         
    for idx, (X, Y) in enumerate(test_loader):
        if idx == 2:
            break 
        assert (X.size() == torch.Size([20, 25]))
        assert (Y.size() == torch.Size([20, 25]))
        print(X[1,:])
        print()
        # print(Y[1,:])



