import torch
import numpy as np
import math
from torch.utils.data import Dataset, IterableDataset, DataLoader


N_TRAINING = 2000000
N_VAL = 10000 
N_TEST = 10000 

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


def make_data_points(vocab_size, seq_len, copy_num, n_samples):
    # return
    num_fill = math.ceil(seq_len/copy_num)
    X1 = torch.randint(0, vocab_size-1, size=[n_samples, num_fill], dtype=torch.long)
    X2 = torch.ones(size=[n_samples, seq_len - num_fill], dtype=torch.long)*(vocab_size-1)
    X = torch.cat([X1, X2], dim=1)
    Y = tile(X, dim=1, n_tile=copy_num)[:, :seq_len]
    return X.squeeze_(), Y.squeeze_()


class ExpandSeqTrainDataset(IterableDataset):
    def __init__(self, vocab_size, seq_len, copy_num, n_samples):
        super(ExpandSeqTrainDataset).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.copy_num = copy_num
        self.n_samples = n_samples

    def __iter__(self):
        for i in range(self.n_samples):
            yield make_data_points(self.vocab_size, self.seq_len, self.copy_num, 1)


class ExpandSeqValDataset(Dataset):
    def __init__(self, vocab_size, seq_len, copy_num, n_samples):
        super(ExpandSeqValDataset).__init__()
        self.n_samples = n_samples
        self.X, self.Y = make_data_points(vocab_size, seq_len, copy_num, n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]


def expand_seq_dataset(
        vocab_size,  # including special blank character
        seq_len,
        copy_num,
        batch_size,
        n_train_samples,
        n_val_samples):

    train_set = ExpandSeqTrainDataset(vocab_size, seq_len, copy_num, n_train_samples)
    val_set = ExpandSeqValDataset(vocab_size, seq_len, copy_num, n_val_samples)

    # random batches
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader


def test_dataste():
    train_loader, val_loader = expand_seq_dataset(5, 10, 2, 20, 40, 20)
    for (X, Y) in train_loader:
        assert(torch.max(X) == 4)
        assert(torch.max(Y) == 3)
        assert(X.size() == torch.Size([20, 10]))  # batch_size * sequence length
        assert(X.dtype == torch.long)
        assert(Y.size() == torch.Size([20, 10]))
        assert(Y.dtype == torch.long)
    for (X, Y) in val_loader:
        assert(torch.max(X) == 4)
        assert(torch.max(Y) == 3)
        assert(X.size() == torch.Size([20, 10]))  # batch_size * sequence length
        assert(X.dtype == torch.long)
        assert(Y.size() == torch.Size([20, 10]))
        assert(Y.dtype == torch.long)


#########################################################
################ Copy memory dataset ####################
##########################################################

def gen_a_copy_sample(T):
    # X: {0,...,7} x 10 + {8} x T + {9} x 1 + {8} x 10
    # Y: {8} x (T+10) + {0...7} x 10
    x = torch.cat([torch.randint(0, 8, size=[10], dtype=torch.long),
                   torch.ones(T-1, dtype=torch.long)*8,
                   torch.tensor([9]),
                   torch.ones(10, dtype=torch.long)*8])
    y = torch.ones(T+20, dtype=torch.long)*8
    y[-10:] = x[:10]
    return x, y


def gen_n_copy_samples(T, n):

    x = torch.cat([torch.randint(0, 8, size=[n, 10], dtype=torch.long),
                   torch.ones([n, T-1], dtype=torch.long)*8,
                   torch.ones([n, 1], dtype=torch.long)*9,
                   torch.ones([n, 10], dtype=torch.long)*8], dim=1)

    y = torch.ones([n, T+20], dtype=torch.long)*8
    y[:, -10:] = x[:, :10]
    return x, y


class CopyDataset(IterableDataset):
    def __init__(self, T, n_samples):
        super(CopyDataset).__init__()
        self.T = T
        self.n_samples = n_samples

    def __iter__(self):
        for i in range(self.n_samples):
            yield gen_a_copy_sample(self.T)


class CopyDatasetValidation(Dataset):
    def __init__(self, T, n_samples):
        super(CopyDatasetValidation).__init__()
        self.T = T
        self.n_samples = n_samples
        self.X, self.Y = gen_n_copy_samples(self.T, self.n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :]


def prepare_copy_datasets(T, n_train, n_val, batch_size):
    train_set = CopyDataset(T, n_train)
    val_set = CopyDatasetValidation(T, n_val)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader


# def test_copymemory_dataset():
#     seq_len = 10
#     num_train = 20
#     num_val = 5
#     batch_sz = 5
#     devce = "cuda:0"
    
#     train_loader, val_loader = prepare_copy_datasets(seq_len, num_train, \
#         num_val, batch_sz, devce)

#     for (X, Y) in train_loader:
#         assert (X.size() == torch.Size([5, 30])) # X: {0-7} * 10 + 8 * (T-1) + 9 + {0-7} * 1-
#         assert (Y.size() == torch.Size([5, 30])) # Y: 8 * (10 + T) + {0-7} * 10 



def prepare_loaders(task, seq_len, batch_size, n_training=N_TRAINING, n_val=N_VAL):
    if task == 'expandsequence':
        vocab_dim = 5
        return expand_seq_dataset(vocab_dim, seq_len, 2, batch_size, n_training, n_val)

    if task == 'copymemory':
        return prepare_copy_datasets(seq_len, n_training, n_val, batch_size)


def test_prepare_loaders():
    train_loader, val_loader = prepare_loaders('copymemory', 15, 20) 

    for (X, Y) in train_loader:
        print(X.shape) # batch_size * sequence_len, torch.int64, device=cpu 
        print(X.dtype)
        print(X.device)
        print(Y.shape) # batch_size * sequence_len, torch.int64, device=cpu 
        print(Y.dtype)
        break 
    for (X, Y) in val_loader:
        print(X.shape)
        print(X.dtype)
        print(Y.shape)
        print(Y.dtype)
        break       

    