
# from re import I
# import torch
# import numpy as np

# import os
# from datetime import datetime

# import neptune
# import torch
# from torch import nn
# from torch.nn.modules.loss import BCEWithLogitsLoss
# from torch import optim
# import optuna
# import neptunecontrib.monitoring.optuna as opt_utils
# import argparse
# from torch.nn.functional import one_hot
# import pytorch_lightning as pl
# from attrdict import AttrDict
# from torch.utils.data import DataLoader
# import utils
# import datasets


# # global constants fixed for all experiments 
# N_TRAIN_SAMPLES = 2000000
# N_VAL_SAMPLES = 10000
# N_TEST_SAMPLES = 10000 



# class RNN(nn.Module):
#     def __init__(self, 
#                  in_dim, 
#                  hidden_dim, 
#                  out_dim, 
#                  actv, 
#                  lr: float = 0.0001,
#                  b1: float = 0.9,
#                  b2: float = 0.999,
#                  batch_size: int = 20,
#                  **kwargs):
#         super(RNN, self).__init__()
#         self.save_hyperparameters()

#         # model architecture
#         if actv == 'tanh':
#             self.act_fn = torch.nn.Tanh()
#         elif actv == 'relu':
#             self.act_fn = torch.nn.ReLU()
#         else:
#             print("please set the actv in model")
#             exit()

#         self.num_workers = 4 

#         # network
#         # learnable parameters
#         self.W_hh = nn.Parameter(utils.rand_ortho((self.hparams.hidden_dim, self.hparams.hidden_dim),
#                 np.sqrt(6. / (self.hparams.hidden_dim + self.hparams.hidden_dim))))
#         self.W_xh = nn.Parameter(torch.empty(self.hparams.in_dim, self.hparams.hidden_dim))
#         torch.nn.init.normal_(self.W_xh, 0, 0.1)
#         self.W_hy = nn.Parameter(torch.empty(self.hparams.hidden_dim, self.hparams.out_dim))
#         torch.nn.init.normal_(self.W_hy, 0, 0.1)
#         self.b_h = nn.Parameter(torch.zeros([self.hparams.hidden_dim]))
#         self.b_y = nn.Parameter(torch.zeros([self.hparams.out_dim]))

#         # loss 
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
#         self.mse_loss = torch.nn.MSELoss()

#     def prepare_data(self):
#         pass 
#         # download, stuff 

#     def setup(self, stage=None):
#         if stage == 'fit' or stage is None:
#             # self.train_set = 
#             # self.val_set = 
#             pass 
#         if stage == 'test' or stage is None:
#             # self.test_set = 

#     def train_dataloader(self):
#         # return a dataloader
#         return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

#     def val_dataloader(self):
#         # return a dataloader
#         return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

#     def test_dataloader(self):
#         # return a dataloader
#         return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)



# def main():
    
#     parser = argparse.ArgumentParser(description="Training a simple RNN model")
#     parser.add_argument('--task', help='expandsequence or copymemory', required=True)
#     parser.add_argument('--seq_len', help='sequence length', type=int, required=True)
#     parser.add_argument('--method', help='backprop or targetprop', required=True)

#     # user input parameters
#     args = parser.parse_args()
#     if args.task == 'copymemory':
#         in_dim = 10 
#         out_dim = 10
#     elif args.taks == 'expandsequence':
#         in_dim = 5 
#         out_dim = 5 
#     else:
#         print("Plase specify the task --task='copymemory' | 'expandsequence'")
#         exit()

#     # hyperparameters to record and configure 
#     hparams = {
#         'task': args.task,
#         'seq_len': args.seq_len,
#         'method': args.method,
#         'in_dim': in_dim,
#         'out_dim': out_dim,
#         'hidden_dim': 128,
#         'num_workers': 4,
#         'actv': 'tanh',
#         'batch_size': 20,
#         'lr': 0.001,
#         'beta1': 0.9,
#         'beta2': 0.999,
#     }

#     # logg hyperparameters
#     # TODO 

#     hparams = AttrDict(hparams)

#     dm = None # ptl datamodule 

#     model = RNN(hparams.in_dim, 
#               hparams.hidden_dim,
#               hparams.out_dim, 
#               hparams.actv, 
#               hparams.lr, 
#               hparams.beta1,
#               hparams.beta2,
#               hparams.batch_size) 

#     trainer = pl.Trainer(gpus=2, max_epoch=1, progress_bar_refresh_rate=20) 

#     trainer.fit(model, dm)

    
    
    

    
    


    

    
    
    