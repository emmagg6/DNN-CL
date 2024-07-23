'''
Adapted from 
@article{rosenbaum2022relationship,
  title={On the relationship between predictive coding and backpropagation},
  author={Rosenbaum, Robert},
  journal={Plos one},
  volume={17},
  number={3},
  pages={e0266102},
  year={2022},
  publisher={Public Library of Science}
}

by emmagg6 for a PC network class and continual learning support
'''


import torch # We no longer import as tch
import torch.nn as nn
import numpy as np
import torchvision # Contains data sets and functions for image processing
import torchvision.transforms as transforms # Contains MNIST, other image datasets, and image processing functions
import matplotlib.pyplot as plt
from time import time as tm
import torch.nn.functional as F
from copy import deepcopy
# import seaborn as sns

import wandb
import os

# Import TorchSeq2PC from https://github.com/RobertRosenbaum/Torch2PC.git
from Models.PC.Torch2PC import PCInfer

from utils import ToOneHot


class pc_cnn(object):
  def __init__(self, layers, device='cpu'):
    self.layers= layers


    self.grads_rel_diff = []
    self.grads_cos_sim = []
    self.grads_angle = []

    self.CosSim = nn.CosineSimilarity(dim=0, eps=1e-8)

  def RelDiff(self, x,y):
    return np.linalg.norm(x-y)/np.linalg.norm(y)

  def corr2(self, x,y):
    c=np.corrcoef(x,y)
    return c[0,1]




  def train(self, epochs, inference_steps, train_loader, test_loader, loss_fcn, 
            lr = 0.001, batch_size = 64, eta = 0.1, n = 20, device='cpu', log = False, 
            save = False, new_ckpt= '', train_ckpts = '', prev_ckpt = 'None'):
    
    self.opt = torch.optim.Adam(self.layers.parameters(), lr=lr)
    
    if prev_ckpt != 'None':
       load_state(prev_ckpt, lr)

    j, jj = 0, 0     # Counter to keep track of iterations
    t1 = tm() # Start the timer

    N = epochs * inference_steps
    self.train_losses = np.zeros(N)
    self.train_accs =  np.zeros(N)
    self.test_losses = np.zeros(N)
    self.test_accs = np.zeros(N)

    for k in range(epochs):

      # Re-initializes the training iterator (shuffles data for one epoch)
      TrainingIterator=iter(train_loader)

      for i in range(inference_steps): # For each batch

        # Get one batch of training data, reshape it
        # and send it to the current device
        X,Y=next(TrainingIterator)
        X=X.to(device)
        Y=ToOneHot(Y,10).to(device)

        _,Loss,_,_,_ = PCInfer(self.layers, loss_fcn, X, Y, "Strict" , eta, n)

        # if ComputeTrainingMetrics:
        #   modelBP=deepcopy(modelPC)   # Copy the model
        #   YhatBP = modelBP(X)         # Forward pass
        #   LossBP = LossFun(YhatBP, Y)
        #   LossBP.backward()       # Compute gradients
        for layer in range(len(self.layers)):
          gradsPC= self.layers[layer][0].weight.grad.cpu().detach().numpy()
          # modelBP  = deepcopy(self.layers)
          # gradsBP=modelBP[layer][0].weight.grad.cpu().detach().numpy()
          # self.grads_rel_diff[jj,layer]=self.RelDiff(gradsPC,gradsBP)
          # self.grads_cos_sim[jj,layer]=self.CosSim(torch.tensor(gradsPC.flatten()),torch.tensor(gradsBP.flatten())).item()
          # self.grads_rel_diff[jj,layer]=torch.acos(torch.tensor(self.grads_cos_sim[jj,layer])).item()
          # modelBP.zero_grad()

        # Update parameters
        self.opt.step()

        # Zero-out gradients
        self.layers.zero_grad()
        self.opt.zero_grad()

        # Print loss, store loss, compute test loss
        with torch.no_grad():
          if(i%100==0):
            Yhat = self.layers(X)
            # print('Epoch =',k,'i =',i,'L =',Loss.item(), 'A =', (torch.sum(torch.argmax(Y,axis=1)==torch.argmax(Yhat,axis=1))/batch_size).item())
          self.train_losses[jj]=Loss.item()

          Yhat = self.layers(X)
          self.train_accs[jj]=(torch.sum(torch.argmax(Y,axis=1)==torch.argmax(Yhat,axis=1))/batch_size).item()
          self.layers.eval()

          TestingIterator=iter(test_loader)
          Xtest,Ytest=next(TestingIterator)
          Xtest=Xtest.to(device)
          Ytest=ToOneHot(Ytest,10).to(device)
          YhatTest=self.layers(Xtest)
          self.test_losses[jj] = loss_fcn(YhatTest,Ytest).item()
          self.test_accs[jj]=(torch.sum(torch.argmax(Ytest,axis=1)==torch.argmax(YhatTest,axis=1))/batch_size).item()
          self.layers.train()
          jj+=1

      train_loss = np.mean(self.train_losses)
      train_acc = np.mean(self.train_accs)
      test_loss = np.mean(self.test_losses)
      test_acc = np.mean(self.test_accs)

      if log:
              wandb.log({
                  "epoch": k,
                  "train loss": train_loss,
                  "train accuracy": train_acc,
                  "valid loss": test_loss,
                  "valid accuracy": test_acc
              })
      print(f"Epoch: {k},  Train Acc: {train_acc}, Valid Acc: {test_acc}")

      tTrain=tm()-t1
      print('Training time = ',tTrain,'sec')

      if save == 'yes':
        self.save_model(new_ckpt)


    def save_model(self, new_ckpt):
        path = new_ckpt
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_state(self, path, lr):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        # Initialize the optimizer here with the current model parameters
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.opt = True
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])