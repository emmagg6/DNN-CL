import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
import matplotlib.pyplot as plt
import subprocess
import argparse
from datetime import datetime
from utils import *
# from pc_layers import ConvLayer, MaxPool, AvgPool, ProjectionLayer


class pc_net(object):
  def __init__(self, layers, n_inference_steps_train, inference_learning_rate, loss_fn, loss_fn_deriv, device='cpu',numerical_check=False):
    self.layers= layers
    self.n_inference_steps_train = n_inference_steps_train
    self.inference_learning_rate = inference_learning_rate
    self.device = device
    self.loss_fn = loss_fn
    self.loss_fn_deriv = loss_fn_deriv
    self.L = len(self.layers)
    self.outs = [[] for i in  range(self.L+1)]
    self.prediction_errors = [[] for i in range(self.L+1)]
    self.predictions = [[] for i in range(self.L+1)]
    self.mus = [[] for i in range(self.L+1)]
    self.numerical_check = numerical_check
    if self.numerical_check:
      print("Numerical Check Activated!")
      for l in self.layers:
        l.set_weight_parameters()

  def update_weights(self,print_weight_grads=False,get_errors=False):
    weight_diffs = []
    for (i,l) in enumerate(self.layers):
      if i !=1:
        if self.numerical_check:
            true_weight_grad = l.get_true_weight_grad().clone()
        dW = l.update_weights(self.prediction_errors[i+1],update_weights=True)
        true_dW = l.update_weights(self.predictions[i+1],update_weights=True)
        diff = torch.sum((dW -true_dW)**2).item()
        weight_diffs.append(diff)
        if print_weight_grads:
          print("weight grads : ", i)
          print("dW: ", dW*2)
          print("true diffs: ", true_dW * 2)
          if self.numerical_check:
            print("true weights ", true_weight_grad)
    return weight_diffs


  def forward(self,x):
    for i,l in enumerate(self.layers):
      x = l.forward(x)
    return x

  def no_grad_forward(self,x):
    with torch.no_grad():
      for i,l in enumerate(self.layers):
        x = l.forward(x)
      return x

  def infer(self, inp, label, n_inference_steps=None):
    self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
    with torch.no_grad():
      self.mus[0] = inp.clone()
      self.outs[0] = inp.clone()
      for i, l in enumerate(self.layers):
        #initialize mus with forward predictions
        # print("i is: ", i, " mus[i] shape: ", self.mus[i].shape)
        self.mus[i+1] = l.forward(self.mus[i])
        self.outs[i+1] = self.mus[i+1].clone()
      self.mus[-1] = label.clone() #setup final label
      self.prediction_errors[-1] = -self.loss_fn_deriv(self.outs[-1], self.mus[-1])#self.mus[-1] - self.outs[-1] #setup final prediction errors
      self.predictions[-1] = self.prediction_errors[-1].clone()
      for n in range(self.n_inference_steps_train):
      #reversed inference
        for j in reversed(range(len(self.layers))):
          if j != 0:
            self.prediction_errors[j] = self.mus[j] - self.outs[j]
            self.predictions[j] = self.layers[j].backward(self.prediction_errors[j+1])
            dx_l = self.prediction_errors[j] - self.predictions[j]
            self.mus[j] -= self.inference_learning_rate * (2*dx_l)
      #update weights
      weight_diffs = self.update_weights()
      #get loss:
      L = self.loss_fn(self.outs[-1],self.mus[-1]).item()#torch.sum(self.prediction_errors[-1]**2).item()
      #get accuracy
      acc = accuracy(self.no_grad_forward(inp),label)
      return L, acc, weight_diffs
    
  def test_accuracy(self,testset, num_classes=10):
    accs = []
    for i,(inp, label) in enumerate(testset):
        pred_y = self.no_grad_forward(inp.to(DEVICE))
        acc =accuracy(pred_y,onehot(label, num_classes).to(DEVICE)) # (emmagg6) added num_classes for the classitifcation task
        accs.append(acc)
    return np.mean(np.array(accs)),accs

  def train(self, dataset, testset, n_epochs, n_inference_steps, logdir, savedir, old_savedir, save_every=1, print_every=10, num_classes = 10, trial = 0, log = False):
    if old_savedir != "None":
      self.load_model(old_savedir)
    losses = []
    accs = []
    weight_diffs_list = []
    test_accs = []
    # Save initial checkpoint at epoch 0
    print("Epoch: 0")
    epoch = 0
    mean_acc, acclist = self.test_accuracy(dataset)
    accs.append(mean_acc)
    # mean_loss = 0
    # losses.append(mean_loss)
    mean_test_acc, _ = self.test_accuracy(testset)
    test_accs.append(mean_test_acc)
    # weight_diffs_list.append(weight_diffs)
    print("TEST ACCURACY: ", mean_test_acc)
    if log:
        wandb.log({
            "epoch": epoch,
            # "train loss": mean_loss,
            "train accuracy": mean_acc,
            "valid accuracy": mean_test_acc
        })

    for epoch in range(1, n_epochs + 1):
      losslist = []
      print("Epoch: ", epoch)
      for i, (inp, label) in enumerate(dataset):
        if self.loss_fn != cross_entropy_loss:
        #   label = onehot(label).to(DEVICE)
          label = onehot(label, num_classes).to(DEVICE)
            # label = torch.nn.functional.one_hot(label, num_classes=10).to(DEVICE)
        else:
          label = label.long().to(DEVICE)
        L, acc, weight_diffs = self.infer(inp.to(DEVICE), label)
        losslist.append(L)
      mean_acc, acclist = self.test_accuracy(dataset)
      accs.append(mean_acc)
      mean_loss = np.mean(np.array(losslist))
      losses.append(mean_loss)
      mean_test_acc, _ = self.test_accuracy(testset)
      test_accs.append(mean_test_acc)
      weight_diffs_list.append(weight_diffs)
    #   print("TRAIN ACCURACY:", )
      print("TEST ACCURACY: ", mean_test_acc)
      if log:
            wandb.log({
                "epoch": epoch,
                # "train loss": mean_loss,
                "train accuracy": mean_acc,
                "valid accuracy": mean_test_acc
            })
    print("SAVING MODEL")
    self.save_model(logdir, savedir, losses, accs, weight_diffs_list, test_accs)

  def save_model(self,savedir,logdir,losses,accs,weight_diffs_list,test_accs):
      for i,l in enumerate(self.layers):
          l.save_layer(logdir,i)
      np.save(logdir +"/losses.npy",np.array(losses))
      np.save(logdir+"/accs.npy",np.array(accs))
      np.save(logdir+"/weight_diffs.npy",np.array(weight_diffs_list))
      np.save(logdir+"/test_accs.npy",np.array(test_accs))
      subprocess.call(['rsync','--archive','--update','--compress','--progress',str(logdir) +"/",str(savedir)])
      print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
      now = datetime.now()
      current_time = str(now.strftime("%H:%M:%S"))
      subprocess.call(['echo','saved at time: ' + str(current_time)])

  def load_model(self,old_savedir):
      for (i,l) in enumerate(self.layers):
          l.load_layer(old_savedir,i)
