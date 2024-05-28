import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian

import math
import os
import time
import matplotlib.pyplot as plt
import subprocess
import argparse
from datetime import datetime
from utils import *
from Models.PCDTP.pcdtp_layers import ConvLayer_dtp, MaxPool_dtp, AvgPool_dtp, ProjectionLayer_dtp

torch.autograd.set_detect_anomaly(True)

class pcdtp_net(object):
    def __init__(self, layers, n_inference_steps_train, inference_learning_rate, loss_fn, loss_fn_deriv, device='cpu', numerical_check=False):
        self.layers = layers
        self.n_inference_steps_train = n_inference_steps_train
        self.inference_learning_rate = inference_learning_rate
        self.device = device
        self.loss_fn = loss_fn
        self.loss_fn_deriv = loss_fn_deriv
        self.L = len(self.layers)
        self.outs = [[] for i in range(self.L + 1)]
        self.prediction_errors = [[] for i in range(self.L + 1)]
        self.predictions = [[] for i in range(self.L + 1)]
        self.mus = [[] for i in range(self.L + 1)]
        self.numerical_check = numerical_check
        if self.numerical_check:
            print("Numerical Check Activated!")
            for l in self.layers:
                l.set_weight_parameters()

    # def compute_targets(self, inp, label, stepsize):
    #     self.forward(inp)
    #     loss = self.loss_fn(self.outs[-1], label)
        
    #     # # Assuming custom layers do not need zero_grad
    #     for d in range(self.L):
    #         self.layers[d].zero_grad()
    #     # loss.backward(retain_graph=True)

    #     with torch.no_grad():
    #         for d in range(self.L - 1, -1, -1):
    #             print(f"Layer {d}")
    #             if d == self.L - 1:
    #                 print("\n\nLast layer\n\n")
    #                 self.outs[d + 1].retain_grad()
    #                 self.layers[d].target = self.outs[d + 1] - stepsize * self.outs[d + 1].grad
    #             else:
    #                 plane = self.layers[d + 1].backward_function_1.forward(self.layers[d + 1].target)
    #                 diff = self.layers[d + 1].backward_function_2.forward(plane, self.layers[d].output)
    #                 self.layers[d].target = diff


    def compute_targets(self, inp, label, stepsize):
        self.forward(inp)
        loss = self.loss_fn(self.outs[-1], label)
        
        # Zero gradients for all layers
        for d in range(self.L):
            self.layers[d].zero_grad()

        # Perform backward pass to compute gradients
        loss.backward(retain_graph=True)

        with torch.no_grad():
            for d in range(self.L - 1, -1, -1):
                 print(f"\nLayer {d}")
                 self.layers[d].target = self.outs[d + 1] - stepsize * self.outs[d + 1].grad
            for d in reversed(range(self.L -1, -1, -1)):
                print(f"\nLayer {d} -- reversed")
                plane = self.layers[d + 1].backward_function_1.forward(self.layers[d + 1].target)
                diff = self.layers[d + 1].backward_function_2.forward(plane, self.layers[d].output)
                self.layers[d].target = diff

    def update_weights(self, print_weight_grads=False):
        for i, l in enumerate(self.layers):
            if i != 1:
                l.update_weights(self.layers[i].target, update_weights=True)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            # Reshape input if necessary before passing to convolutional layers
            if isinstance(l, ConvLayer_dtp):
                if x.dim() == 2:
                    batch_size, num_features = x.shape
                    side_length = int(num_features ** 0.5)
                    x = x.view(batch_size, 1, side_length, side_length)
            x = l.forward(x)
        return x

    def no_grad_forward(self, x):
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                # Reshape input if necessary before passing to convolutional layers
                if isinstance(l, ConvLayer_dtp):
                    if x.dim() == 2:
                        batch_size, num_features = x.shape
                        side_length = int(num_features ** 0.5)
                        x = x.view(batch_size, 1, side_length, side_length)
                x = l.forward(x)
            return x

    def infer(self, inp, label, n_inference_steps=None):
        self.n_inference_steps_train = n_inference_steps if n_inference_steps is not None else self.n_inference_steps_train
        self.mus[0] = inp.clone().detach().requires_grad_(True)  # Ensure input requires grad
        self.outs[0] = inp.clone()
        
        for i, l in enumerate(self.layers):
            # Reshape input if necessary before passing to convolutional layers
            if isinstance(l, ConvLayer_dtp):
                if self.mus[i].dim() == 2:
                    batch_size, num_features = self.mus[i].shape
                    side_length = int(num_features ** 0.5)
                    self.mus[i] = self.mus[i].view(batch_size, 1, side_length, side_length)
            self.mus[i + 1] = l.forward(self.mus[i])
            self.outs[i + 1] = self.mus[i + 1].clone()
        self.mus[-1] = label.clone()

        # Compute initial prediction errors
        self.prediction_errors[-1] = -self.loss_fn_deriv(self.outs[-1], self.mus[-1])
        self.predictions[-1] = self.prediction_errors[-1].clone()

        for n in range(self.n_inference_steps_train):
            for j in reversed(range(len(self.layers))):
                if j != 0:
                    self.prediction_errors[j] = self.mus[j] - self.outs[j]
                    self.predictions[j] = self.layers[j].backward(self.prediction_errors[j + 1])
                    dx_l = self.prediction_errors[j] - self.predictions[j]
                    self.mus[j] -= self.inference_learning_rate * (2 * dx_l)

        # Compute targets and update weights using the predictions
        self.compute_targets(inp, label, stepsize=self.inference_learning_rate)
        weight_diffs = self.update_weights()
        # Get loss
        L = self.loss_fn(self.outs[-1], self.mus[-1]).item()
        # Get accuracy
        acc = accuracy(self.no_grad_forward(inp), label)
        return L, acc, weight_diffs

    def test_accuracy(self, testset, num_classes=10):
        accs = []
        for i, (inp, label) in enumerate(testset):
            # Debug statement to print input shape
            print(f"Test batch input shape: {inp.shape}")
            pred_y = self.no_grad_forward(inp.to(self.device))
            acc = accuracy(pred_y, onehot(label, num_classes).to(self.device))
            accs.append(acc)
        return np.mean(np.array(accs)), accs

    def train(self, dataset, testset, n_epochs, n_inference_steps, logdir, savedir, old_savedir, save_every=1, print_every=10, num_classes=10, trial=0, log=False, epochs_backward=5):
        if old_savedir != "None":
            self.load_model(old_savedir)
        losses = []
        accs = []
        weight_diffs_list = []
        test_accs = []

        for epoch in range(1, n_epochs + 1):
            losslist = []
            print("Epoch: ", epoch)
            for i, (inp, label) in enumerate(dataset):
                if self.loss_fn != cross_entropy_loss:
                    label = onehot(label, num_classes).to(self.device)
                else:
                    label = label.long().to(self.device)
                # Debug statement to print input shape
                print(f"Train batch input shape: {inp.shape}")
                L, acc, weight_diffs = self.infer(inp.to(self.device), label)
                losslist.append(L)
            mean_acc, acclist = self.test_accuracy(dataset)
            accs.append(mean_acc)
            mean_loss = np.mean(np.array(losslist))
            losses.append(mean_loss)
            mean_test_acc, _ = self.test_accuracy(testset)
            test_accs.append(mean_test_acc)
            weight_diffs_list.append(weight_diffs)
            print("TEST ACCURACY: ", mean_test_acc)
            if log:
                wandb.log({
                    "epoch": epoch,
                    "train accuracy": mean_acc,
                    "valid accuracy": mean_test_acc
                })

        print("SAVING MODEL")
        self.save_model(logdir, savedir, losses, accs, weight_diffs_list, test_accs)

    def save_model(self, savedir, logdir, losses, accs, weight_diffs_list, test_accs):
        for i, l in enumerate(self.layers):
            l.save_layer(logdir, i)
        np.save(logdir + "/losses.npy", np.array(losses))
        np.save(logdir + "/accs.npy", np.array(accs))
        np.save(logdir + "/weight_diffs.npy", np.array(weight_diffs_list))
        np.save(logdir + "/test_accs.npy", np.array(test_accs))
        subprocess.call(['rsync', '--archive', '--update', '--compress', '--progress', str(logdir) + "/", str(savedir)])
        print("Rsynced files from: " + str(logdir) + "/ " + " to" + str(savedir))
        now = datetime.now()
        current_time = str(now.strftime("%H:%M:%S"))
        subprocess.call(['echo', 'saved at time: ' + str(current_time)])

    def load_model(self, old_savedir):
        for (i, l) in enumerate(self.layers):
            l.load_layer(old_savedir, i)