'''
Based on the code used for the experiments conducted in the submitted paper 
"Fixed-Weight Difference Target Propagation" by K. K. S. Wu, K. C. K. Lee, and T. Poggio.

Adaptation for runs of Continual Learning by emmagg6.

'''


from Models.TP.tp_layers import tp_layer
from Models.TP.net import net
from Models.TP.tp_fcns import parameterized_function
from utils import calc_angle
from copy import deepcopy

import sys
import time
import wandb
import numpy as np 
import torch
from torch import nn
from torch.autograd.functional import jacobian
import os
import json


class tp_net(net):
    def __init__(self, depth, direct_depth, in_dim, hid_dim, out_dim, loss_function, device, params=None):
        self.depth = depth
        self.direct_depth = direct_depth
        self.loss_function = loss_function
        self.device = device
        self.MSELoss = nn.MSELoss(reduction="sum")
        self.layers = self.init_layers(in_dim, hid_dim, out_dim, params)
        self.back_trainable = (params["bf1"]["type"] == "parameterized")

    def init_layers(self, in_dim, hid_dim, out_dim, params):
        layers = [None] * self.depth
        dims = [in_dim] + [hid_dim] * (self.depth - 1) + [out_dim]
        for d in range(self.depth - 1):
            # print(f"Layer {d}: {dims[d]} -> {dims[d + 1]}")
            # print(f"ff1: {params['ff1']}")
            layers[d] = tp_layer(dims[d], dims[d + 1], self.device, params)
        params_last = deepcopy(params)
        params_last["ff2"]["act"] = params["last"]
        layers[-1] = tp_layer(dims[-2], dims[-1], self.device, params_last)
        return layers

    def forward(self, x, update=True):
        y = x
        for d in range(self.depth):
            y = self.layers[d].forward(y, update=update)
        return y

    def train(self, train_loader, valid_loader, epochs, lr, lrb, std, stepsize, log, save, hyperparams=None,
              trial = 0, new_ckpt = '', train_ckpts = ''):
        
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        eps = []

        # Pre-train the feedback weights
        for e in range(hyperparams["epochs_backward"]):
            torch.cuda.empty_cache()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.train_back_weights(x, y, lrb, std, loss_type=hyperparams["loss_feedback"])

        # Train the feedforward and feedback weights
        for e in range(epochs + 1):
            print(f"Epoch: {e}")
            torch.cuda.empty_cache()
            start_time = time.time()
            if e > 0:
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    # Train feedback weights
                    for i in range(hyperparams["epochs_backward"]):
                        self.train_back_weights(x, y, lrb, std, loss_type=hyperparams["loss_feedback"])
                    # Train forward weights
                    self.compute_target(x, y, stepsize)
                    self.update_weights(x, lr)
            end_time = time.time()

            # Compute Positive semi-definiteness (the strict condition) and Trace (the weak condition)
            eigenvalues_ratio = [torch.zeros(1, device=self.device) for d in range(self.depth)]
            eigenvalues_trace = [torch.zeros(1, device=self.device) for d in range(self.depth)]
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    self.forward(x)
                    for d in range(1, self.depth - self.direct_depth + 1):
                        h1 = self.layers[d].input[0]
                        gradf = jacobian(self.layers[d].forward, h1)
                        h2 = self.layers[d].forward(h1)
                        gradg = jacobian(self.layers[d].backward_function_1.forward, h2)
                        eig, _ = torch.linalg.eig(gradf @ gradg)
                        eigenvalues_ratio[d] += (eig.real > 0).sum() / len(eig.real)
                        eigenvalues_trace[d] += torch.trace(gradf @ gradg)
            for d in range(self.depth):
                eigenvalues_ratio[d] /= len(valid_loader)
                eigenvalues_trace[d] /= len(valid_loader)

            # Predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)


            train_losses.append(train_loss.item())
            train_accuracies.append(train_acc)
            test_losses.append(valid_loss.item())
            test_accuracies.append(valid_acc)
            eps.append(e)

            # Logging
            if log:
                log_dict = {}
                log_dict["train loss"] = train_loss
                log_dict["valid loss"] = valid_loss
                if train_acc is not None:
                    log_dict["train accuracy"] = train_acc
                if valid_acc is not None:
                    log_dict["valid accuracy"] = valid_acc
                log_dict["time"] = end_time - start_time
                for d in range(1, self.depth - self.direct_depth + 1):
                    log_dict[f"eigenvalue ratio {d}"] = eigenvalues_ratio[d].item()
                    log_dict[f"eigenvalue trace {d}"] = eigenvalues_trace[d].item()

                wandb.log(log_dict)

            else:
                if train_acc is not None:
                    print(f"\tTrain Acc        : {train_acc}")
            if valid_acc is not None:
                    print(f"\tValid Acc        : {valid_acc}")

        if save == 'yes':
            self.save_model(new_ckpt)
            self.save_training_dynamics(train_losses, train_accuracies, test_losses, test_accuracies, trial, train_ckpts)
        

    def train_back_weights(self, x, y, lrb, std, loss_type="DTP"):
        if not self.back_trainable:
            return
        # print("loss_type: ", loss_type)
        self.forward(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            if loss_type == "DTP":
                q = self.layers[d - 1].output.detach().clone()
                q = q + torch.normal(0, std, size=q.shape, device=self.device)
                q_upper = self.layers[d].forward(q)
                h = self.layers[d].backward_function_1.forward(q_upper)
                loss = self.MSELoss(h, q)
            else:
                raise NotImplementedError()
            self.layers[d].zero_grad()
            loss.backward(retain_graph=True)
            self.layers[d].update_backward(lrb / len(x))

    def compute_target(self, x, y, stepsize):
        y_pred = self.forward(x)
        loss = self.loss_function(y_pred, y)
        for d in range(self.depth):
            self.layers[d].zero_grad()
        loss.backward(retain_graph=True)

        with torch.no_grad():
            for d in range(self.depth - self.direct_depth, self.depth):
                self.layers[d].target = self.layers[d].output - \
                    stepsize * self.layers[d].output.grad

            for d in reversed(range(self.depth - self.direct_depth)):
                plane = self.layers[d + 1].backward_function_1.forward(self.layers[d + 1].target)
                diff = self.layers[d + 1].backward_function_2.forward(plane, self.layers[d].output)
                self.layers[d].target = diff

    def update_weights(self, x, lr):
        self.forward(x)
        for d in range(self.depth):
            loss = self.MSELoss(self.layers[d].target, self.layers[d].output)
            self.layers[d].zero_grad()
            loss.backward(retain_graph=True)
            self.layers[d].update_forward(lr / len(x))

    def get_layer_params(self):
        layer_params = {}
        for idx, layer in enumerate(self.layers):
            layer_params[f'layer_{idx}'] = {
                'forward_function_1': layer.forward_function_1.get_params(),
                'forward_function_2': layer.forward_function_2.get_params(),
                'backward_function_1': layer.backward_function_1.get_params(),
                'backward_function_2': layer.backward_function_2.get_params(),
            }
        return layer_params

    def save_model(self, ckpt):
        path = ckpt
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Collect the parameters from each layer's functions
        layer_params = {}
        for idx, layer in enumerate(self.layers):
            layer_params[f'layer_{idx}'] = {
                'forward_function_1': layer.forward_function_1.get_params(),
                'forward_function_2': layer.forward_function_2.get_params(),
                'backward_function_1': layer.backward_function_1.get_params(),
                'backward_function_2': layer.backward_function_2.get_params(),
            }
        # Save the collected parameters to the specified path
        torch.save(layer_params, path)



    def save_training_dynamics(self, train_losses, train_accuracies, test_losses, test_accuracies, trial, ckpt):
        path = ckpt
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Check if the file exists and has content
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r") as file:
                data = json.load(file)
        else:
            # Initialize the dictionary with lists to store results
            data = [{
                "Trial": [],
                "Train Losses": [],
                "Train Accuracies": [],
                "Test Losses": [],
                "Test Accuracies": []
            }]

        # Append new results to each list within the first dictionary entry
        data[0]["Trial"].append(trial)
        data[0]["Train Losses"].append(train_losses)
        data[0]["Train Accuracies"].append(train_accuracies)
        data[0]["Test Losses"].append(test_losses)
        data[0]["Test Accuracies"].append(test_accuracies)

        # Write the updated dictionary back to the file
        with open(path, "w") as file:
            json.dump(data, file, indent=4)



    def load_state(self, state_dict):
        for layer_key, layer_state in state_dict.items():
            layer_idx = int(layer_key.split('_')[-1])
            layer = self.layers[layer_idx]

            if 'forward_function_1' in layer_state and hasattr(layer.forward_function_1, 'load_params'):
                layer.forward_function_1.load_params(layer_state['forward_function_1'])

            if 'forward_function_2' in layer_state and hasattr(layer.forward_function_2, 'load_params'):
                layer.forward_function_2.load_params(layer_state['forward_function_2'])

            if 'backward_function_1' in layer_state and hasattr(layer.backward_function_1, 'load_params'):
                layer.backward_function_1.load_params(layer_state['backward_function_1'])

            if 'backward_function_2' in layer_state and hasattr(layer.backward_function_2, 'load_params'):
                layer.backward_function_2.load_params(layer_state['backward_function_2'])

        print("Model loaded successfully")

    # def external_test(self, test_loader, state_dict=None):
    #     if state_dict is not None:
    #         self.load_state(state_dict)
    #     loss = 0
    #     correct = 0
    #     total = 0
    #     for x, y in test_loader:
    #         x, y = x.to(self.device), y.to(self.device)
    #         y_pred = self.forward(x)
    #         loss += self.loss_function(y_pred, y).item()
    #         _, predicted = y_pred.max(1)
    #         total += y.size(0)
    #         correct += predicted.eq(y).sum().item()
    #     return loss / total, 100 * correct / total


