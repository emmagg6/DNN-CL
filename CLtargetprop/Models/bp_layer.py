import torch
from torch import nn
from utils import batch_normalization

import sys

'''
class bp_layer:
    def __init__(self, in_dim, out_dim, activation_function, device):
        # weights
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        self.fixed_weight = torch.empty(out_dim, in_dim, device=device)
        nn.init.normal_(self.fixed_weight, self.weight.mean().item(), self.weight.std().item())

        # functions
        if activation_function == "linear":
            self.activation_function = (lambda x: x)
            self.activation_derivative = (lambda x: 1)
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
            self.activation_derivative = (lambda x: 1 - torch.tanh(x)**2)
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # activation
        self.linear_activation = None
        self.activation = None

    def forward(self, x, update=True):
        if update:
            self.linear_activation = x @ self.weight.T
            self.activation = self.activation_function(self.linear_activation)
            return self.activation
        else:
            a = x @ self.weight.T
            h = self.activation_function(a)
            return h

    def get_params(self):
        # Return the parameters for saving
        return {
            'weight': self.weight.detach().cpu().numpy(),
            # Include other parameters you wish to save, if any
        }

    def load_params(self, params):
        # Load the parameters from a saved state
        self.weight.data.copy_(torch.tensor(params['weight'], device=self.weight.device))

        '''

class bp_layer:
    def __init__(self, in_dim, out_dim, activation_function, device, continual=False):
        self.device = device
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        # Initialize use_batch_norm to False by default
        self.use_batch_norm = False

        self.cont = continual

        # Determine if batch normalization should be used
        if activation_function == "linear-BN" or activation_function == "tanh-BN": 
            self.use_batch_norm = True


        if self.use_batch_norm:
            # Initialize batch normalization layer
            self.batch_norm = nn.BatchNorm1d(out_dim).to(device)
            # Remove '-BN' from the activation function string to get the base activation function
            # base_activation_function = activation_function.replace('-BN', '')
            base_activation_function = activation_function[:-3]
        else:
            base_activation_function = activation_function

        ###### CHECK #######
        if self.use_batch_norm:
            print("Batch Norm running_mean:", self.batch_norm.running_mean)
            print("Batch Norm running_var:", self.batch_norm.running_var)

        # Set up the activation function and its derivative
        if base_activation_function == "linear":
            self.activation_function = nn.Identity()
            self.activation_derivative = (lambda x: 1)
        elif base_activation_function == "tanh":
            self.activation_function = nn.Tanh()
            self.activation_derivative = (lambda x: 1 - torch.tanh(x) ** 2)
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # activation
        self.linear_activation = None
        self.activation = None

    def forward(self, x, update=True):
        # Apply linear transformation
        self.linear_activation = x @ self.weight.T
        # print(f"Layer forward pass - linear activation (sample): {self.linear_activation[0][:5]}")  # first 5 values of the first sample

        # Apply batch normalization if it is being used
        if self.use_batch_norm:
            self.linear_activation = self.batch_norm(self.linear_activation)
            # print(f"Layer forward pass - after batch norm (sample): {self.linear_activation[0][:5]}")  # first 5 values of the first sample

        # Apply the activation function
        self.activation = self.activation_function(self.linear_activation)
        # print(f"Layer forward pass - after activation (sample): {self.activation[0][:5]}")  # first 5 values of the first sample
        return self.activation

    def get_params(self):
        # Return the parameters for saving, including batch norm parameters if applicable
        params = {'weight': self.weight.detach().cpu().numpy()}
        if self.use_batch_norm:
            params['batch_norm'] = {
                'weight': self.batch_norm.weight.detach().cpu().numpy(),
                'bias': self.batch_norm.bias.detach().cpu().numpy(),
                'running_mean': self.batch_norm.running_mean.cpu().numpy(),
                'running_var': self.batch_norm.running_var.cpu().numpy(),
            }
        return params

    def load_params(self, params):
        self.weight.data.copy_(torch.tensor(params['weight'], device=self.device))
        if self.use_batch_norm and 'batch_norm' in params:
            bn_params = params['batch_norm']
            self.batch_norm.weight.data.copy_(torch.tensor(bn_params['weight'], device=self.device))
            self.batch_norm.bias.data.copy_(torch.tensor(bn_params['bias'], device=self.device))
            self.batch_norm.running_mean.copy_(torch.tensor(bn_params['running_mean'], device=self.device))
            self.batch_norm.running_var.copy_(torch.tensor(bn_params['running_var'], device=self.device))

        # Ensure that the weights remain trainable after loading
        self.weight.requires_grad_(True)
        if self.use_batch_norm:
            self.batch_norm.weight.requires_grad_(True)
            self.batch_norm.bias.requires_grad_(True)

        if self.use_batch_norm and self.cont:
            # Reset running stats if we are continuing training
            self.batch_norm.reset_running_stats()