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
    def __init__(self, in_dim, out_dim, activation_function, device):
        self.device = device
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        # Determine if batch normalization should be used based on the activation_function string
        self.use_batch_norm = '-BN' in activation_function
        if self.use_batch_norm:
            # Initialize batch normalization layer
            self.batch_norm = nn.BatchNorm1d(out_dim).to(device)
            # Remove '-BN' from the activation function string to get the base activation function
            base_activation_function = activation_function.replace('-BN', '')
        else:
            base_activation_function = activation_function

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

        # Apply batch normalization if it is being used
        if self.use_batch_norm:
            self.linear_activation = self.batch_norm(self.linear_activation)

        # Apply the activation function
        self.activation = self.activation_function(self.linear_activation)
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
        # Load the parameters from a saved state
        self.weight.data.copy_(torch.tensor(params['weight'], device=self.device))
        if self.use_batch_norm and 'batch_norm' in params:
            bn_state = params['batch_norm']
            self.batch_norm.weight.data.copy_(torch.tensor(bn_state['weight'], device=self.device))
            self.batch_norm.bias.data.copy_(torch.tensor(bn_state['bias'], device=self.device))
            self.batch_norm.running_mean.copy_(torch.tensor(bn_state['running_mean'], device=self.device))
            self.batch_norm.running_var.copy_(torch.tensor(bn_state['running_var'], device=self.device))