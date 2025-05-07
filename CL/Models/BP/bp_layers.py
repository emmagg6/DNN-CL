'''
Inspired by code used for the experiments conducted in the submitted paper 
"Fixed-Weight Difference Target Propagation" by K. K. S. Wu, K. C. K. Lee, and T. Poggio.

Adaptation for Continual Learning by emmagg6.

'''

import torch
from torch import nn
import os
import time
import wandb

from Models.BP.bp_fcns import ParameterizedFunction


class bp_layers(nn.Module):
    def __init__(self, in_dim, out_dim, device, params):
        super(bp_layers, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        # Initialize forward functions as instances of ParameterizedFunction
        self.forward_function_1 = ParameterizedFunction(in_dim, in_dim, params["ff1"])
        self.forward_function_2 = ParameterizedFunction(in_dim, out_dim, params["ff2"])


    def forward(self, x):
        x = self.forward_function_1(x)  # Call the forward method of the first function
        x = self.forward_function_2(x)  # Call the forward method of the second function
        return x

    @staticmethod
    def set_function(in_dim, out_dim, params):
        # This method might not be necessary if you're directly initializing ParameterizedFunction above
        if params["type"] == "parameterized":
            return ParameterizedFunction(in_dim, out_dim, params)
        else:
            raise NotImplementedError()
