'''
Based on code used for the experiments conducted in the submitted paper 
"Fixed-Weight Difference Target Propagation" by K. K. S. Wu, K. C. K. Lee, and T. Poggio.

Adaptation for Continual Learning by emmagg6.

'''

import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from Models.TP.tp_fcns import *


class tp_layer:
    def __init__(self, in_dim, out_dim, device, params):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        # set forward functions
        self.forward_function_1 = self.set_function(in_dim, in_dim, self, self.device, params["ff1"])
        self.forward_function_2 = self.set_function(in_dim, out_dim, self, self.device, params["ff2"])

        # set backward functions
        self.backward_function_1 = self.set_function(out_dim, in_dim, self, self.device, params["bf1"])
        self.backward_function_2 = self.set_function(in_dim, in_dim, self, self.device, params["bf2"])

        # values
        self.input = None
        self.hidden = None
        self.output = None
        self.target = None

    def forward(self, x, update=True):
        if update:
            self.input = x
            self.hidden = self.forward_function_1.forward(self.input)
            self.output = self.forward_function_2.forward(self.hidden)
            self.output = self.output.requires_grad_()
            self.output.retain_grad()
            return self.output
        else:
            h = self.forward_function_1.forward(x)
            y = self.forward_function_2.forward(h)
            return y

    def update_forward(self, lr):
        self.forward_function_1.update(lr)
        self.forward_function_2.update(lr)

    def update_backward(self, lr):
        self.backward_function_1.update(lr)
        self.backward_function_2.update(lr)

    def zero_grad(self):
        if self.output.grad is not None:
            self.output.grad.zero_()
        self.forward_function_1.zero_grad()
        self.forward_function_2.zero_grad()
        self.backward_function_1.zero_grad()
        self.backward_function_2.zero_grad()

    def get_forward_grad(self):
        ff1_grad = self.forward_function_1.get_grad()
        ff2_grad = self.forward_function_2.get_grad().clone()
        grad = {"ff1": ff1_grad.clone() if ff1_grad is not None else None,
                "ff2": ff2_grad.clone() if ff2_grad is not None else None, }
        return grad


    def set_function(in_dim, out_dim, layer, device, params):
        if params["type"] == "parameterized":
            return parameterized_function(in_dim, out_dim, layer, device, params)
        else:
            raise NotImplementedError()
