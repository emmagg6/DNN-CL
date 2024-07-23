'''
From from https://www.kaggle.com/code/mickaelfaust/99-kolmogorov-arnold-network-conv2d-and-mnist?scriptVersionId=177633280 

'''


import numpy as np
import torch
import torch.nn.functional as F
import os
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class KolmogorovActivation(nn.Module):
    """
    The KolmogorovActivation class defines an activation function based on the Kolmogorov-Arnold representation theorem

        1. Sine term: The code computes the sine of the input x using torch.sin(x). The sine function introduces periodic behavior and captures non-linear patterns in the input
    
    The KolmogorovActivation class provides an activation function that leverages the Kolmogorov-Arnold representation theorem. The theorem states that any continuous function can be represented as a superposition of a finite number of continuous functions. By combining the input, sine, the activation function can approximate a wide range of non-linear functions and capture complex patterns in the input data
    The use of the Kolmogorov-Arnold representation theorem in the activation function allows the neural network to learn more expressive and flexible representations of the input data. The combination of different mathematical terms introduces non-linearity and enables the network to capture intricate relationships and patterns in the data
    """
    def forward(self, x):
        return x + torch.sin(x)
    

class KANPreprocessing(torch.nn.Module):
    def __init__(self, in_dim, out_dim, device=device):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.register_buffer('norm_mean', torch.zeros(in_dim, device=device))
        self.register_buffer('norm_std', torch.ones(in_dim, device=device))
        
        self.projection_matrix = torch.nn.Parameter(torch.eye(in_dim, out_dim, device=device))
    
    def forward(self, x):
        x_norm = (x - self.norm_mean) / (self.norm_std + 1e-8)
        x_proj = F.linear(x_norm, self.projection_matrix)
        return x_proj
    
        