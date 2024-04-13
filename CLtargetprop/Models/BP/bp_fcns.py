'''
Inspired by code used for the experiments conducted in the submitted paper 
"Fixed-Weight Difference Target Propagation" by K. K. S. Wu, K. C. K. Lee, and T. Poggio.

Adaptation for Continual Learning by emmagg6.

'''

import torch
from torch import nn
from utils import batch_normalization


class ParameterizedFunction(nn.Module):
    def __init__(self, in_dim, out_dim, params):
        super(ParameterizedFunction, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the parameter tensor and initialize it
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        # self.initialize_weights(params["init"])  # NOT WORKING NOT SURE WHY
        # print(f"initial_weights : {self.weight}")
        nn.init.orthogonal_(self.weight)

        # Define the activation function
        if params["act"] == "tanh":
            self.activation_function = nn.Tanh()
        elif params["act"] == "linear":
            self.activation_function = nn.Identity()
        elif params["act"] == "tanh-BN":
            # Define a batch normalization layer followed by Tanh
            self.activation_function = nn.Sequential(
                nn.BatchNorm1d(out_dim),  # Use out_dim instead of in_dim
                nn.Tanh()
            )
        elif params["act"] == "linear-BN":
            self.activation_function = nn.BatchNorm1d(out_dim)  # Use out_dim instead of in_dim


    # NOT WORKING NOT SURE WHY SO SKIPPING AND JUST INITIALIZING ABOVE
    def initialize_weights(self, init_type): # NOT WORKING
        if init_type == "uniform":
            nn.init.uniform_(self.weight, -1e-2, 1e-2)
        elif init_type == "gaussian":
            nn.init.normal_(self.weight, 0, 1e-3)
        elif init_type == "orthogonal":
            print("orthogonal")
            nn.init.orthogonal_(self.weight)
            print(f"orthogonal_weights : {self.weight}")
        else:
            raise NotImplementedError()
    ##################################################################

    def forward(self, input):
        # Apply the weights to the input and pass it through the activation function
        output = torch.matmul(input, self.weight.t())
        return self.activation_function(output)

    def get_params(self):
        # Return the weights of the parameterized function
        return {'weight': self.weight.detach().cpu().numpy()}
    
    def load_params(self, params):
        # Assuming 'weight' is the only parameter
        if 'weight' in params:
            self.weight.data.copy_(torch.from_numpy(params['weight']).to(self.device))