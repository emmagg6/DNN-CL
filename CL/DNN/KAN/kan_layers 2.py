'''
Adapted from https://www.kaggle.com/code/mickaelfaust/99-kolmogorov-arnold-network-conv2d-and-mnist?scriptVersionId=177633280 

by emmagg6
'''


import numpy as np
import os
import torch
import torch.nn.functional as F
import math
import torchvision
import  matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.cuda.amp import autocast
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor
# from tqdm.auto import tqdm

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
        self.device = device
        
        self.register_buffer('norm_mean', torch.zeros(in_dim, device=device))
        self.register_buffer('norm_std', torch.ones(in_dim, device=device))
        
        self.projection_matrix = torch.nn.Parameter(torch.eye(in_dim, out_dim, device=device))
    
    def forward(self, x):
        x_norm = (x - self.norm_mean) / (self.norm_std + 1e-8)
        x_proj = F.linear(x_norm, self.projection_matrix)
        return x_proj
    
        
class KANLinearFFT(torch.nn.Module):
    """
    The code implements a Kolmogorov-Arnold Network (KAN) linear layer using the Fast Fourier Transform (FFT)

        1. Spline basis functions: The code computes spline basis functions using cosine and sine functions. These basis functions are used to approximate the input data
        2. Coefficients: The code initializes the coefficients (self.coeff) with random values scaled by noise_scale. These coefficients are learned during training and used to weight the spline basis functions
        3. Base function: The code applies a base function (self.base_fun) to the input data and scales it using self.scale_base. This base function is added to the spline output
        4. Spline output: The code computes the spline output using Einstein summation (torch.einsum). It multiplies the spline basis functions with the scaled coefficients

    The code aims to implement a KAN linear layer that respects the Kolmogorov-Arnold theorem and networks. The mathematical components work together to approximate the input data using spline basis functions and learned coefficients while applying regularization to promote desirable properties in the learned representation
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size=5,
        noise_scale=0.1,
        noise_scale_base=0.1,
        scale_spline=None,
        base_fun=KolmogorovActivation(),
        bias=False,
        bias_trainable=True,
        sp_trainable=True,
        sb_trainable=True,
        device=device,
        preprocess_dim=None
    ):
        torch.nn.Module.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_fun = base_fun
        self.size = in_dim * out_dim
        self.grid_size = grid_size
        self.device = device

        if preprocess_dim is None:
            preprocess_dim = in_dim
            
        self.preprocessing = KANPreprocessing(in_dim, preprocess_dim, device=device)

        k = torch.arange(1, self.grid_size + 1, device=device).view(
            1, 1, self.grid_size
        ).to(torch.float32)
        self.register_buffer("k", k)

        if scale_spline is not None:
            self.scale_spline = torch.nn.Parameter(
                torch.full(
                    (
                        out_dim,
                        in_dim,
                    ),
                    fill_value=scale_spline,
                    device=device,
                ),
                requires_grad=sp_trainable,
            )
        else:
            self.register_buffer("scale_spline", torch.tensor([1.0], device=device))

        self.coeff = torch.nn.Parameter(
            torch.randn(2, out_dim, in_dim, grid_size, device=device, dtype=torch.float32)
            * noise_scale
            / (np.sqrt(in_dim) * np.sqrt(grid_size)),
        )

        self.scale_base = torch.nn.Parameter(
            torch.randn(self.out_dim, self.in_dim, device=device) * math.sqrt(2 / (in_dim + out_dim)),
            requires_grad=sb_trainable,
        )

        if bias is True:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_dim, device=device), requires_grad=bias_trainable
            )
        else:
            self.bias = None

    def forward(self, x):
        x = self.preprocessing(x)
        shape = x.shape[:-1]
        x = x.view(-1, self.in_dim)

        x_unsqueezed = x.unsqueeze(-1)
        splines = torch.stack(
            [
                torch.cos(x_unsqueezed * self.k),
                torch.sin(x_unsqueezed * self.k),
            ],
            dim=1,
        ).view(x.shape[0], -1)

        batch_size = x.shape[0]
        y_b = F.linear(self.base_fun(x), self.scale_base)

        y_spline = torch.einsum(
            "bk,ok->bo",
            splines,
            (self.coeff * self.scale_spline.unsqueeze(-1)).view(self.out_dim, -1),
        )

        y = y_b + y_spline

        if self.bias is not None:
            y = y + self.bias


        y = y.view(*shape, self.out_dim)

        return y




# class KANConv2d(torch.nn.Module):
#     """
#     The code implements a 2D convolutional layer using Kolmogorov-Arnold Networks (KANs)

#         1. Unfolding: The code uses torch.nn.functional.unfold to extract patches from the input tensor. This operation is equivalent to sliding a window over the input tensor and extracting the patches at each position
#         2. KANLinearFFT layers: The code creates a ModuleList of KANLinearFFT layers, one for each input channel. These layers are applied to the unfolded patches of the corresponding input channel
#         3. Permutation and reshaping: The code permutes and reshapes the unfolded tensor to prepare it for applying the KANLinearFFT layers. The tensor is reshaped to have dimensions (batch_size, num_patches, in_channels, kernel_size^2)
#         4. Applying KANLinearFFT layers: The code applies the KANLinearFFT layers to each input channel using a list comprehension and torch.stack. The outputs of the KANLinearFFT layers are stacked along a new dimension
#         5. Summing the outputs: The code sums the outputs of the KANLinearFFT layers along the channel dimension using torch.sum(dim=2). This operation combines the contributions from each input channel
#         6. Adding bias: If a bias term is specified, the code adds the bias to the output tensor
#         7. Computing output spatial dimensions: The code computes the output spatial dimensions (h and w) based on the input shape, kernel size, stride, padding, and dilation. These dimensions are used to reshape the output tensor
#         8. Reshaping the output: The code permutes and reshapes the output tensor to have dimensions (*shape[:-3], out_channels, h, w), where *shape[:-3] represents any additional dimensions from the input tensor
    
#     The code aims to implement a 2D convolutional layer using KANs, which respects the Kolmogorov-Arnold theorem and networks. The mathematical components work together to extract patches from the input tensor, apply KANLinearFFT layers to each input channel, combine the outputs, and reshape the result to obtain the final output tensor
#     """
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         bias=True,
#         device=device,
#     ):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride
#         self.dilation = dilation

#         if padding == 'same':
#             padding = (kernel_size - 1) // 2
#         self.padding = padding

#         self.kernels = torch.nn.ModuleList()
#         for _ in range(in_channels):
#             self.kernels.append(
#                 KANLinearFFT(kernel_size * kernel_size, out_channels, device=device)
#             )

#         if bias:
#             self.bias = torch.nn.Parameter(
#                 torch.zeros(out_channels), requires_grad=True
#             )
#         else:
#             self.bias = bias

#         self.unfold_params = {
#             "kernel_size": kernel_size,
#             "stride": stride,
#             "padding": padding,
#             "dilation": dilation
#         }

#     def forward(self, x):
#         shape = x.shape
#         x = x.view(-1, shape[-3], shape[-2], shape[-1])

#         x = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
#         x = x.permute(0, 2, 1).view(x.shape[0], -1, self.in_channels, self.kernel_size**2)

#         x_out = torch.stack([kernel(x[:, :, i, :].contiguous()) for i, kernel in enumerate(self.kernels)], dim=2)
#         x = x_out.sum(dim=2)

#         if self.bias is not False:
#             x = x + self.bias[None, None, :]

#         h = (shape[-2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
#         w = (shape[-1] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

#         x = x.permute(0, 2, 1).view(*shape[:-3], self.out_channels, h, w)

#         return x

####### takes in a 4D tensor and transforms it into a 2D tensor for the KANLinearFFT layer ########
class KANConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation

        if padding == 'same':
            padding = (kernel_size - 1) // 2
        self.padding = padding

        self.kernels = torch.nn.ModuleList()
        for _ in range(in_channels):
            self.kernels.append(
                KANLinearFFT(kernel_size * kernel_size, out_channels, device=device)
            )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_channels), requires_grad=True
            )
        else:
            self.bias = bias

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape

        # Unfold the input tensor into patches
        patches = torch.nn.functional.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation
        )

        # Calculate the number of patches
        num_patches = patches.shape[-1]

        # Reshape the patches tensor to match the expected shape for KANLinearFFT
        patches = patches.view(batch_size, in_channels, self.kernel_size ** 2, num_patches).permute(0, 1, 3, 2).contiguous()

        # Apply the KANLinearFFT layers to each input channel
        x_out = torch.stack([self.kernels[i](patches[:, i, :, :]) for i in range(in_channels)], dim=1)
        x = x_out.sum(dim=1)

        if self.bias is not False:
            x = x + self.bias

        # Calculate the output height and width
        out_height = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # Reshape the output tensor to match the expected shape
        x = x.view(batch_size, self.out_channels, out_height, out_width)
        # print("reshaped the output tensor")

        return x


@torch.jit.script
def b_splines(x, grid, k: int):
    x = x.unsqueeze(-1)
    value = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

    p = 1
    while p <= k:
        # Compute the differences between grid points for the left and right intervals
        diff_left = grid[:, p:-1] - grid[:, : -(p + 1)]
        diff_right = grid[:, p + 1 :] - grid[:, 1:(-p)]

        # Create masks to identify non-zero differences
        mask_left = torch.ne(diff_left, 0)
        mask_right = torch.ne(diff_right, 0)

        # Compute the ratios for the left and right intervals
        # The ratios represent the relative position of x within each interval
        # If the difference is zero, the ratio is set to zero to avoid division by zero
        ratio_left = torch.where(mask_left, (x - grid[:, : -(p + 1)]) / diff_left, torch.zeros_like(diff_left))
        ratio_right = torch.where(mask_right, (grid[:, p + 1 :] - x) / diff_right, torch.zeros_like(diff_right))

        # Update the value using the weighted average of the left and right intervals
        # The weights are determined by the ratios and normalized by their sum
        # A small constant (1e-8) is added to the denominator to avoid division by zero
        value = (ratio_left * value[:, :, :-1] + ratio_right * value[:, :, 1:]) / (ratio_left + ratio_right + 1e-8)

        p += 1

    return value


def curve2coeff(x, y, grid, k, eps=1e-8):
    splines = torch.einsum('ijk->jik', b_splines(x, grid, k))

    # Perform Singular Value Decomposition (SVD) on the splines matrix
    # u: left singular vectors
    # s: singular values
    # v: right singular vectors
    u, s, v = torch.linalg.svd(splines, full_matrices=False)

    # Compute the inverse of the singular values
    # Create a tensor with the same shape as s and fill it with zeros
    s_inv = torch.zeros_like(s)
    # Set the non-zero singular values to their reciprocal
    s_inv[s != 0] = 1 / s[s != 0]
    
    s_inv = torch.diag_embed(s_inv)

    # Compute the coefficients of the curve using the SVD components and the y values
    # The coefficients are obtained by solving the linear system:
    # splines * coefficients = y
    # The solution is given by:
    # coefficients = v * s_inv * u^T * y
    value = v.transpose(-2, -1) @ s_inv @ u.transpose(-2, -1) @ y.transpose(0, 1)
    # Permute the dimensions of the coefficients tensor to match the desired output shape
    value = value.permute(2, 0, 1)
    
    return value
    
class KANLinear(torch.nn.Module):
    """
    The KANLinear class implements a linear layer using Kolmogorov-Arnold Networks (KANs)

        1. Grid creation: The code creates a grid of knot points for the B-splines using torch.linspace. The grid is parameterized and can be updated during training
        2. Coefficient initialization: The code generates random noise for the coefficients and computes the initial coefficients using the curve2coeff function. The coefficients are parameterized and can be updated during training
        3. Base scale initialization: The code initializes the base scale with random values scaled by noise_scale_base
        4. Spline scale initialization: The code initializes the spline scale with a constant value specified by scale_spline
        5. Forward pass: In the forward method, the code computes the B-spline basis functions for the input x using the b_splines function. It then computes the base function output using the base scale and the spline output using the coefficients and spline scale. The base function output and spline output are added together, and a bias term is added if specified
        6. Grid update: The update_grid method updates the grid points based on the input x. It computes the B-spline basis functions for the input x and the original coefficients scaled by the spline scale. It then computes the output using the splines and original coefficients. The input x is sorted, and percentiles are computed for the grid points. The updated grid is computed using a combination of adaptive and uniform grid points. The grid parameter and coefficients are updated with the new grid points using the curve2coeff function

    The KANLinear class provides a linear layer that can learn complex functions using B-splines and adaptive grid points. The coefficients and grid points are parameterized and can be updated during training to better fit the data. The class also includes regularization terms for the coefficients to encourage sparsity and smoothness
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size=7,
        k=4,
        noise_scale=0.05,  
        noise_scale_base=0.05, 
        scale_spline=1.0,
        base_fun=KolmogorovActivation(),
        bias=True,
        grid_eps=0.01,  
        grid_range=[-1, +1],
        bias_trainable=True,
        sp_trainable=True,
        sb_trainable=True,
        device=device,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.size = in_dim * out_dim
        self.grid_size = grid_size
        self.device = device

        self.register_buffer(
            "grid",
            torch.linspace(grid_range[0], grid_range[1], grid_size + 2 * k + 1, device=device).repeat(self.in_dim, 1)
        )

        noise = torch.randn(grid_size + 1, in_dim, out_dim, device=device) * noise_scale / math.sqrt(grid_size)
        
        coeff_reg = 0.01
        self.coeff = torch.nn.Parameter(
            curve2coeff(
                x=self.grid.T[k:-k], 
                y=noise,
                grid=self.grid,
                k=k,
            ).contiguous()
        )
        self.coeff_reg_loss = coeff_reg * torch.sum(self.coeff ** 2)

        self.scale_base = nn.Parameter(
            (
                1 / (in_dim**0.5)
                + (torch.randn(self.out_dim, self.in_dim, device=device) * 2 - 1)
                * noise_scale_base
            ),
            requires_grad=sb_trainable,
        )
        
        self.scale_spline = nn.Parameter(
            torch.full(
                (
                    out_dim,
                    in_dim,
                ),
                fill_value=scale_spline,
                device=device,
            ),
            requires_grad=sp_trainable,
        )  

        self.mask = torch.nn.Parameter(
            torch.ones(self.out_dim, self.in_dim, device=device)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(1, out_dim, device=device), requires_grad=bias_trainable
            )
        else:
            self.bias = None

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.view(-1, self.in_dim)

        splines = b_splines(x, self.grid, self.k)  


        y_b = torch.einsum('bi,oi->bo', self.base_fun(x), self.scale_base)
        
        y_spline = torch.einsum('bik,oik->bo', splines, self.coeff * self.scale_spline.unsqueeze(-1))
        
        y = y_b + y_spline

        if self.bias is not None:
            y = y + self.bias

        y = y.view(*shape, self.out_dim)


        return y

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        batch_size = x.shape[0]

        splines = b_splines(x, self.grid, self.k)

        orig_coeff = self.coeff * self.scale_spline.unsqueeze(-1)

        y = (splines.permute(1, 0, 2) @ orig_coeff.permute(1, 2, 0)).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]

        percentiles = torch.linspace(0, 100, self.grid_size + 1, device=self.device)
        grid_adaptive = torch.percentile(x_sorted, percentiles, dim=0)

        uniform_step = (
            x_sorted[-1] - x_sorted[0] + 2 * margin
        ) / self.grid_size  # [in_dim]
        grid_uniform = (
            torch.arange(self.grid_size + 1, device=self.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.k, 0, -1, device=self.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.k + 1, device=self.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.coeff.data.copy_(curve2coeff(x, y, self.grid, self.k))

class KANLinear2(KANLinear):
    """
    The KANLinear2 class is a variant of the KANLinear class with some modifications

        1. Grid creation: The code creates a grid of knot points for the B-splines using torch.arange. The grid points are evenly spaced based on the specified grid_range and grid_size. The grid is registered as a buffer to avoid being treated as a learnable parameter
        2. Coefficient initialization: The code generates random noise for the coefficients and computes the initial coefficients using the curve2coeff function. The coefficients are parameterized and can be updated during training
        3. Spline scale initialization: If scale_spline is provided, the code initializes the spline scale with the specified value. Otherwise, it registers a buffer with a default value of 1.0
        4. Base scale initialization: The code initializes the base scale with random values scaled by sqrt(2 / (in_dim + out_dim))
        5. Forward pass: In the forward method, the code computes the B-spline basis functions for the input x using the b_splines function. It then computes the base function output using the base scale and the spline output using the coefficients and spline scale. The base function output and spline output are added together, and a bias term is added if specified
    
    The KANLinear2 class provides a linear layer that can learn complex functions using B-splines and fixed grid points. The coefficients are parameterized and can be updated during training to better fit the data
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size=5,
        k=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        scale_spline=None,
        base_fun=KolmogorovActivation(),
        grid_eps=0.02,
        grid_range=[-1, +1],
        bias=False,
        bias_trainable=True,
        sp_trainable=True,
        sb_trainable=True,
        device=device,
    ):
        torch.nn.Module.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.size = in_dim * out_dim
        self.grid_size = grid_size
        self.device = device

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-k, grid_size + k + 1, device=device) * step + grid_range[0]
        ).repeat(self.in_dim, 1)
        self.register_buffer("grid", grid)  

        if scale_spline is not None:
            self.scale_spline = torch.nn.Parameter(
                torch.full(
                    (
                        out_dim,
                        in_dim,
                    ),
                    fill_value=scale_spline,
                    device=device,
                ),
                requires_grad=sp_trainable,
            )
        else:
            self.register_buffer("scale_spline", torch.tensor([1.0], device=device))

        noise = (
            (torch.randn(grid_size + 1, in_dim, out_dim, device=device))
            * noise_scale
            / math.sqrt(self.grid_size)
        )
        coeff_reg = 0.01
        self.coeff = torch.nn.Parameter(
            curve2coeff(
                x=self.grid.T[k:-k], 
                y=noise,
                grid=self.grid,
                k=k,
            ).contiguous()
        )
        self.coeff_reg_loss = coeff_reg * torch.sum(self.coeff ** 2)

        self.scale_base = torch.nn.Parameter(
            torch.randn(self.out_dim, self.in_dim, device=device) * math.sqrt(2 / (in_dim + out_dim)),
            requires_grad=sb_trainable,
        )

        if bias is True:
            self.bias = torch.nn.Parameter(
                torch.rand(out_dim), requires_grad=bias_trainable
            )
        else:
            self.bias = None
            
    def forward(self, x):
        shape = x.shape[:-1]
        x = x.view(-1, self.in_dim)
        

        splines = b_splines(x, self.grid, self.k) 


        batch_size = x.shape[0]
        y_b = F.linear(self.base_fun(x), self.scale_base)

        y_spline = torch.einsum('bik,oik->bo', splines, self.coeff * self.scale_spline.unsqueeze(-1))

        y = y_b + y_spline

        if self.bias is not None:
            y = y + self.bias


        y = y.view(*shape, self.out_dim)

        return y
