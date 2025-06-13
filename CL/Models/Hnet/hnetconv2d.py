"""
non-von HNetConv2d
CONFIDENTIAL
Copyright (c) 2022-2025, Non-Von LLC, all rights reserved.

Data Structures
===============
HNetConv2d

Callable Functions
==================
"""

import torch
import torch.nn as nn

from .hnetcommon import HNetComponentBank, Binarize

class HNetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_norm=True, n_random_edges=None):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(padding, int):
            padding = (padding, padding) 

        # Conv params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        # HNet specific params
        self.n_random_edges = n_random_edges
        
        self.tier1_hnet = HNetComponentBank(kernel_size[0]*kernel_size[1], out_channels, n_random_edges=n_random_edges)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        self.binarize = Binarize()
        self.pad = nn.ConstantPad2d((self.padding[1], self.padding[1], self.padding[0], self.padding[0]), 0)

    def forward(self, x: torch.Tensor):
        batch_size, _, h, w = x.size()

        kh = self.kernel_size[0] # Kernel Height
        kw = self.kernel_size[1] # Kernel Width
        dh = self.stride[0]      # Kernel Stride
        dw = self.stride[1]      # Kernel Width
        ph = self.padding[0]     # Padding Height
        pw = self.padding[1]     # Padding Width

        h += 2*ph
        w += 2*pw

        assert h - kh > 0 or w - kw > 0, "Kernel size not set correctly"
        assert (h - kh) % dh == 0, "Stride is not set correctly; must satisfy (h - kernel_h) / stride_h must be an integer"
        assert (w - kw) % dw == 0, "Stride is not set correctly; must satisfy (w - kernel_w) / stride_w must be an integer"

        h_out = (h - kh) // dh + 1
        w_out = (w - kw) // dw + 1

        x = self.binarize(x)

        # Turning this conv into a matmul to insert hnet in there better
        # Followed this post as baseline: https://discuss.pytorch.org/t/convolution-that-only-take-channel-wise-summation/21240/3
        if self.padding[0] != 0 and self.padding[1] != 0:
            x = self.pad(x)

        noncontiguous_patches = x.unfold(2, kh, dh).unfold(3, kw, dw) # batch_size, in_channels, h_windows, w_windows, kernel_size_h, kernel_size_w

        patches = noncontiguous_patches.contiguous().view(-1, kh*kw) # batch_size, in_channels, h_windows, w_windows, kernel_size_h*kernel_size_w
        
        del noncontiguous_patches
        del x

        output = self.tier1_hnet(patches).view(batch_size, self.in_channels, h_out, w_out, self.out_channels) # out_channels, all_inputs

        del patches

        output = output.sum(1).permute(0,3,1,2).contiguous() # batch_size, out_channels, h_windows, w_windows

        if self.use_norm:
            output = self.norm(output)

        return output
    