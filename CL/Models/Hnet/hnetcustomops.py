"""
non-von HNetOpPopCount, HNetOpCountBits autograd functions
CONFIDENTIAL
Copyright (c) 2022-2025, Non-Von LLC, all rights reserved.

Callable Functions
==================
HNetOpPopCount:
HNetOpCountBits:

"""
import torch
import torch.nn.functional as F

#@torch.compile()
class HNetOpPopCount(torch.autograd.Function):
    """
    Custom operation for pop-count version of the HNet operation.

    Mathematically this is 2*(x^t*H*x + k) - n_edges + switch_off      
    """
    @staticmethod
    def forward(ctx, H, k, x, edge_num, switched_off, custom_op_kern=True):
        n_cmps, dim, dim = H.size()

        ctx.save_for_backward(H, k, x, switched_off)

        x = torch.einsum('kj,nij,ki->kn', x, H, x)

        # Add k
        x = x + k.unsqueeze(0)

        # Transform from bitcount to popcount
        x = 2*x - edge_num + switched_off

        return x 

    @staticmethod
    def backward(ctx, grad_output):
        H, k, x, switched_off = ctx.saved_tensors

        # Gradients for hamiltonians are calculated using h_grad = 2*x^t*x

        sum_grad = grad_output.sum(dim=0)

        h_grad = 2*torch.einsum('bc,bi,bj->cij', grad_output, x, x)

        k_grad = 2*sum_grad

        # Gradients for x are calculated using x_grad = 4*H*x

        x_grad = 4 * torch.einsum('bc,cij,bj->bi', grad_output, H, x)

        swoff_grad = sum_grad

        return h_grad, k_grad, x_grad, None, swoff_grad

@torch.compile(dynamic=False)
class HNetOpCountBits(torch.autograd.Function):
    """
    Custom operation for count-bits version of the HNet operation.
    
    Mathematically this is x^t*H*x + k
    """
    @staticmethod
    def forward(ctx, H, k, x, custom_op_kern=True):
        n_cmps, dim, dim = H.size()

        ctx.save_for_backward(H, x)

        x = torch.einsum('kj,nij,ki->kn', x, H, x)

        # Add k
        x = x + k.unsqueeze(0)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        H, x = ctx.saved_tensors
        batch_n, node_n = x.shape
        cmp_n, _, _ = H.shape

        # Gradients for hamiltonians are calculated using h_grad = x'*x

        h_grad = torch.einsum('bc,bi,bj->cij', grad_output, x, x)

        k_grad = grad_output.sum(dim=0)

        # Gradients for x are calculated using h_grad = 4*H*x

        x_grad = 2 * torch.einsum('bc,cij,bj->bi', grad_output, H, x)

        return h_grad, k_grad, x_grad

    
    
