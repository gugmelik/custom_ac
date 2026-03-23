import torch 
import torch.nn as nn

"""
This code implements manual activation checkpointing for a single module call. 
It defines a custom autograd function that saves only the input during the forward pass 
and re-runs the module during the backward pass to compute gradients. 
The `checkpoint_block` function is a convenient wrapper around this custom autograd function.
"""
class _CheckpointBlockFn(torch.autograd.Function):
    """Manual activation checkpointing for one module call.

    The forward pass runs under no_grad and saves only the block input.
    During backward we re-run the block with grad enabled and use autograd.grad
    to compute gradients for the input and block parameters.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, module: nn.Module, preserve_rng_state: bool, *params: torch.Tensor) -> torch.Tensor:
        del params  # Parameters are passed only so autograd expects their grads.
        ctx.module = module
        ctx.preserve_rng_state = preserve_rng_state
        ctx.save_for_backward(x)

        if preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            if x.is_cuda:
                ctx.cuda_rng_state = torch.cuda.get_rng_state(x.device)
            else:
                ctx.cuda_rng_state = None

        with torch.no_grad():
            out = module(x)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        x = x.detach().requires_grad_(True)

        if ctx.preserve_rng_state:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(x.device) if x.is_cuda else None

            torch.set_rng_state(ctx.cpu_rng_state)
            if x.is_cuda and ctx.cuda_rng_state is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng_state, x.device)

        with torch.enable_grad():
            out = ctx.module(x)

        param_list = tuple(p for p in ctx.module.parameters() if p.requires_grad)
        grads = torch.autograd.grad(
            outputs=out,
            inputs=(x, *param_list),
            grad_outputs=grad_out,
            allow_unused=True,
        )

        if ctx.preserve_rng_state:
            torch.set_rng_state(cpu_rng_state)
            if x.is_cuda and cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, x.device)

        grad_x = grads[0]
        grad_params = grads[1:]
        return grad_x, None, None, *grad_params


def checkpoint_block(module: nn.Module, x: torch.Tensor, preserve_rng_state: bool = True) -> torch.Tensor:
    params = tuple(p for p in module.parameters() if p.requires_grad)
    return _CheckpointBlockFn.apply(x, module, preserve_rng_state, *params)