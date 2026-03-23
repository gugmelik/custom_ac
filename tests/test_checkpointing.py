import copy
import unittest

import torch
import torch.nn as nn

from src.ckpt import checkpoint_block


def _run_step(module: nn.Module, x: torch.Tensor, use_checkpoint: bool, preserve_rng_state: bool = True):
    module.zero_grad(set_to_none=True)
    x = x.clone().detach().requires_grad_(True)

    if use_checkpoint:
        out = checkpoint_block(module, x, preserve_rng_state=preserve_rng_state)
    else:
        out = module(x)

    loss = (out ** 2).mean()
    loss.backward()

    param_grads = [
        None if p.grad is None else p.grad.detach().clone()
        for p in module.parameters()
    ]
    return out.detach().clone(), x.grad.detach().clone(), param_grads


class TestActivationCheckpointing(unittest.TestCase):
    def test_checkpoint_matches_eager_forward_and_backward(self) -> None:
        torch.manual_seed(0)
        eager_model = nn.Sequential(
            nn.Linear(8, 16),
            nn.GELU(),
            nn.Linear(16, 8),
        )
        ckpt_model = copy.deepcopy(eager_model)

        x = torch.randn(4, 8)

        eager_out, eager_x_grad, eager_param_grads = _run_step(eager_model, x, use_checkpoint=False)
        ckpt_out, ckpt_x_grad, ckpt_param_grads = _run_step(ckpt_model, x, use_checkpoint=True)

        self.assertTrue(torch.allclose(eager_out, ckpt_out, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(eager_x_grad, ckpt_x_grad, atol=1e-6, rtol=1e-5))

        self.assertEqual(len(eager_param_grads), len(ckpt_param_grads))
        for eager_grad, ckpt_grad in zip(eager_param_grads, ckpt_param_grads):
            self.assertIsNotNone(eager_grad)
            self.assertIsNotNone(ckpt_grad)
            self.assertTrue(torch.allclose(eager_grad, ckpt_grad, atol=1e-6, rtol=1e-5))

    def test_checkpoint_preserves_rng_state_with_dropout(self) -> None:
        torch.manual_seed(123)
        eager_model = nn.Sequential(
            nn.Linear(8, 8),
            nn.Dropout(p=0.4),
            nn.Linear(8, 8),
        )
        eager_model.train()
        ckpt_model = copy.deepcopy(eager_model)
        ckpt_model.train()

        x = torch.randn(4, 8)

        torch.manual_seed(999)
        eager_out, eager_x_grad, eager_param_grads = _run_step(eager_model, x, use_checkpoint=False)

        torch.manual_seed(999)
        ckpt_out, ckpt_x_grad, ckpt_param_grads = _run_step(
            ckpt_model,
            x,
            use_checkpoint=True,
            preserve_rng_state=True,
        )

        self.assertTrue(torch.allclose(eager_out, ckpt_out, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(eager_x_grad, ckpt_x_grad, atol=1e-6, rtol=1e-5))

        self.assertEqual(len(eager_param_grads), len(ckpt_param_grads))
        for eager_grad, ckpt_grad in zip(eager_param_grads, ckpt_param_grads):
            self.assertIsNotNone(eager_grad)
            self.assertIsNotNone(ckpt_grad)
            self.assertTrue(torch.allclose(eager_grad, ckpt_grad, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
