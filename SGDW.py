import torch
from torch.optim import Optimizer

class SGDW(Optimizer):
    r"""SGD with momentum and **decoupled** weight decay (Loshchilov & Hutter, 2019)."""

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.,
                 dampening=0., nesterov=False):
        if lr < 0: raise ValueError("Invalid learning rate")
        if momentum < 0 or dampening < 0: raise ValueError("Invalid momentum/dampening")
        if weight_decay < 0: raise ValueError("Invalid weight_decay value")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            mu, damp, nesterov = group['momentum'], group['dampening'], group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                # --- decoupled weight decay (pre‑grad) ---
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                # --- momentum buffer ---
                if mu != 0:
                    buf = self.state.setdefault(p, {}).setdefault(
                        'momentum_buffer', torch.zeros_like(p))
                    buf.mul_(mu).add_(g, alpha=1 - damp)
                    d_p = buf if not nesterov else g.add(buf, alpha=mu)
                else:
                    d_p = g

                # --- gradient step ---
                p.add_(d_p, alpha=-lr)

        return loss