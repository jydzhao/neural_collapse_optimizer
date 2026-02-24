import torch
from torch.optim.optimizer import Optimizer, required
import math

class AdamAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay_coupled=0, weight_decay_decoupled=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay_coupled=weight_decay_coupled,
                        weight_decay_decoupled=weight_decay_decoupled)
        super().__init__(params, defaults)
        
        print('weight_decay_coupled',weight_decay_coupled)
        print('weight_decay_decoupled',weight_decay_decoupled)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamAdamW does not support sparse gradients')
                
                # Apply coupled weight decay (L2 reg added to gradient)
                if group['weight_decay_coupled'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay_coupled'])

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first and second moment estimates
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # Adam update (without any weight decay)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Apply decoupled weight decay (AdamW update)
                if group['weight_decay_decoupled'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay_decoupled'])
                    
        return loss