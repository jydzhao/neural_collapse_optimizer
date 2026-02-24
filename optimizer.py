import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
import math

from adamadamw import AdamAdamW
from sgd_L1 import SGD_L1
from signum import Signum, Signum_decoupledWD
from SGDW import SGDW


def init_optimizer(opt, lr, model, model_name, lr_schedule, momentum=None, weight_decay=None, total_training_steps=None, warmup_length=0.05, epochs_lr_decay=None, lr_decay=0.1, weight_decay_coupled=0.0, weight_decay_L1=0.0):
    
    
    ''' Initialize Optimizer and LR scheduler

        For Vision transformers, weight decay is only applied on parameters, which are not LayerNorm or Biases.
        
        The Learning rate schedule is per default a MultiStepLR-scheduler, which decays after 1/3 and 2/3 of the epochs, however to train 
        Transformers, it is recommended to use a cosine LR scheduler with warm-up instead.
    '''
    
    
    if opt == 'SGD' or opt == 'GD':
         optimizer = optim.SGD(model.parameters(), lr)
    elif opt == 'SGDMW' or opt == 'GDMW':
        # For ViT, apply WD only on parameters, except LayerNorm and Bias 
        if model_name == 'ViT_B_16' or model_name == 'ViT_custom':
            param_groups = [
                {"params": [], "weight_decay": weight_decay},  # weights
                {"params": [], "weight_decay": 0}  # biases + LayerNorms
            ]
            
            for n, p in model.named_parameters():
                if "bias" in n or "LayerNorm" in n:
                    param_groups[1]["params"].append(p)
                else:
                    param_groups[0]["params"].append(p)
        
            optimizer = optim.SGD(param_groups, lr, momentum)
        else:
            optimizer = optim.SGD(model.parameters(), lr, momentum, weight_decay=weight_decay)
    elif opt == 'SGDW':
        optimizer = SGDW(model.parameters(), lr, momentum, weight_decay=weight_decay)
    elif opt == 'SGD_L1_sign':
        optimizer = SGD_L1(model.parameters(), lr, momentum, weight_decay_L1=weight_decay_L1)
    elif opt == 'SGD_L1_softThr':
        optimizer = SGD_L1(model.parameters(), lr, momentum, weight_decay_L1=weight_decay_L1, soft_threshold=True)
    elif opt == 'Adam': #per default WD on all parameters
        if model_name == 'ViT_B_16' or model_name == 'ViT_custom':
            param_groups = [
                {"params": [], "weight_decay": weight_decay},  # weights
                {"params": [], "weight_decay": 0}  # biases + LayerNorms
            ]
            
            for n, p in model.named_parameters():
                if "bias" in n or "LayerNorm" in n:
                    param_groups[1]["params"].append(p)
                else:
                    param_groups[0]["params"].append(p)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opt == 'Signum': #per default WD on all parameters
        optimizer = Signum(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt == 'Signum_decoupledWD': #per default WD on all parameters
        optimizer = Signum_decoupledWD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)       
    elif opt == 'AdamW':
    
        if model_name == 'ViT_B_16' or model_name == 'ViT_custom':
            # For ViT, apply WD only on parameters, except LayerNorm and Bias 
            param_groups = [
                {"params": [], "weight_decay": weight_decay},  # weights
                {"params": [], "weight_decay": 0}  # biases + LayerNorms
            ]
            
            for n, p in model.named_parameters():
                if "bias" in n or "LayerNorm" in n:
                    param_groups[1]["params"].append(p)
                else:
                    param_groups[0]["params"].append(p)
            optimizer = optim.AdamW(param_groups, lr=lr, betas=(momentum, 0.999), eps=1e-08)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(momentum, 0.999), eps=1e-08)
    elif opt == 'AdamAdamW':
        # weight_decay_coupled + weight_decay_decoupled = weight_decay
        # ==> weight_decay_decoupled = weight_decay - weight_decay_coupled
        assert(weight_decay >= weight_decay_coupled), "Weight decay (total) must be larger than Weight decay decoupled!"

        weight_decay_decoupled = weight_decay - weight_decay_coupled        
        optimizer = AdamAdamW(model.parameters(), lr=lr, betas=(momentum, 0.999), eps=1e-08,
                              weight_decay_coupled=weight_decay_coupled,
                              weight_decay_decoupled=weight_decay_decoupled)

    def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    initial_lr: float,
    peak_lr: float,
    final_lr: float,
    ):
        """Create a schedule with linear warmup and cosine decay."""
        def lr_lambda(current_step: int):
    
            if current_step < num_warmup_steps:
                # Linear warmup: from initial_lr to peak_lr
                warmup_ratio = current_step / max(1, num_warmup_steps)
                return (initial_lr + (peak_lr - initial_lr) * warmup_ratio) / peak_lr
            else:
                # Cosine decay: from peak_lr to final_lr
                progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
                decay_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (peak_lr * decay_ratio + final_lr * (1 - decay_ratio)) / peak_lr
    
        return LambdaLR(optimizer, lr_lambda)
    
    # Cosine LR scheduler for AdamW on ViT
    if lr_schedule == 'cosine_decay':

        warmup_steps = int(warmup_length * total_training_steps)  # 5% warmup
        
        # lr_scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=warmup_steps,
        #     num_training_steps=total_training_steps
        # )
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps, initial_lr=1e-7, peak_lr=lr, final_lr=1e-5)
    elif lr_schedule == 'step_decay':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epochs_lr_decay, lr_decay)
    elif lr_schedule == 'none':
        lr_scheduler = None
    else:
        ValueError(f'LR schedule {lr_schedule} is not implemented!')


    return optimizer, lr_scheduler