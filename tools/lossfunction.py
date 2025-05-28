# tools/lossfunction.py
import torch
import torch.nn.functional as F

def get_loss_fn(config):
    if config['type'] == 'mse':
        return lambda pred, target: F.mse_loss(pred, target)
    elif config['type'] == 'mae':
        return lambda pred, target: F.l1_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss type {config['type']}")
loss_fn = torch.nn.L1Loss()