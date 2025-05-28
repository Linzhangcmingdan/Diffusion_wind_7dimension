# tools/optimization.py
import torch

def get_optimizer(model, cfg):
    if cfg['training']['optimizer'] == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']))
    elif cfg['training']['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    else:
        raise ValueError(f"Unsupported optimizer {cfg['training']['optimizer']}")
