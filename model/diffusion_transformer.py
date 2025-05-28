# model/diffusion_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerDiffusionModel(nn.Module):
    def __init__(self, input_dim, model_dim, seq_len, timesteps, num_layers=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_emb = SinusoidalPosEmb(model_dim)
        self.time_proj = nn.Linear(model_dim, model_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(model_dim, heads=4, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(model_dim, input_dim)

        self.register_buffer("betas", torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - self.alphas_cumprod))

    def forward(self, x, t):
        b, seq, dim = x.shape
        t_emb = self.time_proj(self.time_emb(t))
        t_emb = t_emb.unsqueeze(1).repeat(1, seq, 1)
        x = self.input_proj(x) + t_emb
        x = self.transformer(x)
        return self.output_proj(x)

    def loss_fn(self, x0, loss_fn):
        bsz = x0.size(0)
        device = x0.device
        t = torch.randint(0, len(self.betas), (bsz,), device=device)
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        x_t = sqrt_alpha * x0 + sqrt_1ma * noise
        pred = self(x_t, t)
        return loss_fn(pred, noise)

    @torch.no_grad()
    def sample(self, n_samples):
        x = torch.randn(n_samples, self.seq_len, self.output_proj.out_features).to(next(self.parameters()).device)
        for t in reversed(range(len(self.betas))):
            alpha = self.alphas[t]
            beta = self.betas[t]
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_1ma = torch.sqrt(1 - alpha)
            t_tensor = torch.full((n_samples,), t, device=x.device, dtype=torch.long)
            pred_noise = self(x, t_tensor)
            x = (x - beta * pred_noise / sqrt_1ma) / sqrt_alpha
            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)
        return x.cpu()
