# main.py
import wandb
import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import TimeSeriesDataset
from model.diffusion_transformer import TransformerDiffusionModel
from tools.lossfunction import get_loss_fn
from tools.optimization import get_optimizer
from tools.visualization import plot_samples, init_wandb, log_loss, load_original_from_csv
from tools.preprocess import load_and_clean_csv, normalize_series
from tools.visualization import plot_fancy_dimwise, plot_and_save_normalized_data,plot_samples
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Load & preprocess data
raw_data = load_and_clean_csv("data/train_set_wind.csv")
data, scaler = normalize_series(raw_data)

# Dataset
dataset = TimeSeriesDataset(data, cfg['model']['seq_len'])
loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)

# Model
model = TransformerDiffusionModel(
    input_dim=cfg['model']['input_dim'],
    model_dim=cfg['model']['model_dim'],
    seq_len=cfg['model']['seq_len'],
    timesteps=cfg['model']['timesteps'],
    num_layers=cfg['model']['num_layers'],
    dropout=cfg['model'].get('dropout', 0.1)
).cuda()
print("amount of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


optimizer = get_optimizer(model, cfg)
loss_fn = get_loss_fn(cfg['loss'])

init_wandb(cfg)

# Training
for epoch in range(cfg['training']['epochs']):
    for batch in loader:
        batch = batch.cuda()
        optimizer.zero_grad()
        loss = model.loss_fn(batch, loss_fn)
        if not torch.isfinite(loss):
            print("NaN detected, skipping batch.")
            continue
        loss.backward()
        optimizer.step()

    log_loss(loss.item(), epoch)
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# Sample & plot
samples = model.sample(cfg['sample']['n_samples'])
# 转为numpy并展平
samples_np = samples.cpu().numpy().reshape(-1, samples.shape[-1])  # (n_samples * seq_len, n_dims)
# 反归一化
from tools.preprocess import inverse_normalize
samples_denorm = inverse_normalize(samples_np, scaler)  # (n_samples * seq_len, n_dims)

# 恢复原shape
samples_denorm = samples_denorm.reshape(samples.shape)  # (n_samples, seq_len, n_dims)

original =raw_data[:1000, :]  # 取前1000个样本作为原始数据
generated = samples_denorm[:1000, :].mean(axis=1) # 取前1000个生成样本
print(f"Generated samples shape: {generated.shape}, Original shape: {original.shape}")
# Plotting  
# original = load_original_from_csv("data/train_set_wind.csv", cfg['model']['seq_len'])
# plot_samples(samples_denorm, cfg['sample']['save_path'], original=original)
# plot_multi_dim_samples(samples_denorm, cfg['sample']['save_path'], original=original)
# plot_fancy_samples(samples_denorm, original, cfg['sample']['save_path'])
# plot_fancy_dimwise(original, samples_denorm, cfg['sample']['save_path']) 
# # 导入模块...
# 模型定义...
# 数据准备...
# wandb初始化...


# 训练循环（上述step2）
for epoch in range(cfg['training']['epochs']):
    for batch in loader:
        batch = batch.cuda()
        optimizer.zero_grad()
        loss = model.loss_fn(batch, loss_fn)
        if not torch.isfinite(loss):
            print("NaN detected, skipping batch.")
            continue
        loss.backward()
        optimizer.step()

    log_loss(loss.item(), epoch)
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    if epoch % 20 == 0:
        samples = model.sample(cfg['sample']['n_samples'])
        samples_np = samples.cpu().numpy().reshape(-1, samples.shape[-1])

        from tools.preprocess import inverse_normalize
        samples_denorm = inverse_normalize(samples_np, scaler)
        samples_denorm = samples_denorm.reshape(samples.shape)

        # original = raw_data[:1000, :]
        # generated = samples_denorm[:1000, :].mean(axis=1)

        plot_samples(original, generated, epoch, show_plot=True)

print("✅ 训练完成！")
