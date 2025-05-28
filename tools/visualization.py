# tools/visualization.py
import matplotlib.pyplot as plt
import os
import numpy as np
import wandb
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

seq_len = 48
time_labels = [f"{int(i//2):02d}:{'30' if i%2 else '00'}" for i in range(seq_len)]

# 初始化 wandb 实验
def init_wandb(config):
    wandb.init(project=config['experiment'], config=config)

# 上传 loss 日志
def log_loss(loss, epoch):
    wandb.log({"loss": loss, "epoch": epoch})

# 从 CSV 加载真实数据（用于可视化对比）
def load_original_from_csv(csv_path, window_size):
    df = pd.read_csv(csv_path)
    # data = df.values.astype(np.float32)
    if not np.issubdtype(df.dtypes[0], np.number):
        df = df.iloc[:, 1:]
    data = df.values.astype(np.float32)
    return data[-window_size:].reshape(1, window_size, -1)


# 对比生成样本与真实数据，可选上传 wandb 图像
def plot_samples(samples, save_dir, original=None):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))
    
    if original is not None:
        for i in range(original.shape[-1]):
            plt.plot(original[-1, :, i], linestyle='--', alpha=0.5, label=f"Original Dim {i}")
    
    for i, s in enumerate(samples):
        for j in range(s.shape[1]):
            plt.plot(s[:, j], label=f"Sample {i} Dim {j}", alpha=0.7)
    
    plt.title("Generated Samples vs Original")
    plt.legend(loc="upper right", ncol=4, fontsize=6)
    
    save_path = os.path.join(save_dir, "sample_plot.png")
    plt.savefig(save_path)
    wandb.log({"samples": wandb.Image(save_path)})
    print(f"Saved sample plot to {save_path}")
    plt.close()

def plot_multi_dim_samples(samples, save_dir, original=None, timestamps=None):
    """
    绘制多维度时间序列数据的对比图，每个维度单独显示
    Args:
        samples: 生成的样本数据，形状为 (n_samples, seq_len, n_dims)
        save_dir: 保存图像的目录
        original: 原始数据，形状为 (1, seq_len, n_dims)
        timestamps: 时间戳数据，如果为None则使用索引作为x轴
    """
    os.makedirs(save_dir, exist_ok=True)
    n_dims = samples[0].shape[1]
    
    # 如果没有提供时间戳，使用索引
    if timestamps is None:
        timestamps = np.arange(samples[0].shape[0])
    
    # 为每个维度创建单独的图
    for dim in range(n_dims):
        plt.figure(figsize=(12, 6))
        
        # 绘制原始数据
        if original is not None:
            plt.plot(timestamps, original[0, :, dim], 'k--', 
                    label='Original', linewidth=2, alpha=0.7)
        
        # 绘制生成的样本
        for i, sample in enumerate(samples):
            plt.plot(timestamps, sample[:, dim], 
                    label=f'Sample {i+1}', alpha=0.5)
        
        plt.title(f'Dimension {dim+1} Comparison')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'dim_{dim+1}_comparison.png')
        plt.savefig(save_path)
        wandb.log({f'dim_{dim+1}_comparison': wandb.Image(save_path)})
        plt.close()

def plot_pca_visualization(samples, save_dir, original=None):
    """
    使用PCA将数据降维到3D进行可视化
    Args:
        samples: 生成的样本数据，形状为 (n_samples, seq_len, n_dims)
        save_dir: 保存图像的目录
        original: 原始数据，形状为 (1, seq_len, n_dims)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 重塑数据以适应PCA
    n_samples = len(samples)
    seq_len = samples[0].shape[0]
    n_dims = samples[0].shape[1]
    
    # 将所有样本展平
    all_data = np.vstack([sample.reshape(seq_len, n_dims) for sample in samples])
    if original is not None:
        all_data = np.vstack([all_data, original[0]])
    
    # 执行PCA降维
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(all_data)
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制生成的样本
    for i in range(n_samples):
        start_idx = i * seq_len
        end_idx = (i + 1) * seq_len
        ax.plot(reduced_data[start_idx:end_idx, 0],
                reduced_data[start_idx:end_idx, 1],
                reduced_data[start_idx:end_idx, 2],
                label=f'Sample {i+1}', alpha=0.7)
    
    # 绘制原始数据
    if original is not None:
        original_reduced = reduced_data[-seq_len:]
        ax.plot(original_reduced[:, 0],
                original_reduced[:, 1],
                original_reduced[:, 2],
                'k--', label='Original', linewidth=2)
    
    ax.set_title('PCA 3D Visualization')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 保存图像
    save_path = os.path.join(save_dir, 'pca_3d_visualization.png')
    plt.savefig(save_path)
    wandb.log({'pca_3d_visualization': wandb.Image(save_path)})
    plt.close()
    
def plot_fancy_samples(samples, original, save_path, ylabel="Electricity Consumption [kWh]"):
    """
    samples: (n_samples, seq_len)
    original: (n_original, seq_len)
    """
    seq_len = samples.shape[1]
    time_labels = [f"{int(i//2):02d}:{'30' if i%2 else '00'}" for i in range(seq_len)]
    x = np.arange(seq_len)

    plt.figure(figsize=(5, 5))
    # 画原始数据（蓝色，半透明）
    for i in range(original.shape[0]):
        plt.plot(x, original[i], color='blue', alpha=0.2)
    # 画生成数据（红色，半透明）
    for i in range(samples.shape[0]):
        plt.plot(x, samples[i], color='red', alpha=0.2)
    # 均值线
    plt.plot(x, original.mean(axis=0), color='blue', lw=2, label='Original Mean')
    plt.plot(x, samples.mean(axis=0), color='red', lw=2, label='Generated Mean')

    plt.title("WGAN", fontsize=14)
    plt.xlabel("Hour of the Day [60 minutes]", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(np.linspace(0, seq_len-1, 7), ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    
    import numpy as np
import matplotlib.pyplot as plt

def plot_fancy_dimwise(original, samples, save_dir, ylabel="Electricity Consumption [kWh]"):
    """
    original: (n_original, seq_len, n_dims)
    samples: (n_samples, seq_len, n_dims)
    save_dir: 保存图片的目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    seq_len = original.shape[1]
    n_dims = original.shape[2]
    time_labels = [f"{int(i//2):02d}:{'30' if i%2 else '00'}" for i in range(seq_len)]
    x = np.arange(seq_len)

    for d in range(n_dims):
        plt.figure(figsize=(6, 5))
        # 原始数据（蓝色，半透明）
        for i in range(original.shape[0]):
            plt.plot(x, original[i, :, d], color='blue', alpha=0.15)
        # 生成数据（红色，半透明）
        for i in range(samples.shape[0]):
            plt.plot(x, samples[i, :, d], color='red', alpha=0.15)
        # 均值线
        plt.plot(x, original[:, :, d].mean(axis=0), color='blue', lw=2, label='Original Mean')
        plt.plot(x, samples[:, :, d].mean(axis=0), color='red', lw=2, label='Generated Mean')

        plt.title(f"Dimension {d+1} Comparison", fontsize=15)
        plt.xlabel("Hour of the Day [60 minutes]", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(np.linspace(0, seq_len-1, 7), ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"fancy_dim_{d+1}.png"), dpi=200)
        plt.close()
        
def plot_and_save_normalized_data(data, save_path="normalized_data_example.png", n_samples=1000):
    """
    data: 归一化后的数据，形状为 (N, seq_len, n_dims) 或 (N, seq_len)
    save_path: 图片保存路径
    n_samples: 可视化前n条
    """
    plt.figure(figsize=(10, 6))
    if data.ndim == 3:
        # 只画第一个维度
        for i in range(min(n_samples, data.shape[0])):
            plt.plot(data[i, :, 0], label=f"Sample {i+1}")
    else:
        for i in range(min(n_samples, data.shape[0])):
            plt.plot(data[i, :], label=f"Sample {i+1}")
    plt.title("Normalized Training Data Example")
    plt.xlabel(" 7 Features of Windenergy")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"已保存归一化数据可视化到 {save_path}")
# 定义绘图函数（上述step1）
def plot_samples(original, generated, epoch, save_path="./generated_images", show_plot=False):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    # plt.plot(original[:, 0], original[:, 1], alpha=0.5)
    plt.title("Original Data")
    plt.figure(figsize=(10, 6))
    plt.plot(original.T, alpha = 0.05, c ='b')
    plt.title("Original Wind Energy Data")
    plt.xlabel("Feature 2")

    plt.subplot(1, 2, 2)
    # plt.plot(generated[:, 0], generated[:, 1], alpha=0.5)
    plt.figure(figsize=(10, 6))
    plt.plot(generated.T, alpha = 0.05, c ='b')
    plt.title("Original Wind Energy Data")
    plt.xlabel("Feature 2")
    plt.title(f"Generated Data (Epoch {epoch})")

    if epoch % 50 == 0:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, f"wandb_sample_epoch{epoch}.png")
        plt.tight_layout()
        plt.savefig(filepath)
        wandb.log({"samples": wandb.Image(filepath, caption=f"Epoch {epoch}"), "epoch": epoch})
        print(f"✅ Image saved at {filepath}")

        if show_plot:
            plt.show()
    
    plt.close()
