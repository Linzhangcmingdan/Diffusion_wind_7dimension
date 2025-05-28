# data/dataset.py
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.samples = []
        for i in range(len(data) - window_size):
            self.samples.append(data[i:i + window_size])
        self.samples = torch.tensor(self.samples, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
