import numpy as np
import torch
from torch.utils.data import Dataset

class OceanDataset(Dataset):
    def __init__(self, swh, u10, v10, seq_len=12, pred_len=1, normalize=True):
        # Derived features
        wind_speed = np.sqrt(u10**2 + v10**2)
        wind_dir   = np.arctan2(v10, u10)

        # Stack variables: (T, C, H, W)
        data = np.stack([swh, u10, v10, wind_speed, wind_dir], axis=1)

        if normalize:
            mean = data.mean(axis=(0,2,3), keepdims=True)
            std  = data.std(axis=(0,2,3), keepdims=True) + 1e-6
            data = (data - mean) / std
            self.mean, self.std = mean, std
        else:
            self.mean, self.std = None, None

        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.T = data.shape[0]

    def __len__(self):
        return self.T - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx+self.seq_len]          
        y = self.data[idx+self.seq_len : idx+self.seq_len+self.pred_len, 0:1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
