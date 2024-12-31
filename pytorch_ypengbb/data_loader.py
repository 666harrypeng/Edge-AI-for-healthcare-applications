import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class NABDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return sample, label
        return sample

def create_dataloader(config, mode='train'):
    # Load the preprocessed dataset
    dataset_path = f"{config['data_dir']}/{config['dataset']}.npz"
    data = np.load(dataset_path)

    if mode == 'train':
        data_split = data['training']
    elif mode == 'val':
        data_split = data['val']
    elif mode == 'test':
        data_split = data['test']
    elif mode == 'anomaly_detection':
        data_split = data['full_data_rolling']
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create Dataset and DataLoader
    data_split = torch.tensor(data_split, dtype=torch.float32).unsqueeze(-1)
    dataset = NABDataset(data_split)
    
    if mode == 'anomaly_detection':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=(mode == 'train'))
    
    return dataloader

# plotting the normalized time series (not effective)
def plot_normalized_time_series(dataset, config):
    data = np.load(os.path.join('../datasets/NAB-known-anomaly/', f"{config['dataset']}.npz"))
    train_m = data['train_m']
    train_std = data['train_std']
    readings_normalized = (data['readings'] - train_m) / train_std

    # Plot normalized data
    plt.figure(figsize=(18, 4))
    plt.plot(data['t'], readings_normalized)
    plt.title(f"Normalized {config['dataset']} Data (Mean: {train_m}, Std: {train_std})")
    plt.xlabel("Timestamps")
    plt.ylabel("Normalized Readings")
    plt.grid(True)
    plt.show()