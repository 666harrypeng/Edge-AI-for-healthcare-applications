import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
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

    if mode == 'train' or mode == 'lstm_train':
        data_split = data['training']
    elif mode == 'val' or mode == 'lstm_val':
        data_split = data['val']
    elif mode == 'test' or mode == 'lstm_test':
        data_split = data['test']
    elif mode == 'anomaly_detection' or mode == 'full_data_rolling':
        data_split = data['full_data_rolling']
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create Dataset and DataLoader
    data_split = torch.tensor(data_split, dtype=torch.float32).unsqueeze(-1) # add width dimension (each sensor reading is 1D -> widith is 1)
    dataset = CustomDataset(data_split)
    
    if mode == 'lstm_train' or mode == 'lstm_val' or mode == 'lstm_test' or mode == 'anomaly_detection' or mode == 'full_data_rolling':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=(mode == 'train'))    # shuffle only for training dataloader
    
    return dataloader