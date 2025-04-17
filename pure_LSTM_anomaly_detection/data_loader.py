import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    elif mode == 'anomaly_detection' or mode == 'full_data_rolling':
        data_split = data['full_data_rolling']
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    data_split = data_split.reshape(-1, config['l_win'], config['input_size'])
    
    X, y = [], []
    look_back_num = config['look_back_num']
    for i in range(len(data_split) - look_back_num - 1):
        X.append(data_split[i : i+look_back_num])
        y.append(data_split[i + look_back_num])
        
    # Create Dataset and DataLoader
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y) ,dtype=torch.float32)
    
    dataset = CustomDataset(X, y)
    
    if mode == 'test' or mode == 'full_data_rolling':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=config['lstm_batch_size'], shuffle=(mode == 'train'))
    
    print(f'mode - {mode}- DataLoader created successfully')
    
    return dataloader
