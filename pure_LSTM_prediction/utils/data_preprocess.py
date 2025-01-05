import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset



class Data_preprocessor:
    def __init__(self, args):
        self.args = args
        
    def create_sequences(self, X, y, time_steps=24):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i : i + time_steps])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    def generate_dataset(self):
        device = self.args.device

        df = pd.read_csv(self.args.data_path)
        
        # Drop the timestamp column (first column)
        data_no_timestamp = df.iloc[:, 1:].values 
        
        scaler = MinMaxScaler(feature_range=(-1, 1))    # Normalize the data features (for model training) (linear mapping normalization)
        data_no_timestamp[:, :-1] = scaler.fit_transform(data_no_timestamp[:, :-1])  # Only scale the sensor data
        
        # Split data into features and target
        X = data_no_timestamp[:, :-1]  # (sensors) data
        y = data_no_timestamp[:, -1]   # target
        
        # Reshape for LSTM [samples, time steps, features]
        X_seq, y_seq = self.create_sequences(X, y, self.args.time_steps)

        # Split into training, validation and testing sets
        # X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=True)
        X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, random_state=self.args.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
        
        # convert to PyTorch tensors and move to device
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        print("X_train shape = ", X_train.shape)
        print("y_train shape = ", y_train.shape)
        print("X_val shape = ", X_val.shape)
        print("y_val shape = ", y_val.shape)
        print("X_test shape = ", X_test.shape)
        print("y_test shape = ", y_test.shape)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloader(self, X_train, y_train, X_val, y_val, X_test, y_test):
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.args.batch_size, shuffle=True) # Better use shuffle -> better training (time data will be kept within each sequence & batch) 
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=self.args.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    


