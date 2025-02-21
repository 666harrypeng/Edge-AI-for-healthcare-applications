import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt


class LSTMTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader, config, data_mean, data_std):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.config = config
        self.device = config['device']
        self.criterion = nn.MSELoss()
        self.data_mean = data_mean
        self.data_std = data_std
        self.prediction_errors = []
        self.predicted_anomalies_window_idx = []

        self.epoch = 0
        self.lstm_batch_size = config['lstm_batch_size']
        self.patience = config.get('patience', 20)
        self.best_val_loss = float('inf') 
        self.patience_counter = 0
        self.checkpoint_path = os.path.join(config['checkpoint_dir_lstm'], 'lstm_best_model.pth')
        
    def train(self):
        self.train_losses, self.val_losses = [], []
        with tqdm(total=self.config['num_epochs_lstm'], desc="LSTM Training") as pbar:
            for epoch in range(self.config['num_epochs_lstm']):
                self.epoch = epoch
                train_loss = self._train_epoch()
                val_loss = self._validate_epoch()
                
                # Early stopping & checkpointing
                if val_loss < self.best_val_loss:
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.patience_counter = 0
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_lstm']}: Validation loss improved from {self.best_val_loss:.4f} --> {val_loss:.4f}. New best model saved!")
                    self.best_val_loss = val_loss
                else:
                    self.patience_counter += 1
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_lstm']}: Validation loss did not improve @ {val_loss:.4f}. Patience: {self.patience_counter}/{self.patience}")
                    
                pbar.set_postfix({"Train Loss" : train_loss, "Val Loss" : val_loss})
                pbar.update(1)
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                if self.patience_counter >= self.patience:
                    print("Early Stopping Triggered!")
                    break
        
        self.save_loss_curve(self.train_losses, self.val_losses, model_name="lstm")
        print(f">>> LSTM Training complete. Best model saved at {self.checkpoint_path} <<<")
        
        
    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            outputs = self.model(batch_X, batch_y)
            loss = self.criterion(outputs, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        if self.epoch % self.config["visualization_interval"] == 0:
            outputs, batch_y = outputs[0], batch_y[0]
            self.visualize_pred(outputs, batch_y, mode='train')
        
        return total_loss / len(self.train_loader)
    

    def _validate_epoch(self):
        self.model.eval()
        val_total_loss = 0  
        
        with torch.no_grad():
            for batch_val_X, batch_val_y in self.val_loader:
                batch_val_X, batch_val_y = batch_val_X.to(self.device), batch_val_y.to(self.device)
                val_outputs = self.model(batch_val_X, batch_val_y)
                val_loss = self.criterion(val_outputs, batch_val_y)
                val_total_loss += val_loss.item()
                
        return val_total_loss / len(self.val_loader)
    
                
    def find_anomalies(self, data_loader):
        self.model.eval()
        prediction_errors = []
        threshold_percentage = self.config['threshold_percent']
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                pred_outputs = self.model.lstm_infer(batch_X)
                
                # de-normalize data values (broadcasting)
                pred_outputs = pred_outputs.cpu().detach().numpy() * self.data_std + self.data_mean
                targets = batch_y.cpu().detach().numpy() * self.data_std + self.data_mean
                
                # calculate errors (scope unit: window by window, not single points comparison)
                pred_error = np.mean((pred_outputs - targets) ** 2)
                prediction_errors.append(pred_error)
            
            self.prediction_errors = np.array(prediction_errors)
            anomaly_threshold = np.percentile(self.prediction_errors, threshold_percentage)
            print(f"finish computing threshold at {threshold_percentage}% anomaly rate: {anomaly_threshold:.5f}" )
            
            self.predicted_anomalies_window_idx = np.array([i for i, prediction_err in enumerate(self.prediction_errors) if prediction_err >= anomaly_threshold])
            
        return self.prediction_errors, self.predicted_anomalies_window_idx
        
    def visualize_anomalies(self, full_data_used, adj_true_anomaly_intervals, predicted_anomalies_window_idx, recall):
        plt.figure(figsize=(18 ,8))
        color_list = ['steelblue', 'seagreen', 'darkturquoise', 'mediumslateblue', 'cadetblue', 'teal', 'darkorange', 'purple', 'mediumorchid', 'mediumpurple']
        for modal_idx in range(full_data_used.shape[1]):
            plt.plot(full_data_used[:, modal_idx], label=f"Full_data_modal_{modal_idx+1}", color=color_list[modal_idx])
        
        # Highlight true anomalies
        y_min, y_max = plt.ylim()
        for adj_start, adj_end in adj_true_anomaly_intervals:
            plt.vlines(x=adj_start, ymin=y_min, ymax=y_max, color='gray', linestyles='dashed', alpha=0.5)
            plt.vlines(x=adj_end, ymin=y_min, ymax=y_max, color='gray', linestyles='dashed', alpha=0.5)
            plt.axvspan(adj_start, adj_end, color='gray', alpha=0.2, label='True Anomaly Windows' if adj_start == adj_true_anomaly_intervals[0][0] else "")
                        
        # Highlight predicted anomalies
        predicted_anomalies_window_intervals = [(predicted_anomalies_window_idx[i] * self.config['l_win'], (predicted_anomalies_window_idx[i] + 1) * self.config['l_win']) for i in range(len(predicted_anomalies_window_idx))]
        for adj_start, adj_end in predicted_anomalies_window_intervals:
            plt.vlines(x=adj_start, ymin=y_min, ymax=y_max, color='green', linestyles=':', alpha=0.7)
            plt.vlines(x=adj_end, ymin=y_min, ymax=y_max, color='green', linestyles=':', alpha=0.7)
            plt.axvspan(adj_start, adj_end, color='green', alpha=0.2, label='Predicted Anomaly Windows' if adj_start == predicted_anomalies_window_intervals[0][0] else "")
            
        
        exp_dataset = self.config['dataset']
        plt.title(f"{exp_dataset} - LSTM - Anomaly Detection Visualization\nRecall:{recall:.4f}" )
        plt.xlabel("Timestamps")
        plt.ylabel("Normalized Readings")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['result_dir'], "anomaly_detection_comparison.png"))
        plt.close()    
        
        
        
    def visualize_pred(self, outputs, batch_y, mode):
        outputs = outputs.cpu().detach().numpy()   
        batch_y = batch_y.cpu().detach().numpy()  
        
        plt.figure(figsize=(12, 6))
        color_list = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
        for modal_idx in range(outputs.shape[1]):
            plt.plot(outputs[:, modal_idx], label=f"prediction_modal_{modal_idx+1}", linestyle="dashed", color=color_list[modal_idx])
            plt.plot(batch_y[:, modal_idx], label=f"raw_data_modal_{modal_idx+1}", linestyle='solid', color=color_list[modal_idx])
        plt.xlabel("time steps")
        plt.ylabel("normalized data values")
        plt.title(f"LSTM Prediction - {mode} Epoch {self.epoch}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['result_dir'], f"LSTM_prediction_{mode}_epoch_{self.epoch}.png"))
        plt.close()
        
        
        
    def save_loss_curve(self, train_losses, val_losses, model_name):
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name.upper()} Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(self.config['result_dir'], f"{model_name}_loss_curve.png"))
        plt.close()