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
        self.patience = config.get('patience', 20)
        self.best_val_loss = float('inf') 
        self.patience_counter = 0
        self.checkpoint_path = os.path.join(config['checkpoint_dir_lstm'], 'lstm_best_model.pth')
        self.look_back_num = config['look_back_num']
        
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
            
            if self.epoch % self.config["visualization_interval"] == 0:
                self.visualize_pred(batch_val_y, val_outputs, mode='val')
                
        return val_total_loss / len(self.val_loader)
    
                
    def find_anomalies(self, data_loader):
        '''
        : data_loader -> rolling windows list
        We will find anomalies based on the error values
        '''
        self.model.eval()

        prediction_errors = []
        error_deviation_ratios = []
        
        threshold_percentage = self.config['threshold_percent']
        prediction_error_deviation_ratio_threshold = self.config['prediction_error_deviation_ratio_threshold'] # 0 ~ 1
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # predict (infer from model)
                pred_outputs = self.model.lstm_infer(batch_X)
                
                # de-normalize data values (broadcasting) (optional)
                # pred_outputs = pred_outputs.cpu().detach().numpy() * self.data_std + self.data_mean
                # targets = batch_y.cpu().detach().numpy() * self.data_std + self.data_mean
                pred_outputs = pred_outputs.cpu().detach().numpy()
                targets = batch_y.cpu().detach().numpy()
                
                # calculate errors (scope unit: window by window, not single points comparison)
                '''
                raw data is split into non-overlapping rolling windows
                these non-overlapping windows are again grouped into "overlapped" rolling groups
                if error is calculated by MSE, then we will raise anomaly by the unit of a sole window
                '''
                pred_error = np.mean((pred_outputs - targets) ** 2)
                prediction_errors.append(pred_error)
                
                epsilon = 1e-10  # Small constant to avoid division by zero
                error_deviation_ratio = np.abs((np.mean(pred_error) - np.mean(targets)) / (np.mean(targets) + epsilon))
                error_deviation_ratios.append(error_deviation_ratio)
                
            self.prediction_errors = np.array(prediction_errors)
            anomaly_threshold = np.percentile(self.prediction_errors, threshold_percentage)
            print(f"finish computing threshold at {threshold_percentage}% anomaly rate: {anomaly_threshold:.5f}" )
            
            # get global indices for predicted anomaly "window" in this current full dataset
            # ATTENTION: The first lstm_look_back_num windows in the test_loader are used as input context
            # and not included in the anomaly_scores list, so we need to offset the indices
            raw_predicted_anomalies_window_count = 0
            for i, prediction_err in enumerate(self.prediction_errors):
                if prediction_err >= anomaly_threshold:
                    raw_predicted_anomalies_window_count += 1
                    if error_deviation_ratios[i] >= prediction_error_deviation_ratio_threshold:
                        # Calculate the actual window index with offset
                        window_idx = i + self.look_back_num
                        # Only add indices that are within the valid range
                        if window_idx < len(self.prediction_errors) + self.look_back_num:
                            self.predicted_anomalies_window_idx.append(window_idx)
            print(f"number of predicted anomalies windows: {len(self.predicted_anomalies_window_idx)} (raw anomaly window number: {raw_predicted_anomalies_window_count}; prediction error deviation ratio threshold: {prediction_error_deviation_ratio_threshold})")
            self.predicted_anomalies_window_idx = np.array(self.predicted_anomalies_window_idx) 
            
        return self.prediction_errors, self.predicted_anomalies_window_idx
        
    def visualize_anomalies(self, full_data_used, adj_true_anomaly_intervals, predicted_anomalies_window_idx, recall, precision=None, f_beta=None):
        """
        Visualize anomaly detection results.
        
        Args:
            full_data_used: The full dataset used for anomaly detection
            adj_true_anomaly_intervals: List of true anomaly intervals
            predicted_anomalies_window_idx: List of predicted anomaly window indices
            recall: Detection rate (recall)
            precision: Precision score
            f_beta: F-beta score
        """
        plt.figure(figsize=(20, 8))
        color_list = ['#4292c6','#084594', '#88419d', 'mediumpurple', '#F8AC8C', '#F6CAE5', '#96CCCB']
        for modal_idx in range(full_data_used.shape[1]):
            plt.plot(full_data_used[:, modal_idx], label=f"Full_data_modal_{modal_idx+1}", color=color_list[modal_idx % len(color_list)], alpha=0.8)
        
        # Highlight true anomalies
        for start, end in adj_true_anomaly_intervals:
            plt.axvspan(start, end, color='red', alpha=0.2, label='True Anomaly Windows' if start == adj_true_anomaly_intervals[0][0] else "")
                        
        # Highlight predicted anomalies
        for idx in predicted_anomalies_window_idx:
            start = idx * self.config['l_win']
            end = (idx + 1) * self.config['l_win']
            plt.axvspan(start, end, color='green', alpha=0.2, label='Predicted Anomaly Windows' if start == predicted_anomalies_window_idx[0] else "")
            
        # Define the saved path
        exp_dataset = self.config['dataset']
        saved_path = os.path.join(self.config['result_dir'], f"anomaly_detection_comparison_thre{self.config['threshold_percent']}.png")
        
        # Create title with metrics
        title = f"{exp_dataset} - LSTM - Anomaly Detection Results\n(Recall: {recall:.4f}"
        if precision is not None:
            title += f", Precision: {precision:.4f}"
        if f_beta is not None:
            title += f", F-beta: {f_beta:.4f} (beta={self.config.get('f_beta', 2)})"
        if len(adj_true_anomaly_intervals) <= 0:
            title += f"\nNo True Anomaly Windows"
        title += ")"
        
        plt.title(title)
        plt.xlabel("Timestamps")
        plt.ylabel("Normalized Readings")
        plt.legend()
        plt.tight_layout()
        plt.savefig(saved_path)
        plt.close()    
        
        
        
    def visualize_pred(self, batch_y, outputs, mode):
        outputs = outputs.cpu().detach().numpy()   
        batch_y = batch_y.cpu().detach().numpy()
        outputs = np.transpose(outputs, (0, 2, 1)) # [batch_size, input_dim, l_win]
        batch_y = np.transpose(batch_y, (0, 2, 1)) # [batch_size, input_dim, l_win]
        
        # plot & comparison
        plot_group_num = 3
        plt.figure(figsize=(12, 10))
        colors = ["dodgerblue", "darkorange", "limegreen", "mediumvioletred", "gold", "darkcyan", "crimson"]
        handles, labels = [], []
        
        for i in range(min(plot_group_num, batch_y.shape[0])):
            plt.subplot(plot_group_num, 1, i + 1)
            for modal_idx in range(outputs.shape[1]):
                if i == 0:
                    gt_line, = plt.plot(batch_y[i, modal_idx], 
                                        label=f"Ground Truth Modal {modal_idx+1}", 
                                        linestyle='solid',
                                        color=colors[modal_idx])
                    pred_line, = plt.plot(outputs[i, modal_idx],
                                        label=f"Predicted Modal {modal_idx+1}",
                                        linestyle='dashed', 
                                        color=colors[modal_idx])
                    handles.extend([gt_line, pred_line])
                    labels.extend([f"Ground Truth Modal {modal_idx+1}", f"Predicted Modal {modal_idx+1}"])
                else:   # Other subplots: skip labels
                    plt.plot(batch_y[i, modal_idx], 
                        linestyle='solid', 
                        color=colors[modal_idx])
                    plt.plot(outputs[i, modal_idx],
                        linestyle='dashed', 
                        color=colors[modal_idx])
                    
            plt.title(f"Sample {i+1}", 
                        y=-0.25,    # put title under the subplot
                        loc='center', 
                        fontsize=10)
        
        # add a general title for the whole figure
        plt.suptitle(f"LSTM Prediction Visualization ({mode.capitalize()}) - Epoch {self.epoch}", 
                    y=0.95,
                    fontweight='bold',
                    fontsize=14)
        
        # Add a single legend below all subplots
        plt.figlegend(handles, labels, 
                    loc='lower center', 
                    ncol=outputs.shape[1],  # Number of columns in legend
                    bbox_to_anchor=(0.5, 0.02)  # Adjust vertical position
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)  # Make space for the legend
        plt.savefig(os.path.join(self.config['result_dir'], f"lstm_predictions_mode_{mode}_epoch_{self.epoch}.png"))
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