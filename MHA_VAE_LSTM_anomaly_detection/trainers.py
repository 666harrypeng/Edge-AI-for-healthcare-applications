import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt


class VAETrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config, device, mean, std):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.data_mean = mean
        self.data_std = std
        self.epoch = 0
        self.patience = config.get('vae_patience', 50)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_path = os.path.join(config['checkpoint_dir'], 'vae_best_model.pth')
        self.vae_learning_rate = config.get('vae_learning_rate', 0.0005)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, 
            mode='min', 
            factor=0.5,
            patience=50,
            min_lr=0.000001,
            cooldown=0,
            )
        
    def train(self):
        self.train_losses, self.val_losses = [], []
        with tqdm(total=self.config['num_epochs_vae'], desc="VAE Training") as pbar:
            for epoch in range(self.config['num_epochs_vae']):
                self.epoch = epoch
                train_loss = self._train_epoch()
                val_loss = self._validate_epoch()
                
                # update learning rate
                self.scheduler.step(val_loss)
                
                # Early stopping and checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.patience_counter = 0
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_vae']}: New best model saved with validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_vae']}: lr: {self.scheduler.get_last_lr()[0]}, Validation loss did not improve. Patience: {self.patience_counter}/{self.patience}")
                    
                pbar.set_postfix({'Train Loss': train_loss, 'Val Loss': val_loss})
                pbar.update(1)
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered!")
                    break
                
        self.save_loss_curve(self.train_losses, self.val_losses, model_name='vae')
        print(f">>> VAE Training complete. Best model saved at: {self.checkpoint_path}")
        
    def _train_epoch(self):
        self.model.train()
        total_loss = 0
            
        for batch in self.train_loader:
            batch = batch.to(self.device)
            batch = batch.permute(0, 2, 1, 3)
            self.optimizer.zero_grad()  
            reconstructed, mean, logvar = self.model(batch)
            loss = self._vae_loss(batch, reconstructed, mean, logvar)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                batch = batch.permute(0, 2, 1, 3)
                reconstructed, mean, logvar = self.model(batch)
                loss = self._vae_loss(batch, reconstructed, mean, logvar)
                total_loss += loss.item()
            
            if self.epoch % self.config["vae_visualization_interval"] == 0:
                self.visualize_reconstruction(batch, reconstructed, self.epoch, mode='val')
                
        return total_loss / len(self.val_loader)

    def _vae_loss(self, original, reconstructed, mean, logvar):
        """Combined VAE loss with beta-VAE formulation"""
        # Beta parameter for KL divergence
        beta = 0.8
        # Use the model's weighted reconstruction loss
        reconstruction_loss = self.model.weighted_reconstruction_loss(original, reconstructed)
        
        # KL divergence loss (mean over batch)
        kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Combined loss with beta weighting
        return reconstruction_loss + beta * kl_divergence
    
        
    def generate_lstm_embeddings(self, vae_model, train_loader, val_loader):
        """
        Generate embeddings from the VAE's encoder for use in LSTM training.
        Uses only the mean vector from the encoder (no reparameterization) for more stable embeddings.
        """
        train_embeddings, val_embeddings = [], []
        vae_model.eval()

        def _generate_embeddings(data_loader):
            embeddings = []
            with torch.no_grad():
                for batch in data_loader:   # batch_size should be 1, no shuffle, keep sequence order
                    batch = batch.to(self.device)
                    batch = batch.permute(0, 2, 1, 3)
                    
                    # Get mean vector from encoder (without reparameterization)
                    mean, _ = vae_model.encode(batch)
                    
                    mean = mean.squeeze(-1).squeeze(0)
                    mean = mean.permute(1, 0)
                    
                    embeddings.append(mean.cpu())
                    
            return embeddings

        train_embeddings = _generate_embeddings(train_loader)
        val_embeddings = _generate_embeddings(val_loader)
        
        return train_embeddings, val_embeddings

    def find_anomalies(self, test_loader, lstm_model):
        '''
        Use VAE reconstruction error & LSTM prediction error to find anomalies
        '''
        self.model.eval()
        lstm_model.eval()
        anomaly_scores = []
        reconstruction_errors = []
        prediction_errors = []
        prediction_error_deviation_ratios = []
        predicted_anomalies_window_idx = []
        
        window_indices = np.arange(0, self.config['l_win'])
        lstm_look_back_num = self.config['look_back_num']
        threshold_percent = self.config['threshold_percent']
        prediction_error_deviation_ratio_threshold = self.config['prediction_error_deviation_ratio_threshold']
        
        with torch.no_grad():
            ##### Generate VAE reconstruction errors #####
            for i, batch in enumerate(test_loader):  # batch_size should be 1
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                batch = batch.permute(0, 2, 1, 3)
                
                assert batch.shape[1:] == (self.config['input_size'], self.config['l_win'], 1), "Final Visual VAE input shape is not correct!"
                
                # VAE Reconstruction Error - use weighted loss
                reconstructed, _, _ = self.model(batch)
                reconstruction_error = self.model.weighted_reconstruction_loss(batch, reconstructed).cpu().numpy()
                reconstruction_errors.append(reconstruction_error)
            
            reconstruction_errors = reconstruction_errors[lstm_look_back_num : ]
            
            print("\nfinish reconstruction\n")
            
            ########### Generate lstm embeddings ################
            embeddings = []
            for batch in test_loader:   # batch_size should be 1
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                
                batch = batch.permute(0, 2, 1, 3)
                
                assert batch.shape[1:] == (self.config['input_size'], self.config['l_win'], 1), "Final Visual Embeddings input shape is not correct!"

                # Get mean vector from encoder (without reparameterization)
                mean, _ = self.model.encode(batch)
                mean = mean.squeeze(-1).squeeze(0)
                mean = mean.permute(1, 0)
                embeddings.append(mean.cpu())
            
            print("\nfinish lstm embeddings\n")    
            
            # Create lstm sequence
            x_lstm, y_lstm = [], []
            for i in range(len(embeddings) - lstm_look_back_num):
                x_lstm.append(embeddings[i : i + lstm_look_back_num])
                y_lstm.append(embeddings[i + lstm_look_back_num])
            x_lstm = np.array(x_lstm)
            y_lstm = np.array(y_lstm)
            
            # Inference from LSTM
            for i in range(0, len(x_lstm)): # step should be 1
                inputs = torch.tensor(x_lstm[i : i + 1], dtype=torch.float32).to(self.device)
                targets = torch.tensor(y_lstm[i : i + 1], dtype=torch.float32).to(self.device)
                
                # Forward pass through LSTM
                outputs = lstm_model.lstm_infer(inputs)
                
                # reconstruct -> prediction error calculation
                targets = targets.permute(0, 2, 1).unsqueeze(-1)
                outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
                
                # Decode both target and predicted embeddings
                decode_targets_recon = self.model.decode(targets)
                decode_outputs_recon = self.model.decode(outputs)
                
                decode_targets_recon = decode_targets_recon.cpu().detach().numpy()
                decode_outputs_recon = decode_outputs_recon.cpu().detach().numpy()
                
                # Calculate prediction error
                prediction_error = np.mean(((decode_targets_recon - decode_outputs_recon) ** 2))
                prediction_errors.append(prediction_error)
                
                epsilon = 1e-10  # Small constant to avoid division by zero
                prediction_error_deviation_ratio = np.abs((np.mean(prediction_error) - np.mean(decode_targets_recon)) / (np.mean(decode_targets_recon) + epsilon))
                prediction_error_deviation_ratios.append(prediction_error_deviation_ratio)
                
            print("\nfinish prediction\n")

            # Combined Score
            assert len(reconstruction_errors) == len(prediction_errors), "Error list lengths do not match!"
            anomaly_scores = np.array(reconstruction_errors) + 0.5 * np.array(prediction_errors)
            
            # Find anomalies
            anomaly_threshold = np.percentile(anomaly_scores, threshold_percent)
            print(f"finish computing threshold at {threshold_percent}% anomaly rate: {anomaly_threshold:.5f}" )
            
            # get global indices for predicted anomaly "window" in this current full dataset
            # ATTENTION: The first lstm_look_back_num windows in the test_loader are used as input context
            # and not included in the anomaly_scores list, so we need to offset the indices
            raw_predicted_anomalies_window_count = 0
            for i, score in enumerate(anomaly_scores):
                if score >= anomaly_threshold:
                    raw_predicted_anomalies_window_count += 1
                    if prediction_error_deviation_ratios[i] >= prediction_error_deviation_ratio_threshold:
                        # Calculate the actual window index with offset
                        window_idx = i + lstm_look_back_num
                        # Only add indices that are within the valid range
                        if window_idx < len(anomaly_scores) + lstm_look_back_num:
                            predicted_anomalies_window_idx.append(window_idx)
            print(f"number of predicted anomalies windows: {len(predicted_anomalies_window_idx)} (raw anomaly window number: {raw_predicted_anomalies_window_count}; prediction error deviation ratio threshold: {prediction_error_deviation_ratio_threshold})")
            predicted_anomalies_window_idx = np.array(predicted_anomalies_window_idx)
            
        return anomaly_scores, predicted_anomalies_window_idx


    def visualize_anomalies(self, full_data_used, adj_true_anomaly_intervals, predicted_anomalies_window_idx, detect_rate, precision=None, f_beta=None):
        """
        Visualize anomaly detection results.
        
        Args:
            full_data_used: The full dataset used for anomaly detection
            adj_true_anomaly_intervals: List of true anomaly intervals
            predicted_anomalies_window_idx: List of predicted anomaly window indices
            detect_rate: Detection rate (recall)
            precision: Precision score
            f_beta: F-beta score
        """
        plt.figure(figsize=(20, 8))
        color_list = ['#4292c6','#084594', '#88419d', 'mediumpurple', '#F8AC8C', '#F6CAE5', '#96CCCB']
        
        # Plot the data
        for i in range(full_data_used.shape[1]):
            plt.plot(full_data_used[:, i], label=f'Channel {i+1}', alpha=0.8, color=color_list[i % len(color_list)])
        
        # Highlight true anomalies
        for start, end in adj_true_anomaly_intervals:
            plt.axvspan(start, end, color='red', alpha=0.2, label='True Anomaly' if start == adj_true_anomaly_intervals[0][0] else "")
        
        # Highlight predicted anomalies
        for idx in predicted_anomalies_window_idx:
            start = idx * self.config['l_win']
            end = (idx + 1) * self.config['l_win']
            plt.axvspan(start, end, color='green', alpha=0.2, label='Predicted Anomaly' if start == predicted_anomalies_window_idx[0] else "")
        
        # Define the saved path
        exp_dataset = self.config['dataset']
        saved_path = os.path.join(self.config['result_dir'], f"anomaly_detection_comparison_thre{self.config['threshold_percent']}.png")
        
        # Create title with metrics
        title = f"{exp_dataset} - MHA-VAE-LSTM - Anomaly Detection Results\n(Recall: {detect_rate:.4f}"
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
        print(f"Anomaly detection visualization on full dataset saved @ {saved_path}")
    

    def visualize_reconstruction(self, raw_data, reconstructed_data, epoch, mode='train'):
        """
        Visualize and save VAE reconstruction performance.
        :param raw_data: Original input data.
        :param reconstructed_data: Reconstructed data from the VAE.
        :param epoch: Current training epoch.
        :param mode: 'train' or 'val' mode.
        """
        raw_data = raw_data.cpu().detach().numpy()
        reconstructed_data = reconstructed_data.cpu().detach().numpy()
        
        colors = ["dodgerblue", "darkorange", "limegreen", "mediumvioletred", "gold", "darkcyan", "crimson"]
        
        # Initialize variables to store legend handles/labels
        handles, labels = [], []
        
        plot_group_num = 3
        plt.figure(figsize=(12, 10))
        for i in range(min(plot_group_num, raw_data.shape[0])):
            plt.subplot(plot_group_num, 1, i + 1)
            for modal_idx in range(raw_data.shape[1]):  # Loop through each modal/sensor type
                # only add label to legend once
                if i == 0: # first subplot to collect legend handles/labels
                    raw_line, = plt.plot(raw_data[i, modal_idx].squeeze(-1), 
                                        label=f"Raw Data Modal {modal_idx+1}", 
                                        linestyle='solid',
                                        color=colors[modal_idx])
                    recon_line, = plt.plot(reconstructed_data[i, modal_idx].squeeze(-1),
                                        label=f"Reconstructed Modal {modal_idx+1}",
                                        linestyle='dashed', 
                                        color=colors[modal_idx])
                    handles.extend([raw_line, recon_line])
                    labels.extend([f"Raw Modal {modal_idx+1}", f"Recon Modal {modal_idx+1}"])
                else:   # Other subplots: skip labels
                    plt.plot(raw_data[i, modal_idx].squeeze(-1), 
                        linestyle='solid', 
                        color=colors[modal_idx])
                    plt.plot(reconstructed_data[i, modal_idx].squeeze(-1),
                        linestyle='dashed', 
                        color=colors[modal_idx])
                    
            plt.title(f"Sample {i+1}", 
                        y=-0.25,    # put title under the subplot
                        loc='center', 
                        fontsize=10)

        # add a general title for the whole figure
        plt.suptitle(f"VAE Reconstruction Visualization ({mode.capitalize()}) - Epoch {epoch}", 
                    y=0.95,
                    fontweight='bold',
                    fontsize=14)
        
        # Add a single legend below all subplots
        plt.figlegend(handles, labels, 
                    loc='lower center', 
                    ncol=raw_data.shape[1],  # Number of columns in legend
                    bbox_to_anchor=(0.5, 0.02)  # Adjust vertical position
        )
        
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)  # Make space for the legend
        plt.savefig(os.path.join(self.config['result_dir'], f"vae_reconstruction_{mode}_epoch_{epoch}.png"))
        plt.close()
    
    def save_loss_curve(self, train_losses, val_losses, model_name):
        """
        Save the training and validation loss curves.
        :param train_losses: List of training losses.
        :param val_losses: List of validation losses.
        :param model_name: 'vae' or 'lstm'.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name.upper()} Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config['result_dir'], f"{model_name}_loss_curve.png"))
        plt.close()
    
    
class LSTMTrainer:
    def __init__(self, model, lstm_train_embeddings, lstm_val_embeddings, optimizer, config, device, mean, std, vae_model):
        self.model = model
        self.lstm_train_embeddings = lstm_train_embeddings
        self.lstm_val_embeddings = lstm_val_embeddings
        self.look_back_num = config['look_back_num'] 
        self.x_lstm_train = []
        self.y_lstm_train = []
        self.x_lstm_val = []
        self.y_lstm_val = []
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.data_mean = mean
        self.data_std = std
        self.vae_model = vae_model
        self.l_win = config['l_win']
        self.epoch = 0
        self.lstm_batch_size = config['lstm_batch_size']
        self.patience = config.get('lstm_patience', 30)
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
                
                # Early stopping and checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.patience_counter = 0
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_lstm']}: New best model saved with validation loss: {val_loss}") # or -> val_loss:.4f
                else:
                    self.patience_counter += 1
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_lstm']}: Validation loss did not improve. Patience: {self.patience_counter}/{self.patience}")
                    
                pbar.set_postfix({'Train Loss': train_loss, 'Val Loss': val_loss})
                pbar.update(1)
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered!")
                    break
            
        self.save_loss_curve(self.train_losses, self.val_losses, model_name='lstm')
        print(f">>> LSTM Training complete. Best model saved at: {self.checkpoint_path}")
        
    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        train_round_num = 0
        
        for i in range(0, len(self.x_lstm_train), self.lstm_batch_size):
            inputs = torch.tensor(self.x_lstm_train[i : i + self.lstm_batch_size], dtype=torch.float32).to(self.device)
            targets = torch.tensor(self.y_lstm_train[i : i + self.lstm_batch_size], dtype=torch.float32).to(self.device)
            
            # forward pass
            outputs = self.model(inputs, targets)
            
            loss = F.mse_loss(outputs, targets, reduction='mean')
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # log
            total_loss += loss.item()
            train_round_num += 1
        
        avg_loss = total_loss / train_round_num
            
        return avg_loss
            
    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        val_round_num = 0
        avg_loss = 0

        with torch.no_grad(): 
            for i in range(0, len(self.x_lstm_val), self.lstm_batch_size):
                inputs = torch.tensor(self.x_lstm_val[i : i + self.lstm_batch_size], dtype=torch.float32).to(self.device)
                targets = torch.tensor(self.y_lstm_val[i : i + self.lstm_batch_size], dtype=torch.float32).to(self.device)
                
                # forward pass
                outputs = self.model(inputs, targets)
                loss = F.mse_loss(outputs, targets, reduction='mean')
                
                # log
                total_loss += loss.item()
                val_round_num += 1
            
            avg_loss = total_loss / val_round_num

            if self.epoch % self.config['lstm_visualization_interval'] == 0:
                targets = targets.permute(0, 2, 1).unsqueeze(-1)
                outputs = outputs.permute(0, 2, 1).unsqueeze(-1)

                decode_targets_recon = self.vae_model.decode(targets)
                decode_outputs_recon = self.vae_model.decode(outputs)

                self.visualize_predictions(decode_targets_recon, decode_outputs_recon, self.epoch, avg_loss, mode='val')
                
        return avg_loss
    
    
    def create_lstm_sequence(self):   # create lstm input & prediction target sequences
        '''
        embeddings: generated from VAE's encoder (keep initial sequence order)
        look_back_num: use <look_back_num> number of embeddings for LSTM's input
        '''
        def _create_embedding_sequence(embeddings):
            x, y = [], []
            for i in range(len(embeddings) - self.look_back_num - 1):
                x.append(embeddings[i : i + self.look_back_num])    # LSTM's input (past embeddings)
                y.append(embeddings[i + self.look_back_num])        # LSTM's prediction target
            return np.array(x), np.array(y)
        
        self.x_lstm_train, self.y_lstm_train = _create_embedding_sequence(self.lstm_train_embeddings)
        self.x_lstm_val, self.y_lstm_val = _create_embedding_sequence(self.lstm_val_embeddings)
        
        
    def visualize_predictions(self, ground_truth, predictions, epoch, loss, mode='train'):
        """
        Visualize and save LSTM prediction quality.
        :param ground_truth: True embeddings.
        :param predictions: Predicted embeddings.
        :param epoch: Current training epoch.
        """
        
        ground_truth = ground_truth.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        
        colors = ["dodgerblue", "darkorange", "limegreen", "mediumvioletred", "gold", "darkcyan", "crimson"]
        
        # Initialize variables to store legend handles/labels
        handles, labels = [], []
        
        plot_group_num = 3
        plt.figure(figsize=(12, 10))
        for i in range(min(plot_group_num, ground_truth.shape[0])):
            plt.subplot(plot_group_num, 1, i + 1)
            for modal_idx in range(ground_truth.shape[1]):  # Loop through each modal/sensor type
                # only add label to legend once
                if i == 0: # first subplot to collect legend handles/labels
                    gt_line, = plt.plot(ground_truth[i, modal_idx].squeeze(-1), 
                                        label=f"Ground Truth Modal {modal_idx+1}", 
                                        linestyle='solid',
                                        color=colors[modal_idx])
                    pred_line, = plt.plot(predictions[i, modal_idx].squeeze(-1),
                                        label=f"Predicted Modal {modal_idx+1}",
                                        linestyle='dashed', 
                                        color=colors[modal_idx])
                    handles.extend([gt_line, pred_line])
                    labels.extend([f"Ground Truth Modal {modal_idx+1}", f"Predicted Modal {modal_idx+1}"])
                else:   # Other subplots: skip labels
                    plt.plot(ground_truth[i, modal_idx].squeeze(-1), 
                        linestyle='solid', 
                        color=colors[modal_idx])
                    plt.plot(predictions[i, modal_idx].squeeze(-1),
                        linestyle='dashed', 
                        color=colors[modal_idx])
                    
            plt.title(f"Sample {i+1}", 
                        y=-0.25,    # put title under the subplot
                        loc='center', 
                        fontsize=10)

        # add a general title for the whole figure
        plt.suptitle(f"LSTM Prediction Visualization ({mode.capitalize()}) - Epoch {epoch}", 
                    y=0.95,
                    fontweight='bold',
                    fontsize=14)
        
        # Add a single legend below all subplots
        plt.figlegend(handles, labels, 
                    loc='lower center', 
                    ncol=ground_truth.shape[1],  # Number of columns in legend
                    bbox_to_anchor=(0.5, 0.02)  # Adjust vertical position
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)  # Make space for the legend
        plt.savefig(os.path.join(self.config['result_dir'], f"lstm_predictions_mode_{mode}_epoch_{epoch}.png"))
        plt.close()
    
    def save_loss_curve(self, train_losses, val_losses, model_name):
        """
        Save the training and validation loss curves.
        :param train_losses: List of training losses.
        :param val_losses: List of validation losses.
        :param model_name: 'vae' or 'lstm'.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name.upper()} Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config['result_dir'], f"{model_name}_loss_curve.png"))
        plt.close()