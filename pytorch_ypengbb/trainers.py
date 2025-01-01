import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
        # self.writer = SummaryWriter(log_dir=config['summary_dir'])
        self.epoch = 0
        self.patience = config.get('patience', 20)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_path = os.path.join(config['checkpoint_dir'], 'vae_best_model.pth')

    def train(self):
        self.train_losses, self.val_losses = [], []
        with tqdm(total=self.config['num_epochs_vae'], desc="VAE Training") as pbar:
            for epoch in range(self.config['num_epochs_vae']):
                self.epoch = epoch
                train_loss = self._train_epoch()
                val_loss = self._validate_epoch()
                # self.writer.add_scalar("Loss/Train", train_loss, epoch)
                # self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                
                # Early stopping and checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.patience_counter = 0
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_vae']}: New best model saved with validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_vae']}: Validation loss did not improve. Patience: {self.patience_counter}/{self.patience}")
                    
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
            batch = batch.unsqueeze(1)  
            self.optimizer.zero_grad()
            reconstructed, mean, logvar = self.model(batch) 
            loss = self._vae_loss(batch, reconstructed, mean, logvar)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        if self.epoch % self.config["visualization_interval"] == 0:
            batch = batch.squeeze()  
            reconstructed = reconstructed.squeeze()  
            self.visualize_reconstruction(batch, reconstructed, self.epoch, mode='train')
            
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                batch = batch.unsqueeze(1) 
                reconstructed, mean, logvar = self.model(batch) 
                loss = self._vae_loss(batch, reconstructed, mean, logvar)
                total_loss += loss.item()
            
            if self.epoch % self.config["visualization_interval"] == 0:
                batch = batch.squeeze()  
                reconstructed = reconstructed.squeeze()  
                self.visualize_reconstruction(batch, reconstructed, self.epoch, mode='val')
                
        return total_loss / len(self.val_loader)

    @staticmethod
    def _vae_loss(original, reconstructed, mean, logvar):
        reconstruction_loss = F.mse_loss(reconstructed, original, reduction='mean') 
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence

    def generate_lstm_embeddings(self, vae_model, train_loader, val_loader):
        """
        Generate embeddings from the VAE for use in LSTM training.
        """
        train_embeddings, val_embeddings = [], []
        vae_model.eval()

        def _generate_embeddings(data_loader):
            embeddings = []
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(self.device)
                    batch = batch.unsqueeze(1)  
                    mean, logvar = vae_model.encode(batch)
                    mean = mean.squeeze()  
                    mean = mean.permute(0, 2, 1)
                    embeddings.append(mean.cpu())
            return embeddings

        train_embeddings = _generate_embeddings(train_loader)
        val_embeddings = _generate_embeddings(val_loader)
        
        return train_embeddings, val_embeddings

    def compute_anomaly_scores(self, test_loader, lstm_model):
        """
        Compute combined anomaly scores for the test set.
        :param test_loader: DataLoader for the test set.
        :param lstm_model: Trained LSTM model for prediction errors.
        :return: List of combined anomaly scores and their indices.
        """
        self.model.eval()
        lstm_model.eval()
        anomaly_scores = []
        window_indices = np.arange(0, self.config['l_win'])  
        reconstruction_errors = []
        prediction_errors = []
        
        with torch.no_grad():
            ##### Generate VAE reconstruction errors #####
            for i, batch in enumerate(test_loader):
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                
                batch = batch.unsqueeze(1)  
                assert batch.shape[1:] == (self.config['n_channel'], self.config['l_win'], 1), "Final Visual VAE input shape is not correct!"
                
                # VAE Reconstruction Error
                reconstructed, _, _ = self.model(batch)
                
                reconstruction_error = ((batch - reconstructed) ** 2).squeeze().cpu().numpy()
                reconstruction_errors.extend(reconstruction_error)
                
                # for final visualization of anomaly detection, we choose to use batch_size=1, each interval length is rolling window length
                window_indices = np.vstack((window_indices, np.arange(i * self.config['l_win'], (i + 1) * self.config['l_win'])))
            window_indices = window_indices[1:]
            
            print("\nfinish reconstruction\n")
                
            ########### Generate lstm embeddings ################
            embeddings = []
            for batch in test_loader:
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                batch = batch.unsqueeze(1)
                
                assert batch.shape[1:] == (self.config['n_channel'], self.config['l_win'], 1), "Final Visual Embeddings input shape is not correct!"

                mean, logvar = self.model.encode(batch)
                embed_mean = mean.squeeze(-1) 
                embed_mean = embed_mean.permute(0, 2, 1)  
                embeddings.append(embed_mean.cpu())
                
            print("\nfinish lstm embeddings\n")    
            
            prediction_idx = self.config['l_seq']
            for i, batch in enumerate(embeddings):
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                
                # LSTM Prediction Error
                inputs, targets = batch[:, :-prediction_idx, :], batch[:, -prediction_idx:, :]
                
                # predict
                pred_embeddings = lstm_model(inputs)
                pred_outputs = torch.cat((inputs, pred_embeddings), dim=1) 
                pred_targets = batch   
                pred_outputs = pred_outputs.permute(0, 2, 1).unsqueeze(-1)
                pred_targets = pred_targets.permute(0, 2, 1).unsqueeze(-1)
                
                decode_outputs_recon = self.model.decode(pred_outputs).squeeze().cpu().detach().numpy()
                decode_targets_recon = self.model.decode(pred_targets).squeeze().cpu().detach().numpy()                
                decode_outputs_recon = decode_outputs_recon * self.data_std + self.data_mean
                decode_targets_recon = decode_targets_recon * self.data_std + self.data_mean

                prediction_error = (decode_targets_recon - decode_outputs_recon) ** 2    

                prediction_errors.extend(prediction_error)


            print("\nfinish prediction\n")
            
            # Combined Score
            assert len(reconstruction_errors) == len(prediction_errors), "Error list lengths do not match!"
            anomaly_scores = np.array(reconstruction_errors) + 0.5 * np.array(prediction_errors)

        return window_indices, anomaly_scores


    def visualize_anomalies(self, full_data, true_anomalies_idx, predicted_anomalies_idx, precision, recall, f1):
        plt.figure(figsize=(18, 6))
        plt.plot(full_data, label="Full Data", color="blue")

        # Highlight true anomalies
        for i, anomaly in enumerate(true_anomalies_idx):
            plt.axvline(x=anomaly, color="red", linestyle="-", label="True Anomalies" if i == 0 else "")
            
        # Highlight predicted anomalies
        for i, anomaly in enumerate(predicted_anomalies_idx):
            plt.axvline(x=anomaly, color="green", linestyle=":", label="Predicted Anomalies" if i==0 else "")

        detect_dataset = self.config['dataset']
        plt.title(f"{detect_dataset} - Anomaly Detection Visualization\nPrecision:{precision:.4f}, Recall:{recall:.4f}, F1:{f1:.4f}")
        plt.xlabel("Timestamps")
        plt.ylabel("Normalized Readings")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['result_dir'], "anomaly_detection_comparison.png"))
        plt.close()    
    

    def visualize_reconstruction(self, raw_data, reconstructed_data, epoch, mode='train'):
        raw_data = raw_data.cpu().numpy()
        reconstructed_data = reconstructed_data.cpu().detach().numpy()
        # De-normalize the data
        raw_data = raw_data * self.data_std + self.data_mean
        reconstructed_data = reconstructed_data * self.data_std + self.data_mean
        
        plt.figure(figsize=(12, 6))
        for i in range(min(5, raw_data.shape[0])):  # Plot 5 samples
            plt.subplot(5, 1, i + 1)
            plt.plot(raw_data[i], label="Raw Data", linestyle='solid', color='red')
            plt.plot(reconstructed_data[i], label="Reconstructed Data", linestyle='dashed', color='blue')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['result_dir'], f"vae_reconstruction_{mode}_epoch_{epoch}.png"))
        plt.close()
    
    def save_loss_curve(self, train_losses, val_losses, model_name):
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
    def __init__(self, model, train_loader, val_loader, optimizer, config, device, mean, std, vae_model):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.data_mean = mean
        self.data_std = std
        self.vae_model = vae_model
        # self.writer = SummaryWriter(log_dir=config['summary_dir'])
        self.epoch = 0
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
                # self.writer.add_scalar("Loss/Train_LSTM", train_loss, epoch)
                # self.writer.add_scalar("Loss/Validation_LSTM", val_loss, epoch)
                
                # Early stopping and checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    self.patience_counter = 0
                    print(f"Epoch {epoch + 1}/{self.config['num_epochs_lstm']}: New best model saved with validation loss: {val_loss:.4f}")
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
        prediction_idx = self.config['l_seq']
        for batch in self.train_loader:
            batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
            self.optimizer.zero_grad()
            inputs, targets = batch[:, :-prediction_idx, :], batch[:, -prediction_idx:, :]  
            outputs = self.model(inputs)
            
            assert outputs.shape == targets.shape, "LSTM output shape does not match target shape!"
            
            loss = F.mse_loss(outputs, targets, reduction='mean')
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        if self.epoch % self.config['visualization_interval'] == 0:
            pred_outputs = torch.cat((inputs, outputs), dim=1)  
            pred_targets = batch
            pred_outputs = pred_outputs.permute(0, 2, 1).unsqueeze(-1)
            pred_targets = pred_targets.permute(0, 2, 1).unsqueeze(-1)
            
            decode_outputs_recon = self.vae_model.decode(pred_outputs)
            decode_targets_recon = self.vae_model.decode(pred_targets)
            
            self.visualize_predictions(decode_targets_recon, decode_outputs_recon, self.epoch, mode='train')
            
        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        prediction_idx = self.config['l_seq']
        with torch.no_grad(): 
            for batch in self.val_loader:
                batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
                inputs, targets = batch[:, :-prediction_idx, :], batch[:, -prediction_idx:, :]
                outputs = self.model(inputs)
                loss = F.mse_loss(outputs, targets, reduction='mean')
                total_loss += loss.item()
                
            if self.epoch % self.config['visualization_interval'] == 0:
                pred_outputs = torch.cat((inputs, outputs), dim=1)  
                pred_targets = batch    
                    
                pred_outputs = pred_outputs.permute(0, 2, 1).unsqueeze(-1)
                pred_targets = pred_targets.permute(0, 2, 1).unsqueeze(-1)
                
                decode_outputs_recon = self.vae_model.decode(pred_outputs)
                decode_targets_recon = self.vae_model.decode(pred_targets)
                
                self.visualize_predictions(decode_targets_recon, decode_outputs_recon, self.epoch, mode='val')
                
        return total_loss / len(self.val_loader)
    
    
    def visualize_predictions(self, ground_truth, predictions, epoch, mode='train'):
        ground_truth = ground_truth.squeeze().cpu().detach().numpy()
        predictions = predictions.squeeze().cpu().detach().numpy()
        
        # De-normalize the data
        ground_truth = ground_truth * self.data_std + self.data_mean
        predictions = predictions * self.data_std + self.data_mean
        
        plt.figure(figsize=(12, 6))
        for i in range(min(5, ground_truth.shape[0])):  # Plot 5 samples
            plt.subplot(5, 1, i + 1)
            plt.plot(ground_truth[i, :], label="Ground Truth", linestyle='solid', color='red')
            plt.plot(predictions[i, :], label="Predicted", linestyle='dashed', color='blue')
            plt.legend()
        pred_start_idx = self.config['l_win'] - self.config['l_seq']
        whole_seq_len = self.config['l_win']
        plt.suptitle(f'LSTM Predictions\n(whole sequence length:{whole_seq_len} start idx @ {pred_start_idx})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['result_dir'], f"lstm_predictions_mode_{mode}_epoch_{epoch}.png"))
        plt.close()
    
    def save_loss_curve(self, train_losses, val_losses, model_name):
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