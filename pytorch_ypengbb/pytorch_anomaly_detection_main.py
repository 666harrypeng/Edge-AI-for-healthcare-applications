import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data_loader import create_dataloader
from models import VAEmodel, LSTMmodel
from trainers import VAETrainer, LSTMTrainer
from utils import process_config, create_dirs, save_config, get_args, evaluate_detection


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as e:
        print(f"Error in processing config: {e}")
        exit(0)

    # Create necessary directories
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    save_config(config)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # Create DataLoaders
    dataset_path = f"{config['data_dir']}/{config['dataset']}.npz"
    data = np.load(dataset_path)
    readings_mean = data['train_m']
    readings_std = data['train_std']
    
    
    train_loader = create_dataloader(config, mode='train')
    val_loader = create_dataloader(config, mode='val')
    test_loader = create_dataloader(config, mode='test')

    # Train VAE
    vae_checkpoint_path = os.path.join(config['checkpoint_dir'], 'vae_best_model.pth')
    vae_model = VAEmodel(config).to(device)
    if config['TRAIN_VAE'] or not os.path.exists(vae_checkpoint_path):
        optimizer = optim.Adam(vae_model.parameters(), lr=config['vae_learning_rate'])
        vae_trainer = VAETrainer(vae_model, train_loader, val_loader, optimizer, config, device, readings_mean, readings_std)
        
        if config['num_epochs_vae'] > 0:
            vae_trainer.train()
        print("VAE Training complete!")
    else:
        vae_model.load_state_dict(torch.load(vae_checkpoint_path, weights_only=True))
        optimizer = optim.Adam(vae_model.parameters(), lr=config['vae_learning_rate'])
        vae_trainer = VAETrainer(vae_model, train_loader, val_loader, optimizer, config, device, readings_mean, readings_std)
        print(f"Loaded VAE model from checkpoint: {vae_checkpoint_path}")
    
    # Train LSTM
    lstm_checkpoint_path = os.path.join(config['checkpoint_dir_lstm'], 'lstm_best_model.pth')
    lstm_model = LSTMmodel(config).to(device)

    if config['TRAIN_LSTM'] or not os.path.exists(lstm_checkpoint_path):
        vae_model.load_state_dict(torch.load(vae_checkpoint_path, weights_only=True))
        vae_model.eval()

        lstm_train_loader, lstm_val_loader = vae_trainer.generate_lstm_embeddings(vae_model, train_loader, val_loader)
        
        optimizer = optim.Adam(lstm_model.parameters(), lr=config['lstm_learning_rate'])
        lstm_trainer = LSTMTrainer(lstm_model, lstm_train_loader, lstm_val_loader, optimizer, config, device, readings_mean, readings_std, vae_model)

        if config['num_epochs_lstm'] > 0:
            lstm_trainer.train()
        print("LSTM Training complete!")
    else:
        lstm_model.load_state_dict(torch.load(lstm_checkpoint_path, weights_only=True))
        print(f"Loaded LSTM model from checkpoint: {lstm_checkpoint_path}")
        
    print("Training complete!")
    
    # Compute and visualize anomalies on the full dataset
    if config['Visualize_anomaly']:
        dataset_path = f"{config['data_dir']}/{config['dataset']}.npz"
        data = np.load(dataset_path)
        full_anomalies_idx = data['idx_anomaly_full']
        full_loader = create_dataloader(config, mode='anomaly_detection')
        indices, scores = vae_trainer.compute_anomaly_scores(full_loader, lstm_model)
        
        threshold_percentage = config['threshold_percent']
        threshold = np.percentile(scores, threshold_percentage)
        print(f"finish computing anomaly scores & threshold. ----> threshold percent: {threshold_percentage}; threshold: {threshold}")
        
        
        full_data_used = data['readings_normalized'][ : len(scores)]
        
        predicted_anomalies_idx = np.array([i for i, score in enumerate(scores) if score > threshold])
        
        # Evaluate precision, recall, and F1 score
        precision, recall, f1 = evaluate_detection(true_anomalies=full_anomalies_idx,
                                                   predicted_anomalies=predicted_anomalies_idx,
                                                   dataset_length=len(full_data_used))
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
        # Visualize anomalies
        vae_trainer.visualize_anomalies(full_data_used, full_anomalies_idx, predicted_anomalies_idx, precision, recall, f1)
        print("Anomaly detection visualization on full dataset saved.")
        
    print(">>> finish all! <<<")


if __name__ == '__main__':
    main()