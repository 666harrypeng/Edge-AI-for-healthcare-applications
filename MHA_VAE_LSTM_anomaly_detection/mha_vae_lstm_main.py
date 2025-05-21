import os
import glob
import numpy as np
import torch
import torch.optim as optim
from data_loader import create_dataloader
from models import VAEmodel, LSTMmodel
from trainers import VAETrainer, LSTMTrainer
from utils import process_config, create_dirs, save_config, get_args, evaluate_detection


def main():
    # Capture the config path from the run arguments
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
    config['device'] = device
    print(f"Using device: {device}")
        
    # Create DataLoaders
    dataset_path = f"{config['data_dir']}/{config['dataset']}.npz"
    data = np.load(dataset_path)
    readings_mean = data['train_m']
    readings_std = data['train_std']
    
    
    vae_train_loader = create_dataloader(config, mode='train')
    vae_val_loader = create_dataloader(config, mode='val')
    vae_test_loader = create_dataloader(config, mode='test')
    
    ########### Train VAE ###########
    vae_checkpoint_path = os.path.join(config['checkpoint_dir'], 'vae_best_model.pth')
    vae_model = VAEmodel(config).to(device)
    # check if pre-trained VAE model exists
    vae_pretrained_exist = glob.glob(vae_checkpoint_path) # This will return a list, containing models' paths in string
    if vae_pretrained_exist:
        print("Find Pre-trained VAE model!")
        vae_model.load_state_dict(torch.load(vae_pretrained_exist[0], weights_only=True))
    else:
        print("No pre-trained VAE model found.")   
    
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=config['vae_learning_rate'], weight_decay=config['vae_weight_decay'])
    vae_trainer = VAETrainer(vae_model, vae_train_loader, vae_val_loader, vae_optimizer, config, device, readings_mean, readings_std)
    if config['TRAIN_VAE'] or not os.path.exists(vae_checkpoint_path):
        vae_trainer.train()
        print("VAE Training complete!")
    else: # no need to train VAE & already loaded the pre-trained VAE model
        print(f"No need to train VAE & already loaded the pre-trained VAE model from checkpoint: {vae_checkpoint_path}")
    
    ########### Train LSTM ###########
    lstm_checkpoint_path = os.path.join(config['checkpoint_dir_lstm'], 'lstm_best_model.pth')
    lstm_model = LSTMmodel(config).to(device)
    # check if pre-trained LSTM model exists
    lstm_pretrained_exist = glob.glob(lstm_checkpoint_path) # This will return a list, containing models' paths in string
    if lstm_pretrained_exist:
        print("Find Pre-trained LSTM model!")
        lstm_model.load_state_dict(torch.load(lstm_pretrained_exist[0], weights_only=True))
    else:
        print("No pre-trained LSTM model found.")   

    
    if config['TRAIN_LSTM'] or not os.path.exists(lstm_checkpoint_path):
        # Reload the trained VAE for embedding generation
        vae_model.load_state_dict(torch.load(vae_checkpoint_path, weights_only=True))
        vae_model.eval()

        # Generate embeddings using the trained VAE  
        lstm_embeddings_train_loader = create_dataloader(config, mode='lstm_train')
        lstm_embeddings_val_loader = create_dataloader(config, mode='lstm_val')
        lstm_train_embeddings, lstm_val_embeddings = vae_trainer.generate_lstm_embeddings(vae_model, lstm_embeddings_train_loader, lstm_embeddings_val_loader)
        
        # Train a new LSTM model if requested or if no checkpoint exists
        lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=config['lstm_learning_rate'])
        lstm_trainer = LSTMTrainer(lstm_model, lstm_train_embeddings, lstm_val_embeddings, lstm_optimizer, config, device, readings_mean, readings_std, vae_model)

        # create lstm input & target sequences
        lstm_trainer.create_lstm_sequence()
        
        lstm_trainer.train()
        print("LSTM Training complete!")
    else: # no need to train LSTM & already loaded the pre-trained LSTM model
        print(f"No need to train LSTM & already loaded the pre-trained LSTM model from checkpoint: {lstm_checkpoint_path}")
        
    print("VAE & LSTM Training complete!")
    
    ########### Anomaly Detection & Inference ###########
    if config['Anomaly_Detection']:
        # Load full dataset for visualization
        dataset_path = f"{config['data_dir']}/{config['dataset']}.npz"
        data = np.load(dataset_path)

        full_anomalies_idx = data['idx_anomaly_full']
        full_adj_anomaly_intervals = data['adj_anomaly_intervals']
        
        full_loader = create_dataloader(config, mode='full_data_rolling')   # load rolling windows for raw full dataset (including anomalies) (no shuffle with batch_size as 1)
        
        anomaly_scores, predicted_anomalies_window_idx = vae_trainer.find_anomalies(full_loader, lstm_model)
        
        used_data_length = len(full_loader) * config['l_win']
        full_normalized_data_used = data['readings_normalized'][ : used_data_length]
    
        # Convert predicted window indices to intervals
        predicted_intervals = [(win_idx * config['l_win'], (win_idx + 1) * config['l_win']) for win_idx in predicted_anomalies_window_idx]
        
        # Calculate confusion matrix metrics
        # Create binary arrays for true and predicted anomalies
        total_windows = len(full_loader)
        true_anomaly_windows = np.zeros(total_windows, dtype=bool)
        pred_anomaly_windows = np.zeros(total_windows, dtype=bool)
        
        # Mark true anomaly windows
        for true_start, true_end in full_adj_anomaly_intervals:
            start_window = true_start // config['l_win']
            end_window = (true_end - 1) // config['l_win'] + 1
            true_anomaly_windows[start_window:end_window] = True
        
        # Mark predicted anomaly windows
        # Filter out any indices that are out of bounds
        valid_indices = predicted_anomalies_window_idx[predicted_anomalies_window_idx < total_windows]
        if len(valid_indices) < len(predicted_anomalies_window_idx):
            print(f"Warning: {len(predicted_anomalies_window_idx) - len(valid_indices)} predicted anomaly indices were out of bounds and have been filtered out.")
        pred_anomaly_windows[valid_indices] = True
        
        # Calculate confusion matrix elements
        true_positives = np.sum(np.logical_and(true_anomaly_windows, pred_anomaly_windows))
        false_positives = np.sum(np.logical_and(np.logical_not(true_anomaly_windows), pred_anomaly_windows))
        false_negatives = np.sum(np.logical_and(true_anomaly_windows, np.logical_not(pred_anomaly_windows)))
        true_negatives = np.sum(np.logical_and(np.logical_not(true_anomaly_windows), np.logical_not(pred_anomaly_windows)))
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F-beta score (default beta=2 to emphasize recall)
        beta = config.get('f_beta', 2)
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
        
        # Calculate detection accuracy based on window overlaps (original method)
        detected_anomalies = 0
        for true_start, true_end in full_adj_anomaly_intervals:
            for pred_start, pred_end in predicted_intervals:
                if not (pred_end < true_start or pred_start > true_end):
                    detected_anomalies += 1
                    break  # Once we find an overlapping prediction, move to next ground truth
        
        # Calculate recall (detection rate)
        detect_rate = detected_anomalies / len(full_adj_anomaly_intervals)

        print(f'Window-based Detection Metrics:')
        print(f'Detected {detected_anomalies} out of {len(full_adj_anomaly_intervals)} anomalies')
        print(f'Detection Rate (Recall): {detect_rate:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'F-{beta} Score: {f_beta:.4f}')
            
        # Save detailed metrics to a separate file
        metrics_path = os.path.join(config['result_dir'], "detection_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"threshold,{config['threshold_percent']}\n")
            f.write(f"recall,{detect_rate:.4f}\n")
            f.write(f"precision,{precision:.4f}\n")
            f.write(f"f_beta,{f_beta:.4f}\n")
            f.write(f"true_positives,{true_positives}\n")
            f.write(f"false_positives,{false_positives}\n")
            f.write(f"false_negatives,{false_negatives}\n")
            f.write(f"true_negatives,{true_negatives}\n")
        
        # visualize anomalies
        vae_trainer.visualize_anomalies(full_data_used=full_normalized_data_used,
                                       adj_true_anomaly_intervals=full_adj_anomaly_intervals, 
                                       predicted_anomalies_window_idx=predicted_anomalies_window_idx,
                                       detect_rate=detect_rate,
                                       precision=precision,
                                       f_beta=f_beta)   
        
    print(">>> finish all! <<<")


if __name__ == '__main__':
    main()