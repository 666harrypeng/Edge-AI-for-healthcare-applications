import os
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import create_dataloader
from lstm_model import LSTMModel
from trainers import LSTMTrainer
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
    
    # dataloader has shuffled
    train_loader = create_dataloader(config, mode='train')
    val_loader = create_dataloader(config, mode='val')
    test_loader = create_dataloader(config, mode='test')
    
    # TRAIN LSTM
    lstm_checkpoint_path = os.path.join(config['checkpoint_dir_lstm'], 'lstm_best_model.pth')
    lstm_model = LSTMModel(config).to(device)
    
    pretrained_exist = glob.glob(lstm_checkpoint_path) # This will return a list, containing models' paths in string
    if pretrained_exist:
        print("Find Pre-trained model!")
        lstm_model.load_state_dict(torch.load(pretrained_exist[0], weights_only=True))
    else:
        print("No pre-trained model found.")   
    
    optimizer = optim.Adam(lstm_model.parameters(), lr=config['lstm_learning_rate'])
    lstm_trainer = LSTMTrainer(lstm_model, optimizer, train_loader, val_loader, config, readings_mean, readings_std)
    
    if config['TRAIN_LSTM'] or not os.path.exists(lstm_checkpoint_path):
        lstm_trainer.train()
        print("LSTM Training Completed.")
    else:
        lstm_model.load_state_dict(torch.load(lstm_checkpoint_path, weights_only=True))
        print(f"Loaded LSTM model from checkpoint: {lstm_checkpoint_path}.")
        
    print("Training Complete!")
    
    # Anomaly Detection & Inference
    if config['Anomaly_Detection']:
        # Load full dataset for visualization
        dataset_path = f"{config['data_dir']}/{config['dataset']}.npz"
        data = np.load(dataset_path)
    
        full_anomalies_idx = data['idx_anomaly_full']
        full_adj_anomaly_intervals = data['adj_anomaly_intervals']
        
        full_loader = create_dataloader(config, mode='full_data_rolling')   # load rolling windows for raw full dataset (including anomalies) (no shuffle with batch_size as 1)

        # calculate the errors & find the anomalies & save in the LSTM Trainer object
        prediction_errors, predicted_anomalies_window_idx = lstm_trainer.find_anomalies(full_loader)
        
        used_data_length = len(full_loader) * config['l_win']
        full_normalized_data_used = data['readings_normalized'][ : used_data_length]

        # Convert predicted window indices to intervals
        predicted_intervals = [(win_idx * config['l_win'], (win_idx + 1) * config['l_win']) for win_idx in predicted_anomalies_window_idx]

        # Calculate detection accuracy based on window overlaps
        detected_anomalies = 0
        for true_start, true_end in full_adj_anomaly_intervals:
            # For each ground truth anomaly window, check if any predicted window overlaps
            for pred_start, pred_end in predicted_intervals:
                if not (pred_end < true_start or pred_start > true_end):
                    detected_anomalies += 1
                    break  # Once we find an overlapping prediction, move to next ground truth
        
        # Calculate recall (detection accuracy)
        recall = detected_anomalies / len(full_adj_anomaly_intervals)

        print(f'Window-based Detection Accuracy:')
        print(f'Detected {detected_anomalies} out of {len(full_adj_anomaly_intervals)} anomalies')
        print(f'Detection Rate (Recall): {recall:.4f}')
        
        lstm_trainer.visualize_anomalies(full_data_used=full_normalized_data_used,
                                       adj_true_anomaly_intervals=full_adj_anomaly_intervals, 
                                       predicted_anomalies_window_idx=predicted_anomalies_window_idx,
                                       recall=recall)
        
        saved_path = os.path.join(config['result_dir'], "anomaly_detection_comparison.png")
        print(f"Anomaly detection visualization on full dataset saved @ {saved_path}.")


    print(" >>> Finish ALL <<<")
    
if __name__ == "__main__":
    main()


