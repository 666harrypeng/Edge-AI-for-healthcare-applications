import os
import argparse
import json
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt


def load_collected_data(csv_folder_path, dataset, config):
    # Load dataset-specific configuration
    with open('dataset_config.json', 'r') as f:
        dataset_config = json.load(f)
    
    if dataset not in dataset_config:
        raise ValueError(f"Dataset {dataset} not found in dataset_config.json")
    
    dataset_specific_config = dataset_config[dataset]
    anomaly_intervals = dataset_specific_config['anomaly_intervals']
    recording_start_line = dataset_specific_config['recording_start_line']
    
    data_file = os.path.join(csv_folder_path, f'sensor_data_{dataset}_with_anomalies.csv')
    adj_anomaly_intervals = []
    anomalies = []
    
    with open(data_file, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV) # skip header row
        rows = list(readCSV)
        
        # Start from line recording_start_line where valid readings begin
        valid_rows = rows[recording_start_line:]
        
        # Extract timestamps for each anomaly interval
        for start_line, end_line in anomaly_intervals:
            # Adjust indices to account for the recording_start_line offset
            adj_start = start_line - recording_start_line
            adj_end = end_line - recording_start_line   
            adj_anomaly_intervals.append((adj_start, adj_end))
            
            # Extract timestamps between start and end lines
            for i in range(adj_start, adj_end + 1):
                if i < len(valid_rows):
                    anomalies.append(i)

        t = []
        temp_readings = []
        hum_readings = []
        audio_readings = []
        for i, row in enumerate(valid_rows):
            t.append(i)
            audio_readings.append(float(row[1]))
            temp_readings.append(float(row[2]))
            hum_readings.append(float(row[3]))
        t = np.asarray(t)
        temp_readings = np.asarray(temp_readings)
        hum_readings = np.asarray(hum_readings)
        audio_readings = np.asarray(audio_readings)
        
        print(f">>> Original {dataset} csv file contains {t.shape} timestamps.")
        print(f">>> Processed time series contain {temp_readings.shape} temperature readings, {hum_readings.shape} humidity readings, and {audio_readings.shape} audio readings.")
        print(f">>> Include {len(anomalies)} anomalies")

        return t, temp_readings, hum_readings, audio_readings, anomalies, adj_anomaly_intervals

def create_rolling_windows(data, window_size):
    ## non-overlapped windows
    window_num = len(data) // window_size
    return np.array([data[i * window_size : (i+1) * window_size] for i in range(window_num)])

def preprocess_and_save(csv_folder_path, dataset, data_dir, config):
    t, temp_readings, hum_readings, audio_readings, anomalies, adj_anomaly_intervals = load_collected_data(csv_folder_path, dataset, config)
    print(f">>> Dataset {dataset} contains -> temp shape: {temp_readings.shape}, hum shape: {hum_readings.shape}, and audio shape: {audio_readings.shape}")
    print(">>> Successfully loaded data >>> Continue Data Preprocessing")
    
    # normalize by training mean and std
    temp_readings_mean, temp_readings_std = np.mean(temp_readings), np.std(temp_readings)
    hum_readings_mean, hum_readings_std = np.mean(hum_readings), np.std(hum_readings)
    audio_readings_mean, audio_readings_std = np.mean(audio_readings), np.std(audio_readings)
    temp_readings_normalized = (temp_readings - temp_readings_mean) / temp_readings_std
    hum_readings_normalized = (hum_readings - hum_readings_mean) / hum_readings_std
    audio_readings_normalized = (audio_readings - audio_readings_mean) / audio_readings_std
    
    # concatenate temperature and humidity readings into a single time series
    assert len(temp_readings) == len(hum_readings) == len(audio_readings), "Temperature, humidity, and audio readings must have the same length"
    raw_readings = np.column_stack((temp_readings, hum_readings, audio_readings))
    readings_normalized = np.column_stack((temp_readings_normalized, hum_readings_normalized, audio_readings_normalized))

    # split into training and test sets
    split_ratio = config.get('split_ratio', 0.9)
    split_idx = int(len(temp_readings) * split_ratio)
    train_readings = readings_normalized[:split_idx]
    train_t = t[:split_idx]
    test_readings = readings_normalized[split_idx:]
    test_t = t[split_idx:]  
    
    # adjust anomaly indices for the test set
    test_idx_anomaly = [idx - split_idx for idx in anomalies if idx >= split_idx]
    
    # Filter out anomalies from training and validation
    train_idx_anomaly = [idx for idx in anomalies if idx < split_idx]
    filtered_training_readings = np.delete(train_readings, train_idx_anomaly, axis=0)
    filtered_training_t = np.delete(train_t, train_idx_anomaly, axis=0)
    assert len(filtered_training_readings) == len(filtered_training_t)  # the lengths should match
    
    # split into training and validation sets
    val_ratio = config.get('val_ratio', 0.1)
    val_idx = int(len(filtered_training_readings) * (1-val_ratio))
    train_readings = filtered_training_readings[:val_idx]
    train_t = filtered_training_t[:val_idx]
    val_readings = filtered_training_readings[val_idx:]
    val_t = filtered_training_t[val_idx:]
    assert len(train_readings) == len(train_t)  # the lengths should match
    assert len(val_readings) == len(val_t)  # the lengths should match
    
    # create rolling windows for train, val, and test sets
    window_size = config.get('l_win', 60)
    train_rolling = create_rolling_windows(train_readings, window_size)
    val_rolling = create_rolling_windows(val_readings, window_size)
    test_rolling = create_rolling_windows(test_readings, window_size)
    
    # create rolling windows for full data  (for final anomaly detection visualization)
    full_rolling = create_rolling_windows(readings_normalized, window_size)
    window_num = len(readings_normalized) // window_size
    full_rolling_anomaly_idx = [idx for idx in anomalies if idx <= (window_num * window_size)]    # dispose anomalies in the remaining data points that cannot be constructed into one window
    adj_anomaly_intervals = [(adj_start, adj_end) for adj_start, adj_end in adj_anomaly_intervals if adj_end <= (window_num * window_size)]
   
    # save the preprocessed dataset
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f'{dataset}.npz')
    np.savez(save_path, 
             t=t, 
             readings=raw_readings,  # Original readings (unmodified)
             readings_normalized=readings_normalized, # Normalized readings
             idx_anomaly=anomalies, 
             training=train_rolling, 
             val=val_rolling,
             test=test_rolling, 
             full_data_rolling=full_rolling,
             t_train=train_t, 
             t_val=val_t, 
             t_test=test_t, 
             idx_anomaly_train=train_idx_anomaly,
             idx_anomaly_test=test_idx_anomaly,
             idx_anomaly_full=full_rolling_anomaly_idx,
             adj_anomaly_intervals=adj_anomaly_intervals,
             train_m=(temp_readings_mean, hum_readings_mean, audio_readings_mean), 
             train_std=(temp_readings_std, hum_readings_std, audio_readings_std))
    
    # Print anomaly statistics
    print("\nAnomaly Statistics after processed:")
    print(f"Number of anomaly points in training set: {len(train_idx_anomaly)}")
    print(f"Number of anomaly points in validation set: 0 (only to validate prediction training, no anomalies in validation set)")  # Since we filtered anomalies from training+validation
    print(f"Number of anomaly points in test set: {len(test_idx_anomaly)}")
    print(f"Total number of anomaly points: {len(anomalies)}")
    
    print(f"\n>>> Preprocessed {dataset} dataset saved at {save_path} <<<")
    
###### main function for data preprocessing ######
def main():
    parser = argparse.ArgumentParser(description="Dataset Preprocessing and Save")
    parser.add_argument('-c', '--config', default='./mha_vae_lstm_config.json', help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    csv_folder_path = config['csv_folder_path']
    data_dir = config['data_dir']
    
    dataset = config['dataset']
    preprocess_and_save(csv_folder_path, dataset, data_dir, config)

    print("\nCollected data preprocessing and saving complete!")
    

if __name__ == '__main__':
    main()