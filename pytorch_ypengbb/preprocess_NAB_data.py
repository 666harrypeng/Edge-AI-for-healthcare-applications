import os
import argparse
import json
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

def load_data(csv_folder_path, dataset):
    anomalies = []
    
    if dataset == 'ambient_temp':
        data_file = os.path.join(csv_folder_path, 'ambient_temperature_system_failure.csv')
        anomalies = ['2013-12-22 20:00:00', '2014-04-13 09:00:00']
    elif dataset == 'cpu_utilization':
        data_file = os.path.join(csv_folder_path, 'cpu_utilization_asg_misconfiguration.csv')
        anomalies = ['2014-07-12 02:04:00', '2014-07-14 21:44:00']
    elif dataset == 'ec2_request':
        data_file = os.path.join(csv_folder_path, 'ec2_request_latency_system_failure.csv')
        anomalies = ['2014-03-14 09:06:00', '2014-03-18 22:41:00', '2014-03-21 03:01:00']
    elif dataset == 'machine_temp':
        data_file = os.path.join(csv_folder_path, 'machine_temperature_system_failure.csv')
        anomalies = ['2013-12-11 06:00:00', '2013-12-16 17:25:00', '2014-01-28 13:55:00', '2014-02-08 14:30:00']
    elif dataset == 'rogue_agent_key_hold':
        data_file = os.path.join(csv_folder_path, 'rogue_agent_key_hold.csv')
        anomalies = ['2014-07-15 08:30:00', '2014-07-17 09:50:00']
    elif dataset == 'rogue_agent_key_updown':
        data_file = os.path.join(csv_folder_path, 'rogue_agent_key_updown.csv')
        anomalies = ['2014-07-15 04:00:00', '2014-07-17 08:50:00']
    elif dataset == 'nyc_taxi':
        data_file = os.path.join(csv_folder_path, 'nyc_taxi.csv')
        anomalies = ['2014-11-01 19:00:00', '2014-11-27 15:30:00', '2014-12-25 15:00:00', '2015-01-01 01:00:00', 
                     '2015-01-27 00:00:00']
        
    idx_anomaly = []
    readings = []
    t = []
    i = 0
    
    with open(data_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        print("\n--> Anomalies occur at:")
        for row in readCSV:
            if i > 0:  # Skip header row
                t.append(i)
                readings.append(float(row[1]))
                for anomaly in anomalies:
                    if row[0] == anomaly:
                        idx_anomaly.append(i)
                        print("  timestamp: {}".format(row[0]))
            i += 1
    t = np.asarray(t)
    readings = np.asarray(readings)
    print(f"\nOriginal {dataset} csv file contains {t.shape} timestamps.")
    print(f"Processed time series contain {readings.shape} readings.")
    print(f"Anomaly indices are {idx_anomaly}")
    
    return t, readings, idx_anomaly
            
def create_rolling_windows(data, window_size):
    # ## Overlapped Windows
    # return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])
    
    ## Non-overlapped Windows
    window_num = len(data) // window_size
    return np.array([data[i * window_size : (i+1) * window_size] for i in range(window_num)])

def preprocess_and_save(csv_folder_path, dataset, save_dir_path, val_ratio=0.1, split_ratio=0.9, save_file=True, window_size=48, json_file='./pytorch_NAB_config.json'):
    # Load raw data
    t, readings, idx_anomaly = load_data(csv_folder_path, dataset)
    print(">>> Successfully loaded data >>> Continue Data Preprocessing")
    
    # normalize by training mean and std
    readings_mean, readings_std = np.mean(readings), np.std(readings)
    readings_normalized = (readings - readings_mean) / readings_std
    
    # split into training and test sets
    with open(json_file, 'r') as config_file:
        config = json.load(config_file)
    split_ratio = config.get('split_ratio', 0.9)
    split_idx = int(len(readings) * split_ratio)
    train_readings = readings_normalized[:split_idx]
    train_t = t[:split_idx]
    test_readings = readings_normalized[split_idx:]
    test_t = t[split_idx:]
    
    # adjust anomaly indices for the test set
    test_idx_anomaly = [idx - split_idx for idx in idx_anomaly if idx >= split_idx]
    
    # Filter out anomalies from training and validation
    train_idx_anomaly = [idx for idx in idx_anomaly if idx < split_idx]
    filtered_training_readings = np.delete(train_readings, train_idx_anomaly, axis=0)
    filtered_training_t = np.delete(train_t, train_idx_anomaly, axis=0)
    assert len(filtered_training_readings) == len(filtered_training_t) 
    # split into training and validation sets
    val_idx = int(len(filtered_training_readings) * (1-val_ratio))
    val_readings = filtered_training_readings[val_idx:]
    val_t = filtered_training_t[val_idx:]
    train_readings = filtered_training_readings[:val_idx]
    train_t = filtered_training_t[:val_idx]
    assert len(train_readings) == len(train_t) 
    assert len(val_readings) == len(val_t) 
    
    
    # create rolling windows for train, val, and test sets
    train_rolling = create_rolling_windows(train_readings, window_size)
    val_rolling = create_rolling_windows(val_readings, window_size)
    test_rolling = create_rolling_windows(test_readings, window_size)
    
    # create rolling windows for full data  (for final anomaly detection visualization)
    full_rolling = create_rolling_windows(readings_normalized, window_size)
    window_num = len(readings_normalized) // window_size
    full_rolling_anomaly_idx = [idx for idx in idx_anomaly if idx <= (window_num * window_size)]   
    
    # save the preprocessed dataset
    os.makedirs(save_dir_path, exist_ok=True)
    if save_file:
        save_path = os.path.join(save_dir_path, f"{dataset}.npz")
        np.savez(save_path, 
                 t=t, 
                 readings=readings,  
                 readings_normalized=readings_normalized, 
                 idx_anomaly=idx_anomaly, 
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
                 train_m=readings_mean, 
                 train_std=readings_std)
        
        print(f"\nPreprocessed {dataset} dataset saved at {save_path}")
    else:
        print(f"\nPreprocessed {dataset} dataset NOT saved.")
    
    
###### main function for data preprocessing ######
def main():
    parser = argparse.ArgumentParser(description="NAB Dataset Preprocessing and Save")
    parser.add_argument('-c', '--config', default='./pytorch_NAB_config.json', help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    csv_folder_path = config['csv_folder_path']
    save_dir_path = config['data_dir']
    
    datasets = ['ambient_temp', 
                'cpu_utilization', 
                'ec2_request', 
                'machine_temp', 
                # 'rogue_agent_key_hold', 
                # 'rogue_agent_key_updown', 
                'nyc_taxi']
    
    for dataset in datasets:
        preprocess_and_save(csv_folder_path, dataset, save_dir_path, window_size=config['l_win'])
    
    print("\nData preprocessing and saving complete!")
    
    
if __name__ == '__main__':
    main()
    