import os
import sys
import time
import argparse
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
class real_time_detection:
    def __init__(self, config_file, model_path, data_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.win_len = self.config['l_win']
        self.look_back_num = self.config['look_back_num']
        self.input_size = self.config['input_size']
        
        self.model_path = model_path
        self.data_file = data_file
        self.min_local_samples = self.win_len * (self.look_back_num + 1) # look_back_num & current window
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config['device'] = self.device
        print(f"Using device: {self.device}")
        
        self.model = self.load_model(model_path)
        self.model.eval()
        
        self.last_processed_line = 0
        self.start_time_stamp = None
        self.end_time_stamp = None
        
        self.look_back_windows_data = []
        self.current_window_data = []
        self.future_window_data = []
        
        self.max_local_data_length = 10000
        self.t_local = np.array([])
        self.temp_readings_local = np.array([])
        self.hum_readings_local = np.array([])
        self.audio_readings_local = np.array([])
        self.temp_local_mean, self.temp_local_std = 0, 0
        self.hum_local_mean, self.hum_local_std = 0, 0
        self.audio_local_mean, self.audio_local_std = 0, 0
        self.readings_local_mean, self.readings_local_std = None, None
        self.readings_normalized_local = []
        self.full_data_rolling = []
        self.dataloader_local = None
        
        self.prediction_errors = []
        self.predicted_anomalies_local_window_idx_list = None
        
        self.running = True
        self.visualization_thread = None
        self.local_data_ready = False
        self.current_time = None
        
    def load_collected_data(self): 
        '''
        return True <- data file is ready & loaded in lists
        return False <- data file is not ready
        '''
        if not os.path.exists(self.data_file):
            print(f"Data file {self.data_file} does not exist")
            return False

        try: 
            with open(self.data_file, 'r') as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                next(readCSV) # skip header row
                rows = list(readCSV)
                
                total_lines = len(rows)
                
                # check whether new data
                if total_lines < self.min_local_samples: # check all file length > min_local_samples
                    print(f"Fail to load data --> but data file has {total_lines} samples, need at least {self.min_local_samples} samples")
                    time.sleep(1)
                    return False
                
                new_lines = total_lines - self.last_processed_line
                
                if new_lines <= 0:
                    print("No new data to read")
                    time.sleep(1)
                    return False
                
                # check whether enough new data
                if new_lines < self.win_len:
                    print(f"Data file has {new_lines} new samples, need at least {self.win_len} samples")
                    time.sleep(1)
                    return False
                
                ##### we have enough new data #####
                valid_rows = rows[self.last_processed_line : total_lines]
                for i, row in enumerate(valid_rows):
                    self.t_local = np.append(self.t_local, i)
                    self.audio_readings_local = np.append(self.audio_readings_local, float(row[1]))
                    self.temp_readings_local = np.append(self.temp_readings_local, float(row[2]))
                    self.hum_readings_local = np.append(self.hum_readings_local, float(row[3]))
                
                # check each local list length -> if too long, trim the oldest data
                if len(self.t_local) > self.max_local_data_length:
                    self.t_local = self.t_local[-self.max_local_data_length : ]
                    self.temp_readings_local = self.temp_readings_local[-self.max_local_data_length : ]
                    self.hum_readings_local = self.hum_readings_local[-self.max_local_data_length : ]
                    self.audio_readings_local = self.audio_readings_local[-self.max_local_data_length : ]
                    
                self.last_processed_line = total_lines
                # self.start_time_stamp = self.t_local[0]
                # self.end_time_stamp = self.t_local[-1]
                
                return True
            
        except Exception as e:
            print(f"Error in load_collected_data: {e}")
            time.sleep(1)
            return False
            
    def preprocess_data(self):
        try:
            ### data is ready! >>> continue preprocess data ###
            # normalization on local data
            self.data_normalization()
            
            # create rolling windows
            self.full_data_rolling = self.create_rolling_windows(self.readings_normalized_local)
            
            # create dataloader
            self.full_data_rolling = self.full_data_rolling.reshape(-1, self.win_len, self.input_size)
            
            X, y = [], []
            for i in range(len(self.full_data_rolling) - self.look_back_num - 1):
                X.append(self.full_data_rolling[i : i+self.look_back_num])
                y.append(self.full_data_rolling[i + self.look_back_num])
            X = torch.tensor(np.array(X), dtype=torch.float32)
            y = torch.tensor(np.array(y), dtype=torch.float32)
            
            dataset = CustomDataset(X, y)
            self.dataloader_local = DataLoader(dataset, batch_size=1, shuffle=False)
            if self.dataloader_local is not None:
                print("Dataloader created successfully")
            else:
                print("Dataloader creation failed")
                
            return
        
        except Exception as e:
            print(f"Error in preprocess_data loop: {e}")
            time.sleep(1)
    
    
    def find_anomalies(self):
        # reset the prediction errors
        self.prediction_errors = []
        
        threshold_percentage = self.config['threshold_percent']
        
        with torch.no_grad():
            print(f"{self.current_time} --- In find_anomalies, dataloader_local length: {len(self.dataloader_local)}")
            for batch_X, batch_y in self.dataloader_local:
                print(f"{self.current_time} --- In find_anomalies, batch_X shape: {batch_X.shape}, batch_y shape: {batch_y.shape}")
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # predict 
                print(f"{self.current_time} --- In find_anomalies, predicting...")
                pred_outputs = self.model.lstm_infer(batch_X)
                
                # de-normalize data values
                print(f"{self.current_time} --- In find_anomalies, de-normalizing data values...")
                pred_outputs = pred_outputs.cpu().detach().numpy() * self.readings_local_std + self.readings_local_mean
                targets = batch_y.cpu().detach().numpy() * self.readings_local_std + self.readings_local_mean
                
                # calculate errors
                print(f"{self.current_time} --- In find_anomalies, calculating errors...")
                pred_error = np.mean((pred_outputs - targets) ** 2)
                self.prediction_errors.append(pred_error)
            
            self.prediction_errors = np.array(self.prediction_errors)
            print(f"{self.current_time} --- In find_anomalies, prediction_errors shape: {self.prediction_errors.shape}")
            anomaly_threshold = np.percentile(self.prediction_errors, threshold_percentage)
            print(f"{self.current_time} --- finish computing threshold at {threshold_percentage}% anomaly rate: {anomaly_threshold:.5f}" )
            
            # get indices for predicted anomaly "window" in this current full dataset 
            print(f"{self.current_time} --- In find_anomalies, getting indices for predicted anomaly windows...")
            self.predicted_anomalies_local_window_idx_list = np.array([i for i, prediction_err in enumerate(self.prediction_errors) if prediction_err >= anomaly_threshold])
        
    def create_rolling_windows(self, data):
        ## non-overlapped windows
        window_size = self.win_len
        window_num = len(data) // window_size
        return np.array([data[i * window_size : (i+1) * window_size] for i in range(window_num)])
    
    def data_normalization(self):
        self.temp_local_mean, self.temp_local_std = np.mean(self.temp_readings_local), np.std(self.temp_readings_local)
        self.hum_local_mean, self.hum_local_std = np.mean(self.hum_readings_local), np.std(self.hum_readings_local)
        self.audio_local_mean, self.audio_local_std = np.mean(self.audio_readings_local), np.std(self.audio_readings_local)
        
        self.temp_readings_local = (self.temp_readings_local - self.temp_local_mean) / self.temp_local_std
        self.hum_readings_local = (self.hum_readings_local - self.hum_local_mean) / self.hum_local_std
        self.audio_readings_local = (self.audio_readings_local - self.audio_local_mean) / self.audio_local_std
        
        # stack temp, hum, audio
        self.readings_normalized_local = np.column_stack((self.temp_readings_local, self.hum_readings_local, self.audio_readings_local))
        
        # update mean & std in stack list
        self.readings_local_mean = (self.temp_local_mean, self.hum_local_mean, self.audio_local_mean)
        self.readings_local_std = (self.temp_local_std, self.hum_local_std, self.audio_local_std)
        
    def load_model(self, model_path):
        """Load the LSTM model"""
        # Import the model class
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pure_LSTM_anomaly_detection.lstm_model import LSTMModel
        
        # Create and load the model
        model = LSTMModel(self.config).to(self.device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        return model
    
    def visualize_anomalies(self, full_normalized_data_used):
        # close previous plot if exists
        plt.close("all")
        # plt.clf()
        
        # plot new figure
        plt.figure(figsize=(12, 5))
        color_list = ['#4292c6','#084594', '#88419d', 'mediumpurple', '#F8AC8C', '#F6CAE5', '#96CCCB']
        # plot full data of current local windows
        for modal_idx in range(full_normalized_data_used.shape[1]):
            plt.plot(full_normalized_data_used[:, modal_idx], label=f"Full_data_modal_{modal_idx+1}", color=color_list[modal_idx % len(color_list)], alpha=0.8)
        
        # Highlight predicted anomalies
        for idx in self.predicted_anomalies_local_window_idx_list:
            start = idx * self.win_len
            end = (idx + 1) * self.win_len
            plt.axvspan(start, end, color='green', alpha=0.2, label='Predicted Anomaly Windows' if start == self.predicted_anomalies_local_window_idx_list[0] else "")
            
        # visualize
        plt.title(f"Anomaly Detection Results\n(End Time: {self.current_time})")
        plt.xlabel("Timestamps")
        plt.ylabel("Normalized Readings")
        plt.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(2) # last for 2 seconds
    
    def real_time_detection_main_loop(self):
        plt.ion() # interactive mode
        
        while self.running:
            try:
                self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{self.current_time} --- Real-time detection main loop running...")
                # check data ready
                if not self.local_data_ready:
                    print(f"{self.current_time} --- Loading collected data...")
                    self.local_data_ready = self.load_collected_data()
                    if not self.local_data_ready:
                        time.sleep(1)
                        print(f"{self.current_time} --- Data not ready, continue...\n")
                        continue
                
                # preprocess data
                print(f"{self.current_time} --- Preprocessing data...")
                self.preprocess_data()
                
                # check dataloader_local length
                if len(self.dataloader_local) == 0:
                    print(f"{self.current_time} --- dataloader_local length is 0 (too short), continue...")
                    time.sleep(1)
                    continue

                # find anomalies
                print(f"{self.current_time} --- Finding anomalies...")
                self.find_anomalies()
                
                # convert predicted window indices to intervals
                print(f"{self.current_time} --- Converting predicted window indices to intervals...")
                predicted_intervals = [(win_idx * self.win_len, (win_idx + 1) * self.win_len) for win_idx in self.predicted_anomalies_local_window_idx_list]

                # visualize anomalies
                print(f"{self.current_time} --- Visualizing anomalies...")
                used_data_length = len(self.dataloader_local) * self.win_len
                full_normalized_data_used = self.readings_normalized_local[ : used_data_length]
                self.visualize_anomalies(full_normalized_data_used)
                
                # reset data ready status -> wait for new data
                self.local_data_ready = False
                
                print(f"{self.current_time} --- Real-time detection main loop finished\n")
            except Exception as e:
                print(f"Error in real_time_detection_main_loop: {e}\n")
                time.sleep(1)
        plt.ioff() # non-interactive mode
            
    # def start_detection(self):
    #     if not self.running:
    #         print("Starting detection...")
    #         self.running = True
    #         self.visualization_thread = threading.Thread(target=self.real_time_detection_main_loop)
    #         self.visualization_thread.daemon = True
    #         self.visualization_thread.start()
            
            
    
    # def stop_detection(self):
    #     if self.running:
    #         self.running = False
            
    #         plt.close()
            
    #         print("Detection stopped")
        
        
def main():
    parser = argparse.ArgumentParser(description="Real-time sleep apnea detection")
    
    parser.add_argument("--config_file", type=str, default="./checkpoint_models/pureLSTM/pure_LSTM_config.json",
                        help="Path to the configuration file")
    parser.add_argument("--model_path", type=str, default="./checkpoint_models/pureLSTM/lstm_best_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--data_file", type=str, default="../hardware_ypengbb/sensor_data.csv",
                        help="Path to the CSV file containing sensor data")
    
    args = parser.parse_args()

    detector = real_time_detection(args.config_file, args.model_path, args.data_file)
    
    # plt.ion() # interactive mode
    
    detector.real_time_detection_main_loop()
    
    # The detection will run until the window is closed
    # When the window is closed, the detection will stop automatically
    
    # plt.ioff() # non-interactive mode
    
if __name__ == "__main__":
    main()