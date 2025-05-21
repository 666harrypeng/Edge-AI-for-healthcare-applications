import os
import numpy as np
import torch
import pandas as pd
import dash
from dash import dcc, html
import plotly.graph_objs as go
from collections import deque
import time
from datetime import datetime
import json
from VAE_LSTM_models import VAEmodel, LSTMmodel
from VAE_LSTM_data_loader import create_dataloader

class RealTimeAnomalyDetector:
    def __init__(self, config_path, vae_model_path, lstm_model_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config['device'] = self.device
        
        # Load models
        self.vae_model = VAEmodel(self.config).to(self.device)
        self.lstm_model = LSTMmodel(self.config).to(self.device)
        
        # Load model weights
        self.vae_model.load_state_dict(torch.load(vae_model_path, weights_only=True, map_location=self.device))
        self.lstm_model.load_state_dict(torch.load(lstm_model_path, weights_only=True, map_location=self.device))
        
        # Set models to evaluation mode
        self.vae_model.eval()
        self.lstm_model.eval()
        
        # Initialize data buffers
        self.window_size = self.config['l_win']
        self.look_back_num = self.config['look_back_num']
        self.input_size = self.config['input_size']
        
        # Calculate minimum required data points
        self.min_required_data = self.window_size * (self.look_back_num + 1) # look_back_num -> take as input, 1 -> next window for prediction
        
        # Set maximum length for sensor readings
        self.max_readings_length = self.config['l_win'] * 20  # Adjust this value based on your needs
        
        # Initialize data storage using deque for automatic old data removal
        self.temp_readings = deque(maxlen=self.max_readings_length)
        self.hum_readings = deque(maxlen=self.max_readings_length)
        self.audio_readings = deque(maxlen=self.max_readings_length)
        self.timestamps = deque(maxlen=self.max_readings_length)
        
        # Track last processed timestamp
        self.last_processed_timestamp = None
        
        # Track last processed window index
        self.last_processed_window_idx = -1
        
        # Initialize normalization parameters
        self.temp_mean = None
        self.temp_std = None
        self.hum_mean = None
        self.hum_std = None
        self.audio_mean = None
        self.audio_std = None
        
        
        # Initialize anomaly detection parameters
        self.threshold_percent = self.config['threshold_percent']
        self.prediction_error_deviation_ratio_threshold = self.config.get('prediction_error_deviation_ratio_threshold', 0.8)
        
        # Initialize visualization data
        self.data_length = 50000  # total data length to show
        self.x = deque(maxlen=self.data_length)
        self.y_temp = deque(maxlen=self.data_length)
        self.y_hum = deque(maxlen=self.data_length)
        self.y_audio = deque(maxlen=self.data_length)
        self.y_marked = deque(maxlen=self.data_length)
        
        # Initialize normalization update tracking
        self.norm_update_threshold = self.config['l_win'] * 5 
        self.last_norm_update_count = 0
        
        
        # Initialize anomaly scores
        self.anomaly_scores = []
        self.predicted_anomalies = []
        
        # Status message
        self.status_message = "Waiting for sufficient data..."
        self.last_window_anomaly_status = False
        
    def update_normalization_parameters(self):
        """Update normalization parameters using recent data"""
        if len(self.temp_readings) < self.norm_update_threshold:
            return False
            
        # Get the most recent data up to the threshold
        recent_data = np.column_stack((
            list(self.temp_readings)[-self.norm_update_threshold:],
            list(self.hum_readings)[-self.norm_update_threshold:],
            list(self.audio_readings)[-self.norm_update_threshold:]
        ))
        
        # Calculate new statistics
        new_temp_mean, new_temp_std = np.mean(recent_data[:, 0]), np.std(recent_data[:, 0])
        new_hum_mean, new_hum_std = np.mean(recent_data[:, 1]), np.std(recent_data[:, 1])
        new_audio_mean, new_audio_std = np.mean(recent_data[:, 2]), np.std(recent_data[:, 2])
        
        # Update parameters
        self.temp_mean, self.temp_std = new_temp_mean, new_temp_std
        self.hum_mean, self.hum_std = new_hum_mean, new_hum_std
        self.audio_mean, self.audio_std = new_audio_mean, new_audio_std
        
        # Update the counter
        self.last_norm_update_count = len(self.temp_readings)
        
        # Create status message with statistics
        stats_msg = (
            f"Normalization parameters updated:\n"
            f"Temperature: mean={self.temp_mean:.2f}, std={self.temp_std:.2f}\n"
            f"Humidity: mean={self.hum_mean:.2f}, std={self.hum_std:.2f}\n"
            f"Audio: mean={self.audio_mean:.2f}, std={self.audio_std:.2f}"
        )
        print(stats_msg)
        self.status_message = stats_msg
        
        return True
        
    def normalize_data(self, data):
        """Normalize the data using current mean and std"""
        if self.temp_mean is None:
            # First-time normalization
            self.update_normalization_parameters()
            if self.temp_mean is None:  # If still None after update attempt
                self.temp_mean, self.temp_std = np.mean(data[:, 0]), np.std(data[:, 0])
                self.hum_mean, self.hum_std = np.mean(data[:, 1]), np.std(data[:, 1])
                self.audio_mean, self.audio_std = np.mean(data[:, 2]), np.std(data[:, 2])
        
        normalized_data = np.zeros_like(data)
        normalized_data[:, 0] = (data[:, 0] - self.temp_mean) / self.temp_std
        normalized_data[:, 1] = (data[:, 1] - self.hum_mean) / self.hum_std
        normalized_data[:, 2] = (data[:, 2] - self.audio_mean) / self.audio_std
        
        return normalized_data
    
    def create_rolling_windows(self, data):
        """Create non-overlapping rolling windows"""
        # Convert data to numpy array if it's not already
        if isinstance(data, list) or isinstance(data, deque):
            data = np.array(list(data))
        window_num = len(data) // self.window_size
        return np.array([data[i * self.window_size : (i+1) * self.window_size] for i in range(window_num)])
    
    def update_data(self, csv_file):
        """Update data from CSV file and perform anomaly detection"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}]============ update_data =============")
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Convert timestamp strings to datetime objects
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # If this is the first update, process all data
            if self.last_processed_timestamp is None:
                self.last_processed_timestamp = df['Timestamp'].iloc[0]
                new_data = df
            else:
                # Get only new data since last processed timestamp
                new_data = df[df['Timestamp'] > self.last_processed_timestamp]
            
            if len(new_data) == 0:
                self.status_message = f"Waiting for new data... Last update: {self.last_processed_timestamp}"
                return False
            
            # Update last processed timestamp
            self.last_processed_timestamp = new_data['Timestamp'].iloc[-1]
            
            # Extract sensor readings
            temp_readings = new_data['Temperature(C)'].values
            hum_readings = new_data['Humidity(%)'].values
            audio_readings = new_data['Audio'].values
            timestamps = new_data['Timestamp'].values
            
            # Update data buffers (deque will automatically remove old data if maxlen is reached)
            self.temp_readings.extend(temp_readings)
            self.hum_readings.extend(hum_readings)
            self.audio_readings.extend(audio_readings)
            self.timestamps.extend(timestamps)
            
            # Check if we should update normalization parameters
            if len(self.temp_readings) >= self.norm_update_threshold and \
               len(self.temp_readings) - self.last_norm_update_count >= self.norm_update_threshold:
                self.update_normalization_parameters()
            
            # Stack the readings for anomaly detection
            readings = np.column_stack((list(self.temp_readings), list(self.hum_readings), list(self.audio_readings)))
            
            # Normalize the data
            normalized_readings = self.normalize_data(readings)
            
            # Update visualization data with normalized values
            for i in range(len(temp_readings)):
                self.x.append(len(self.x))
                norm_idx = len(normalized_readings) - len(temp_readings) + i
                # Guard against out-of-bounds
                if 0 <= norm_idx < len(normalized_readings):
                    self.y_temp.append(normalized_readings[norm_idx, 0])
                    self.y_hum.append(normalized_readings[norm_idx, 1])
                    self.y_audio.append(normalized_readings[norm_idx, 2])
                else:
                    # Fallback: append 0 if index is out of bounds
                    self.y_temp.append(0)
                    self.y_hum.append(0)
                    self.y_audio.append(0)
                self.y_marked.append(1)  # Default marked as normal
            
            # Check if we have enough data for anomaly detection
            if len(self.temp_readings) < self.min_required_data:
                self.status_message = f"Waiting for sufficient data... ({len(self.temp_readings)}/{self.min_required_data} points)"
                return False
            
            # Create rolling windows
            windows = self.create_rolling_windows(normalized_readings)
            
            if len(windows) == 0:
                print(f"[{current_time}] update_data -> NO rolling windows!")
                self.status_message = f"Waiting for complete window... Current data points: {len(normalized_readings)}"
                return False
            
            # Calculate current window index
            current_window_idx = len(windows) - 1
            
            # Only process if we have a new complete window
            if current_window_idx <= self.last_processed_window_idx:
                # Calculate remaining points needed for next window
                total_data_points = len(self.temp_readings)
                processed_points = (self.last_processed_window_idx + 1) * self.window_size
                current_new_points = total_data_points - processed_points
                
                print(f"[{current_time}] update_data -> NO new complete window! current_window_idx: {current_window_idx}, last_window_anomaly_status: {self.last_window_anomaly_status}")
                self.status_message = f"Waiting for new data points... ({current_new_points}/{self.window_size} points needed for next window, last_window_anomaly_status: {self.last_window_anomaly_status})"
                return False
            
            # Update last processed window index
            self.last_processed_window_idx = current_window_idx
            
            # Reshape for model input
            windows = windows.reshape(-1, self.window_size, self.input_size)
            windows = torch.tensor(windows, dtype=torch.float32).to(self.device)
            windows = windows.permute(0, 2, 1).unsqueeze(-1)
            
            # Perform anomaly detection
            with torch.no_grad():
                # VAE reconstruction
                reconstructed, _, _ = self.vae_model(windows)
                # Calculate reconstruction error for each window
                reconstruction_error = torch.sum(((windows - reconstructed) ** 2), dim=(1, 2, 3)).cpu().numpy()
                
                # Generate embeddings
                embeddings = []
                for window in windows:
                    mean, _ = self.vae_model.encode(window.unsqueeze(0))
                    mean = mean.squeeze(-1).squeeze(0)
                    mean = mean.permute(1, 0)
                    embeddings.append(mean.cpu())
                print(f"[{current_time}] update_data -> finish generate embeddings")
                
                # Create LSTM sequences
                x_lstm, y_lstm = [], []
                for i in range(len(embeddings) - self.look_back_num):
                    x_lstm.append(embeddings[i : i + self.look_back_num])
                    y_lstm.append(embeddings[i + self.look_back_num])
                print(f"[{current_time}] update_data -> finish create LSTM sequences")
                
                if len(x_lstm) == 0:
                    print(f"[{current_time}] update_data -> NO LSTM sequences! How come?")
                    self.status_message = f"Waiting for LSTM sequences... Current embeddings: {len(embeddings)}, Required lookback: {self.look_back_num}"
                    return False
                
                x_lstm = torch.tensor(np.array(x_lstm), dtype=torch.float32).to(self.device)
                y_lstm = torch.tensor(np.array(y_lstm), dtype=torch.float32).to(self.device)
                
                # LSTM prediction
                prediction_errors = []
                prediction_error_deviation_ratios = []
                
                for i in range(len(x_lstm)):
                    inputs = x_lstm[i : i+1]
                    targets = y_lstm[i : i+1]
                    
                    outputs = self.lstm_model.lstm_infer(inputs)
                    
                    targets = targets.permute(0, 2, 1).unsqueeze(-1)
                    outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
                    
                    decode_targets_recon = self.vae_model.decode(targets)
                    decode_outputs_recon = self.vae_model.decode(outputs)
                    
                    decode_targets_recon = decode_targets_recon.cpu().detach().numpy()
                    decode_outputs_recon = decode_outputs_recon.cpu().detach().numpy()
                    
                    prediction_error = np.mean(((decode_targets_recon - decode_outputs_recon) ** 2))
                    prediction_errors.append(prediction_error)
                    
                    epsilon = 1e-10
                    prediction_error_deviation_ratio = np.abs((np.mean(prediction_error) - np.mean(decode_targets_recon)) / (np.mean(decode_targets_recon) + epsilon))
                    prediction_error_deviation_ratios.append(prediction_error_deviation_ratio)
                
                # Calculate anomaly scores
                reconstruction_errors = reconstruction_error[self.look_back_num:]
                prediction_errors = np.array(prediction_errors)
                
                anomaly_scores = reconstruction_errors + 0.5 * prediction_errors
                
                # Find anomalies
                anomaly_threshold = np.percentile(anomaly_scores, self.threshold_percent)
                
                predicted_anomalies = []
                for i, score in enumerate(anomaly_scores):
                    if score >= anomaly_threshold: # statistical anomaly
                        if prediction_error_deviation_ratios[i] >= self.prediction_error_deviation_ratio_threshold: # check deviation ratio
                            window_idx = i + self.look_back_num
                            if window_idx < len(anomaly_scores) + self.look_back_num:
                                predicted_anomalies.append(window_idx)
                                                
                # Mark anomalies
                for idx in predicted_anomalies:
                    start_idx = idx * self.window_size
                    end_idx = (idx + 1) * self.window_size
                    for i in range(start_idx, min(end_idx, len(self.y_marked))):
                        self.y_marked[i] = 0
                print(f"[{current_time}] update_data -> finish mark anomalies")
                self.status_message = f"Anomaly detection complete. Found {len(predicted_anomalies)} anomalies. Last window anomaly status: {self.last_window_anomaly_status}"
                return True
                
        except Exception as e:
            print(f"[{current_time}] Error in update_data: {e}")
            self.status_message = f"Error: {str(e)}"
            return False

# Initialize Dash app
app = dash.Dash(__name__)

# Initialize detector
detector = RealTimeAnomalyDetector(
    config_path='./checkpoint_models/VAE-LSTM/VAE-LSTM_config.json',
    vae_model_path='./checkpoint_models/VAE-LSTM/vae_best_model.pth',
    lstm_model_path='./checkpoint_models/VAE-LSTM/lstm_best_model.pth'
)

# Get model config for display
l_win = detector.config.get('l_win', 400)
look_back_num = detector.config.get('look_back_num', 1)

# App layout
app.layout = html.Div([
    html.H1(f'Real-time Respiratory Anomaly Detection (l_win: {l_win}, look_back_num: {look_back_num})'),
    html.Div(id='status-message', style={'margin': '10px', 'font-weight': 'bold'}),
    dcc.Graph(id='sensor-graph'),
    # dcc.Graph(id='anomaly-graph'),
    dcc.Interval(id='interval-component', interval=500)  # Update every <interval> ms
])

@app.callback(
    [dash.dependencies.Output('sensor-graph', 'figure'),
    #  dash.dependencies.Output('anomaly-graph', 'figure'),
     dash.dependencies.Output('status-message', 'children')],
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    try:
        # Update data from CSV file
        detector.update_data('../hardware_ypengbb/sensor_data.csv')
        
        # Create sensor data graph
        sensor_figure = {
            'data': [
                go.Scatter(x=list(detector.x), y=list(detector.y_temp), mode='lines', name='Temperature', line=dict(color='#4292c6')),
                go.Scatter(x=list(detector.x), y=list(detector.y_hum), mode='lines', name='Humidity', line=dict(color='#084594')),
                go.Scatter(x=list(detector.x), y=list(detector.y_audio), mode='lines', name='Audio', line=dict(color='#88419d'))
            ],
            'layout': go.Layout(
                title='Sensor Data',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Value')
            )
        }
        
        # Create anomaly graph with highlighted regions
        shapes = []
        x_vals = list(detector.x)
        marked = list(detector.y_marked)
        window_size = detector.window_size
        n_points = len(x_vals)
        for i in range(0, n_points, window_size):
            # Only consider windows that are fully within the visible region
            if i + window_size <= n_points:
                if all(mark == 0 for mark in marked[i:i + window_size]):
                    shapes.append({
                        'type': 'rect',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': x_vals[i],
                        'x1': x_vals[i + window_size - 1],
                        'y0': 0,
                        'y1': 1,
                        'fillcolor': 'rgba(255, 0, 0, 0.1)',  # Reduced opacity for better visibility
                        'line': {'width': 0}
                    })
        
        anomaly_figure = {
            'data': [
                go.Scatter(x=x_vals, y=list(detector.y_temp), mode='lines', name='Temperature', line=dict(color='#4292c6')),
                go.Scatter(x=x_vals, y=list(detector.y_hum), mode='lines', name='Humidity', line=dict(color='#084594')),
                go.Scatter(x=x_vals, y=list(detector.y_audio), mode='lines', name='Audio', line=dict(color='#88419d'))
            ],
            'layout': go.Layout(
                title='Anomaly Detection',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Value'),
                shapes=shapes
            )
        }
        
        # return sensor_figure, anomaly_figure, detector.status_message
        return sensor_figure, detector.status_message
        
    except Exception as e:
        print(f"Error in update_graph: {e}")
        return {}, {}, f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True) 
    '''
    default port: 8050 (on localhost)
    default URL: http://127.0.0.1:8050/
    
    1. run the script -> python VAE_LSTM_real_time_detection_with_ui.py
    2. open the browser -> http://127.0.0.1:8050/
    '''