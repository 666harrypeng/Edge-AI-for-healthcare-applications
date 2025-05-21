import json
import os
import numpy as np
import argparse
from datetime import datetime
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

def get_args():
    """
    Parse command-line arguments for the configuration file.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PyTorch MHA-VAE-LSTM anomaly detection")
    parser.add_argument('-c', '--config', required=True, default='mha_vae_lstm_config.json', help='Path to the configuration JSON file')
    return parser.parse_args()

def get_config_from_json(json_file):
    """
    Load configuration from a JSON file.
    :param json_file: Path to the JSON file.
    :return: Configuration as a dictionary.
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict


def save_config(config):
    """
    Save the configuration as a JSON file with a timestamped filename.
    :param config: Configuration dictionary.
    """
    date_time_obj = datetime.now()
    timestamp_str = date_time_obj.strftime("%Y-%m-%d_%H-%M")
    filename = os.path.join(config['result_dir'], f'training_config_{timestamp_str}.json')
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)


def process_config(json_file):
    """
    Process the configuration JSON file and set up paths for results and checkpoints.
    :param json_file: Path to the configuration file.
    :return: Updated configuration dictionary.
    """
    config = get_config_from_json(json_file)

    # Define directory structure for results, summaries, and checkpoints
    save_dir = os.path.join("./experiments", config['exp_name'], config['dataset'])
    # save_name = f"{config['exp_name']}-{config['dataset']}-win_len{config['l_win']}-look_back{config['look_back_num']}-future_seq{config['l_seq']}-codesize{config['code_size']}-thre{config['threshold_percent']}-VAEunits{config['num_hidden_units_vae']}-VAElr{config['vae_learning_rate']}-LSTMunits{config['num_hidden_units_lstm']}-LSTMlr{config['lstm_learning_rate']}-bs{config['batch_size']}"
    # save_name = f"{config['exp_name']}-{config['dataset']}-win_len{config['l_win']}-look_back{config['look_back_num']}-future_seq{config['l_seq']}-codesize{config['code_size']}-VAEunits{config['num_hidden_units_vae']}-LSTMunits{config['num_hidden_units_lstm']}"
    save_name = f"{config['exp_name']}-{config['dataset']}-win_len{config['l_win']}-look_back{config['look_back_num']}-batchsize{config['batch_size']}-VAE_units{config['num_hidden_units_vae']}-codesize{config['code_size']}-LSTM_batchsize{config['lstm_batch_size']}-layers{config['num_layers']}-units{config['num_hidden_units_lstm']}"
    config['summary_dir'] = os.path.join(save_dir, save_name, "summary/")
    config['result_dir'] = os.path.join(save_dir, save_name, "result/")
    config['checkpoint_dir'] = os.path.join(save_dir, save_name, "checkpoint/")
    config['checkpoint_dir_lstm'] = os.path.join(config['checkpoint_dir'], "lstm/")

    return config



def create_dirs(dirs):
    """
    Create directories if they do not exist.
    :param dirs: List of directory paths.
    """
    try:
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
    except Exception as err:
        print(f"Error creating directories: {err}")
        exit(-1)


def evaluate_detection(true_anomalies, predicted_anomalies, dataset_length):
    """
    Evaluate precision, recall, and F1 score for anomaly detection.
    :param true_anomalies: Ground truth anomaly indices.
    :param predicted_anomalies: Detected anomaly indices.
    :param dataset_length: Total number of data points.
    """
    y_true = np.zeros(dataset_length)
    y_pred = np.zeros(dataset_length)

    # Mark anomaly points
    y_true[true_anomalies] = 1
    y_pred[predicted_anomalies] = 1

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return precision, recall, f1