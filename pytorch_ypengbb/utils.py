import json
import os
import numpy as np
import argparse
from datetime import datetime
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch VAE-LSTM anomaly detection")
    parser.add_argument('-c', '--config', required=True, default='./pytorch_NAB_config.json', help='Path to the configuration JSON file')
    return parser.parse_args()

def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict


def save_config(config):
    date_time_obj = datetime.now()
    timestamp_str = date_time_obj.strftime("%Y-%m-%d_%H-%M")
    filename = os.path.join(config['result_dir'], f'training_config_{timestamp_str}.json')
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)


def process_config(json_file):
    config = get_config_from_json(json_file)

    # Define directory structure for results, summaries, and checkpoints
    save_dir = os.path.join("./experiments", config['exp_name'], config['dataset'])
    save_name = f"{config['exp_name']}-{config['dataset']}-win{config['l_win']}-seq{config['l_seq']}-codesize{config['code_size']}"
    config['summary_dir'] = os.path.join(save_dir, save_name, "summary/")
    config['result_dir'] = os.path.join(save_dir, save_name, "result/")
    config['checkpoint_dir'] = os.path.join(save_dir, save_name, "checkpoint/")
    config['checkpoint_dir_lstm'] = os.path.join(config['checkpoint_dir'], "lstm/")

    return config



def create_dirs(dirs):
    try:
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
    except Exception as err:
        print(f"Error creating directories: {err}")
        exit(-1)


def evaluate_detection(true_anomalies, predicted_anomalies, dataset_length):
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