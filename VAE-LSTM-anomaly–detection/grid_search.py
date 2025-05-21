import json
import os
import itertools
import subprocess
from datetime import datetime
from pathlib import Path

# Define parameter grid
param_grid = {
    "dataset": ["nosein_noseout_9"],
    "l_win": [300, 600, 1000, 1500, 2000],
    "look_back_num": [1, 2, 3],
    "threshold_percent": [60, 70, 80, 90]
}

# Create directory for grid search configurations
grid_search_dir = Path("grid_search_configs_0425-nosein_noseout9-final_winlen_validation_with_dev_bar80-VAE_LSTM")
grid_search_dir.mkdir(exist_ok=True)

# Create configs directory
configs_dir = grid_search_dir / "configs"
configs_dir.mkdir(exist_ok=True)

# Save parameter grid for reference
with open(grid_search_dir / "param_grid.json", 'w') as f:
    json.dump(param_grid, f, indent=4)

# Create results summary file
results_file = grid_search_dir / "grid_search_results.csv"
with open(results_file, 'w') as f:
    f.write("config_id,dataset,l_win,look_back_num,threshold_percent,detection_rate,f_beta\n")

def get_config_path(config_id, folder_length=20):
    """Get the path for a config file, organizing into subfolders of <folder_length> configs each"""
    # Convert config_id to int for calculation
    id_num = int(config_id)
    # Calculate subfolder number (1-based)
    subfolder = (id_num - 1) // folder_length + 1
    # Create subfolder if it doesn't exist
    subfolder_path = configs_dir / f"batch_{subfolder:03d}"
    subfolder_path.mkdir(exist_ok=True)
    return subfolder_path / f"config_{config_id}.json"

def run_experiment(config, config_id):
    """Run preprocessing and training for a single configuration"""
    # Get organized config path
    config_path = get_config_path(config_id)
    
    # Save config file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    try:
        # Run preprocessing
        print(f"\nRunning preprocessing for config {config_id}...")
        preprocess_cmd = f"python collected_data_preprocess.py -c {config_path}"
        subprocess.run(preprocess_cmd, shell=True, check=True)
        
        # Run training
        print(f"\nRunning training for config {config_id}...")
        train_cmd = f"python vae_lstm_anomaly_detection_main.py -c {config_path}"
        subprocess.run(train_cmd, shell=True, check=True)
        
        # Extract detection rate from results (using the path from config)
        ''' always check the result_dir path with the main script's definition '''
        save_dir = os.path.join("./experiments", config['exp_name'], config['dataset'])
        result_dir = os.path.join(save_dir, f"{config['exp_name']}-{config['dataset']}-win_len{config['l_win']}-look_back{config['look_back_num']}-codesize{config['code_size']}-VAEunits{config['num_hidden_units_vae']}-LSTMunits{config['num_hidden_units_lstm']}", "result/")
        
        # Extract detection rate
        detection_rate = -1
        try:
            with open(os.path.join(result_dir, "detection_metrics.txt"), 'r') as f:
                for line in f:
                    if line.startswith("recall"):
                        detection_rate = float(line.strip().split(',')[1])
                        break
        except:
            detection_rate = -1  # Indicates failure to read detection rate
        
        # Extract F-beta score
        f_beta = -1
        try:
            with open(os.path.join(result_dir, "detection_metrics.txt"), 'r') as f:
                for line in f:
                    if line.startswith("f_beta"):
                        f_beta = float(line.strip().split(',')[1])
                        break
        except:
            f_beta = -1  # Indicates failure to read F-beta score
        
        # Log results
        with open(results_file, 'a') as f:
            f.write(f"{config_id},{config['dataset']},{config['l_win']},{config['look_back_num']},{config['threshold_percent']},{detection_rate},{f_beta}\n")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {config_id}: {e}")
        return False

def main():
    # Load base configuration
    with open('vae_lstm_config.json', 'r') as f:
        base_config = json.load(f)
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = len(list(itertools.product(*param_values)))
    print(f"Total number of combinations to try: {total_combinations}")
    
    # Run grid search
    for i, values in enumerate(itertools.product(*param_values)):
        config_id = f"{i+1:03d}"
        print(f"\nRunning combination {i+1}/{total_combinations}")
        
        # Create config for this combination
        config = base_config.copy()
        for name, value in zip(param_names, values):
            config[name] = value
            
        # Set training flags
        config['TRAIN_VAE'] = False # in the main file: if no existing -> train a new; if existing -> load the existing model
        config['TRAIN_LSTM'] = False # in the main file: if no existing -> train a new; if existing -> load the existing model
        config['Anomaly_Detection'] = True  
        
        # Run experiment
        success = run_experiment(config, config_id)
        if not success:
            print(f"Skipping to next combination due to error")
            continue
        
if __name__ == "__main__":
    main() 