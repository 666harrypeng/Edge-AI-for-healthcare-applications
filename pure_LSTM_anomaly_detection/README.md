# Pure LSTM for Respiratory Anomaly Detection

This repository contains the implementation of a pure LSTM model for respiratory anomaly detection. The implementation supports both the Numenta Anomaly Benchmark (NAB) datasets and project-custom collected respiratory datasets.

## Repository Structure

```
├── lstm_model.py                # Core LSTM model architecture
├── trainers.py                  # Training and evaluation logic
├── data_loader.py              # Custom data loading and preprocessing
├── grid_search.py              # Hyperparameter optimization
├── pure_lstm_anomaly_detection_main.py  # Main training and inference script
├── utils.py                    # Utility functions
├── collected_data_preprocess.py # Preprocessing for project-custom datasets
├── data_preprocess_NAB.py      # Preprocessing for NAB datasets
├── pure_lstm_config.json       # Configuration for project-custom datasets
├── pure_lstm_config_NAB.json   # Configuration for NAB datasets
├── dataset_config.json         # Dataset configuration
├── cmd_script.sh              # Shell script for workflow automation
├── experiments/               # Training logs and model checkpoints
└── datasets/                  # Processed datasets
    ├── NAB-known-anomaly/    # NAB benchmark datasets
    └── collected-known-anomaly/ # Project-custom respiratory datasets
```

## Features

### 1. Model Architecture
- **Pure LSTM**: Temporal pattern recognition
- **Configurable Layers**: Adjustable number of LSTM layers
- **Flexible Input Size**: Supports both single and multi-modal inputs

### 2. Training Pipeline
- **Hyperparameter Optimization**:
  - Grid search for optimal model parameters
  - Automated hyperparameter tuning
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Visualization**: Regular training progress monitoring

### 3. Dataset Support

#### NAB Datasets
- **Preprocessing**: `data_preprocess_NAB.py`
- **Configuration**: `pure_lstm_config_NAB.json`
- **Features**:
  - Single-variable time series
  - Standard benchmark datasets
  - Known anomaly labels

#### Project-Custom Respiratory Datasets
- **Preprocessing**: `collected_data_preprocess.py`
- **Configuration**: `pure_lstm_config.json`
- **Features**:
  - Multi-modal sensor data
  - Custom respiratory patterns
  - Project-specific anomaly intervals

## Configuration

### Project-Custom Configuration (`pure_lstm_config.json`)
- Training parameters (epochs, learning rates, batch sizes)
- Model architecture parameters (hidden units, layers)
- Anomaly detection thresholds
- Dataset-specific parameters

### NAB Configuration (`pure_lstm_config_NAB.json`)
- NAB-specific parameters
- Single-variable input configuration
- Benchmark-specific thresholds

## Usage

### For Project-Custom Datasets

1. **Data Preparation**:
   ```bash
   python collected_data_preprocess.py
   ```

2. **Model Training**:
   ```bash
   # For single training run
   python pure_lstm_anomaly_detection_main.py

   # For hyperparameter grid search
   python grid_search.py
   ```

3. **Automated Workflow**:
   ```bash
   ./cmd_script.sh
   ```

### For NAB Datasets

1. **Data Preparation**:
   ```bash
   python data_preprocess_NAB.py
   ```

2. **Model Training**:
   ```bash
   # Update config to use NAB settings
   python pure_lstm_anomaly_detection_main.py
   ```

## Key Parameters

### Project-Custom Model
- Learning rate: 0.001
- Batch size: 32
- Hidden units: 128
- Layers: 3
- Patience: 15

### NAB Model
- Learning rate: 0.00001
- Batch size: 10
- Hidden units: 256
- Layers: 4
- Patience: 40

## Output

The training process generates:
- Model checkpoints in `experiments/`
- Training logs and metrics
- Visualization plots for model performance
- Anomaly detection results

## Notes

- Ensure proper GPU configuration for training
- NAB datasets use single-variable input, while project-custom datasets support multi-modal input
- Adjust batch sizes based on available memory
- Monitor training progress through visualization intervals
- Use appropriate anomaly thresholds for your specific dataset type
