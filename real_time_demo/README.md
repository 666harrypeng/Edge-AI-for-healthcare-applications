# Real-time Respiratory Anomaly Detection Demo

This repository contains the implementation of real-time respiratory anomaly detection with a graphical user interface (GUI).

> [!NOTE]
> Currently, this demo supports only the VAE-LSTM model for real-time inference.

## Repository Structure

```
├── VAE_LSTM_real_time_detection_with_ui.py  # Main GUI and inference script
├── VAE_LSTM_models.py                       # VAE-LSTM model architecture
├── VAE_LSTM_data_loader.py                  # Real-time data loading
├── checkpoint_models/                       # Pre-trained model checkpoints
│   └── VAE-LSTM/                           # VAE-LSTM model checkpoints
│       ├── model_checkpoint.pth            # Model weights
│       └── model_config.json               # Model configuration
└── .gitignore
```

## Model Checkpoint Organization

### Manual Setup Required
The `checkpoint_models` directory must be manually organized by the user:

1. **Create Directory Structure**:
   ```bash
   mkdir -p checkpoint_models/VAE-LSTM
   ```

2. **Place Model Files**:
   - Copy your trained model checkpoint (`.pth` file) to `checkpoint_models/VAE-LSTM/`
   - Copy the corresponding model configuration (`.json` file) to `checkpoint_models/VAE-LSTM/`
   - Rename files to match the expected names:
     - Model checkpoint: `model_checkpoint.pth`
     - Configuration: `model_config.json`

### Current Limitations
- Only supports VAE-LSTM model inference
- Other models (Pure LSTM, MHA-VAE-LSTM) are not supported due to different inference processes
- Model checkpoints must be manually organized as described above

## Features

### 1. Real-time Inference
- **Continuous Data Processing**:
  - Real-time sensor data acquisition
  - Sliding window processing
  - Immediate anomaly detection

### 2. Graphical User Interface
- **Real-time Visualization**:
  - Sensor data streams
  - Anomaly detection results
  - System status indicators

### 3. Data Processing
- **Multi-modal Input**:
  - Temperature data
  - Humidity data
  - Audio data
- **Preprocessing Pipeline**:
  - Data normalization
  - Window segmentation
  - Feature extraction

## Usage

1. **Setup Model Checkpoints**:
   ```bash
   # Create directory structure
   mkdir -p checkpoint_models/VAE-LSTM
   
   # Copy your model files
   cp /path/to/your/model.pth checkpoint_models/VAE-LSTM/model_checkpoint.pth
   cp /path/to/your/config.json checkpoint_models/VAE-LSTM/model_config.json
   ```

2. **Run the Demo**:
   ```bash
   # GUI side:
   python VAE_LSTM_real_time_detection_with_ui.py

   # Hardware side (with hardware set up):
   python hardware_ypengbb/data_receive_csv.py
   ```
These two scripts should be running simultaneously. The GUI will automatically check and update the data received from the hardware side.

## Configuration

The model configuration file (`model_config.json`) should contain:
- Model architecture parameters
- Inference parameters
- Anomaly detection thresholds
- Data processing parameters

## Hardware Requirements

- **Sensors**:
  - Temperature sensor
  - Humidity sensor
  - Audio sensor
- **Processing**:
  - GPU recommended for real-time inference
  - Sufficient RAM for data buffering

## Notes

- Ensure all required sensors are properly connected
- Monitor system resources during real-time inference
- Check model checkpoint organization before running
- Verify configuration file matches the model checkpoint
- Current implementation only supports VAE-LSTM model
- Other models require different inference processes and are not supported
