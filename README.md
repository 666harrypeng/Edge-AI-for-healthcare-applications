
# Edge-AI-for-healthcare-applications (ZJ01b-24)

* `Contributor`: Yiyan PENG (regulating branch ***model_design***)

This branch stores all necessary files for the whole project based on contributions of Yiyan PENG. All the models (pure-LSTM, VAE-LSTM, MHA-VAE-LSTM) & GUI have been implemented.

## Repository Structure

```
├── real_time_demo/                    # Real-time detection implementation with GUI
│   ├── VAE_LSTM_real_time_detection_with_ui.py
│   ├── VAE_LSTM_data_loader.py
│   ├── VAE_LSTM_models.py
│   └── checkpoint_models/
├── VAE-LSTM-anomaly-detection/        # VAE-LSTM model implementation
├── MHA_VAE_LSTM_anomaly_detection/    # Multi-head attention VAE-LSTM implementation
├── pure_LSTM_anomaly_detection/       # Pure LSTM model implementation
├── pure_LSTM_prediction/             # LSTM prediction model
├── hardware_ypengbb/                 # Hardware interface and sensor integration
├── Informer2020/                     # Informer model baseline implementation
```