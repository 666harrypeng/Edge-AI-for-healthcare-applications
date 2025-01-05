
# Edge-AI-for-healthcare-applications (ZJ01b-24)

* `Last Updated`: Jan 06, 2025 - 01:40
* `Contributor`: Yiyan PENG (regulating branch ***model_design***)

This branch stores all necessary files for the whole project based on contributions of Yiyan PENG.

Most recent works are all about anomaly detection - Informer-based models for prediction, Pure LSTM models for prediction, and VAE-LSTM hybrid models for anomaly detection (which is the most completed repository). 

There is no pth file inside the Informer repository because the large memory size of the general model structure. Please contact me if the Informer’s model saved in the pth file format is needed.

## File Structure

* `esp_now_communication`: ESP-NOW communication base files; Data grasper and converter (from Serial Monitor to CSV files)
* `Informer2020`: Informer related work (mainly implemented prediction)
* `pure_LSTM_prediction`: Pure LSTM related work (mainly implemented prediction)
* `VAE-LSTM_anomaly_detection`: VAE-LSTM Hybrid model for anomaly detection based on PyTorch.

## Key References

`Informer baseline code repo`: https://github.com/zhouhaoyi/Informer2020

`VAE-LSTM baseline code repo`: https://github.com/lin-shuyu/VAE-LSTM-for-anomaly-detection
