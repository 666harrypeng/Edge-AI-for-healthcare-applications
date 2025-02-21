
# Edge-AI-for-healthcare-applications (ZJ01b-24)

* `Last Updated`: Feb 21, 2025 - 21:30
* `Contributor`: Yiyan PENG (regulating branch ***model_design***)

This branch stores all necessary files for the whole project based on contributions of Yiyan PENG.

Most recent works are all about anomaly detection - Informer-based models for prediction, Pure LSTM models for prediction, and VAE-LSTM hybrid models for anomaly detection (which is the most completed repository).

There is no pth file inside the `Informer` repository because the large memory size of the general model structure. Please contact me if the Informer’s model saved in the `pth` file format is needed.

## File Structure

### hardware_ypengbb

* `ESPNow_Basic_Master/Slave` -> Finalized Arduino code scripts for ESP Master board and Slave board, which supports all necessary digital and analog signals' collection and wireless transmission.
* `libraries` contains all necessary libraries/packages for Arduino code compilation and uploading.
* `real_records_with_anomalies` contains raw version of real-collected data in CSV files. Data visualization scripts are provided. The datasets' visualization figures are also included.
* `data_receive_csv.py` -> Utilize `pyserial` package to help Slave board receive, parse, and save transmitted data from Master board. The types of transmitted data should be pre-defined and edited accordingly.

### pure_LSTM_anomaly_detection

Use pure LSTM model to do the prediction and corresponding anomaly detection. Real-collected datasets (sensor readings in 2D and 3D) and NAB public datasets (sensor readings in 1D) all have been tested with experiment results saved in `experiments` folder. Please follow `cmd_script.sh` to use the code if needed.

### VAE-LSTM_anomaly_detection

VAE-LSTM Hybrid model for anomaly detection based on PyTorch.

### pure_LSTM_prediction

Pure LSTM related work (mainly implemented prediction)

### Informer2020

Informer related work (mainly implemented prediction)

## Key References

`Informer baseline code repo`: https://github.com/zhouhaoyi/Informer2020

`VAE-LSTM baseline code repo`: https://github.com/lin-shuyu/VAE-LSTM-for-anomaly-detection
