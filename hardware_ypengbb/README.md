
# FYP Hardware (Data Collection & Visualization)

- This is the hardware developed by `Yiyan PENG`, based on the raw code files from `Chris KUO`.
- Last Update: 2025-March-16, 22:20

## Hardware Overview

The hardware is consisted of two parts:

1. ESP32 Master board
2. ESP32 Slave board

Each board should have already loaded the corresponding `ino` file using `Arduino IDE`. 

> [!NOTE]
> You can just open the `ino` file under this current directory. All the required hardware modules/libraries are included under the `hardware_ypengbb/libraries` folder.

## Preparation

- (Optional but recommended) Use `conda` to create a new environment and install the required packages. For example with Python 3.11:

```bash
conda create -n hw python=3.11
conda activate hw
```

- Core packages installation: `pyserial`

> [!WARNING]
> Please strictly follow the steps below to install the `pyserial` package. NOT the package named `serial` during installation. In the code script, the name of the package for importing is `serial`.

```bash
pip install pyserial==3.5
```

- Other packages can be installed according to actual needs.

## Data Collection

1. Generally, the PC connected with the Slave board will be the data collector (will execute the `data_receive_csv.py` script).

2. The Master board can just be powered on by any way (e.g. battery, USB, etc.). Recommended to use another PC with `Arduino IDE` and the `Serial Monitor` opened to observe the data transmission (callback output is embedded in Master's `ino` file).

3. Once two boards are powered on, execute the `data_receive_csv.py` script to start the data collection.

```bash
cd hardware_ypengbb
python data_receive_csv.py
```

> [!NOTE]
> The Slave board will use `pyserial` to receive and show the data originally displayed on the `Serial Monitor`.
>
> From now on, the `Serial Monitor` by `pyserial` will be denoted as `external Serial Monitor`. The `Serial Monitor` in `Arduino IDE` will be denoted as `Arduino Serial Monitor`.

> [!WARNING]
> Please make sure: PC connected with the Slave board should always ***close*** the `Arduino Serial Monitor` before executing the `data_receive_csv.py` script. Note: just close the `Arduino Serial Monitor` window, which should be fine. No need to close the `Arduino IDE`.

4. PLEASE take note of the starting time of the data collection (by taking a screenshot of the `external Serial Monitor`). You should be able to then figure out the "line number" of the starting time in the `sensor_data.csv` file.
5. When you try to stop breathing, please take a screenshot of the `external Serial Monitor` again. When you try to resume breathing, please take another screenshot of the `external Serial Monitor`. You should be able to then figure out the "line numbers" of the stopping time and the resuming time in the `sensor_data.csv` file. (*Note: Try multiple times of step 5 to get enough data.*)
6. When data collecttion is started, the `sensor_data.csv` file will be saved under the current directory and it will keep recording the data throughout the data collection process.
7. When data collection is finished, rename the `sensor_data.csv` file based on how you breathe (refer to the existing files' names). And then, move the data file to the `real_records_with_anomalies` folder.
8. Refer to the `data_visual.ipynb` file to visualize the data.
9. PLEASE update the ***recording start line number***, ***anomaly starting line number***, and ***anomaly ending line number*** in the `data_visual.ipynb` file.
10. Move the `csv` data file to the corresponding `datasets` folder under each model's folder for anomaly detection tasks.
