# -u -> ensure that all log entries are captured immediately and are not delayed due to buffering.

# finished (2024 Nov 13)
python -u lstm_main.py --model PureLSTM --data ETTh2 --lr 0.0001 --train_epochs 1000 

# todo
python -u lstm_main.py --model PureLSTM --data ETTh2 --lr 0.00005 --train_epochs 2000 \
    --pretrained_base True \
    --pretrained_ckpt_path /home/ypengbb/Respiratory-Anomaly-Detection/LSTM_track/pure_LSTM/checkpoints/PureLSTM_ETTh2_ts24_hs64_ln4_in6_out1_lr0.0001_bs64_ep1000/checkpoint.pth
