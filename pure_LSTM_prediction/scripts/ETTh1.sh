# -u -> ensure that all log entries are captured immediately and are not delayed due to buffering.

python -u lstm_main.py --model PureLSTM --data ETTh1 --lr 0.00005 --train_epochs 200 --pretrained_base True --pretrained_ckpt_path /home/ypengbb/Respiratory-Anomaly-Detection/LSTM_track/pure_LSTM/checkpoints/PureLSTM_ETTh1_ts24_hs64_ln4_in6_out1_lr0.0005_bs64_ep200_pa20_rand21/checkpoint.pth


# finished (2024 Nov 13)
python -u lstm_main.py --model PureLSTM --data ETTh1 --lr 0.00025 --train_epochs 400
# best
python -u lstm_main.py --model PureLSTM --data ETTh1 --lr 0.0005 --train_epochs 200 --pretrained_base True \\
    --pretrained_ckpt_path /home/ypengbb/Respiratory-Anomaly-Detection/LSTM_track/pure_LSTM/checkpoints/PureLSTM_ETTh1_ts24_hs64_ln4_in6_out1_lr0.0005_bs64_ep200_pa20_rand21/checkpoint.pth