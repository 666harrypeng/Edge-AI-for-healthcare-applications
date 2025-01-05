# -u -> ensure that all log entries are captured immediately and are not delayed due to buffering.

# finished (2024 Nov 16)
python -u lstm_main.py --model PureLSTM --data WTH --lr 0.0005 --train_epochs 1000 --pretrained_base True \
    --pretrained_ckpt_path /home/ypengbb/Respiratory-Anomaly-Detection/LSTM_track/pure_LSTM/checkpoints/PureLSTM_WTH_ts24_hs64_ln4_in11_out1_lr0.0005_bs64_ep1000/checkpoint.pth