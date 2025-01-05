# -u -> ensure that all log entries are captured immediately and are not delayed due to buffering.



# finished (2024 Nov 13)
python -u lstm_main.py --model PureLSTM --data ECL --lr 0.0002 --train_epochs 4000
python -u lstm_main.py --model PureLSTM --data ECL --lr 0.001 --train_epochs 4000 --patience 50
# best
python -u lstm_main.py --model PureLSTM --data ECL --lr 0.1 --train_epochs 4000 --patience 50 --lradj True --time_steps 36 --hidden_size 256 --num_layers 4 --batch_size 96 

X_train shape =  torch.Size([18396, 24, 320])
y_train shape =  torch.Size([18396])
X_val shape =  torch.Size([3942, 24, 320])
y_val shape =  torch.Size([3942])
X_test shape =  torch.Size([3942, 24, 320])
y_test shape =  torch.Size([3942])