
### Finish
python -u main_informer.py --model informer --data ETTm2 --root_path ./data/ETT --data_path ETTm2.csv --inverse --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2

python -u main_informer.py --model informer --data ETTm2 --root_path ./data/ETT --data_path ETTm2.csv --inverse --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2
