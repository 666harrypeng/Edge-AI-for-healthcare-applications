'''
# main_informer.py default settings
python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2
######
'''

### finish
python -u main_informer.py --model informer --data ETTm1 --root_path ./data/ETT --data_path ETTm1.csv --inverse --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2

python -u main_informer.py --model informer --data ETTm1 --root_path ./data/ETT --data_path ETTm1.csv --inverse --features M --seq_len 384 --label_len 384 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2
