### TODO
python -u main_informer.py --model informer --data WTH --root_path ./data --data_path WTH.csv --inverse --features M --seq_len 96 --label_len 48 --pred_len 24 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --train_epochs 10 --itr 2

python -u main_informer.py --model informer --data WTH --root_path ./data --data_path WTH.csv --inverse --features M --seq_len 168 --label_len 96 --pred_len 24 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --train_epochs 10 --itr 2

python -u main_informer.py --model informer --data WTH --root_path ./data --data_path WTH.csv --inverse --features M --seq_len 336 --label_len 168 --pred_len 24 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --train_epochs 10 --itr 2