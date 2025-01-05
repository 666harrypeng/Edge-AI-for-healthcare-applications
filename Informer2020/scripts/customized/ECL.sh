### TODO
python -u main_informer.py --model informer --data ECL --root_path ./data --data_path ECL.csv --inverse --features M --seq_len 168 --label_len 48 --pred_len 8 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 2 --train_epochs 10

python -u main_informer.py --model informer --data ECL --root_path ./data --data_path ECL.csv --inverse --features M --seq_len 384 --label_len 48 --pred_len 8 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 2 --train_epochs 10

python -u main_informer.py --model informer --data ECL --root_path ./data --data_path ECL.csv --inverse --features M --seq_len 600 --label_len 384 --pred_len 8 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 2 --train_epochs 10

python -u main_informer.py --model informer --data ECL --root_path ./data --data_path ECL.csv --inverse --features M --seq_len 720 --label_len 480 --pred_len 8 --e_layers 3 --d_layers 2 --attn prob --des 'Exp' --itr 2 --train_epochs 10
