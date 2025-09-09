for pred_len in 96 192 336 720; do
    python main.py --mode train --seq_len 96 --pred_len $pred_len --log_level INFO --batch_size 32 --root_path "../dataset/ETT-small" --data_path "ETTh1.csv" --cuda_num 0
done