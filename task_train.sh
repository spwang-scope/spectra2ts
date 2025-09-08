for pred_len in 96 192 336 720; do
    python main.py --mode train --seq_len 96 --pred_len $pred_len --log_level INFO --batch_size 32 --root_path "../dataset/ETT-small" --data_path "ETTh2.csv" --cuda_num 2
done

for pred_len in 96 192 336 720; do
    python main.py --mode test --seq_len 96 --pred_len $pred_len --log_level INFO --batch_size 64 --root_path "../dataset/ETT-small" --data_path "ETTh2.csv" --cuda_num 2
done