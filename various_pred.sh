for pred_len in 96 192 336 720; do
    python main.py --mode test --context_length 96 --prediction_length $pred_len --log_level INFO --batch_size 256 --data_dir "../dataset/ETT-small" --data_filename "ETTh2.csv" --cuda_num 1 --checkpoint_path "./outputs_20250902_015619_ETTh2_train_96_96/vit_timeseries/checkpoint_epoch_50.pt"
done
