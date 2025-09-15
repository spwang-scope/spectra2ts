# Note


spectra demo: https://colab.research.google.com/drive/1UAW1lWOlxpOAYGGNWdX44401S0uDM5K4

Dataset download/Reference: https://github.com/thuml/Time-Series-Library/tree/main

Put dataset/ under pwd

#

Command example

Training:

python main.py     --mode train  --seq_len 96   --pred_len 96   --log_level INFO  --batch_size 32 


Testing (suppose a model is saved):

python main.py     --mode test  --seq_len 96   --pred_len 96  --log_level INFO  --batch_size 32 --data_dir "../dataset/exchange_rate" --data_filename "exchange_rate.csv" --checkpoint_path "/the/result/final_model.pt"


Training script (slurm)
https://drive.google.com/file/d/1w5sb05GYAG8aoTdcW1g-SDRAFvyo81VI/view?usp=sharing


