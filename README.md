# Note


spectra demo: https://colab.research.google.com/drive/1UAW1lWOlxpOAYGGNWdX44401S0uDM5K4

Dataset download: https://github.com/thuml/Time-Series-Library/tree/main

Planning to switch to TSL's dataloader (always use custom_dataset)

Now complete: image to sequence

#

Inference test run:
python main.py     --mode inference     --data_dir data/     --batch_size 32     --no_checkpoint     --prediction_length 24     --output_dir ./outputs     --log_level INFO

Training:
python main.py     --mode train  --prediction_length 96   --prediction_length 96   --log_level INFO  --batch_size 32 


(need to generate 32 random images for dummy dataset before run)

