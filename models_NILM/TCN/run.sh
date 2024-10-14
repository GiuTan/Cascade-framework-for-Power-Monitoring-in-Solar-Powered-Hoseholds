#!/bin/bash

python3 main.py --seed 0 --seq_len 600 --pred_len 600 --window_stride 480 --gpu_ids 0 --experiment_name 'NILM'
python3 main.py --seed 1 --seq_len 600 --pred_len 600 --window_stride 480 --gpu_ids 0 --experiment_name 'NILM'
python3 main.py --seed 2 --seq_len 600 --pred_len 600 --window_stride 480 --gpu_ids 0 --experiment_name 'NILM'
python3 main.py --seed 3 --seq_len 600 --pred_len 600 --window_stride 480 --gpu_ids 0 --experiment_name 'NILM'
python3 main.py --seed 4 --seq_len 600 --pred_len 600 --window_stride 480 --gpu_ids 0 --experiment_name 'NILM'
python3 main.py --seed 5 --seq_len 600 --pred_len 600 --window_stride 480 --gpu_ids 0 --experiment_name 'NILM'


