#!/bin/bash


python3 main.py --seed 0 --seq_len 600 --pred_len 600 --window_stride 480 --experiment_name 'solo_mse'
python3 main.py --seed 1 --seq_len 600 --pred_len 600 --window_stride 480 --experiment_name 'solo_mse'
python3 main.py --seed 2 --seq_len 600 --pred_len 600 --window_stride 480 --experiment_name 'solo_mse'
python3 main.py --seed 3 --seq_len 600 --pred_len 600 --window_stride 480 --experiment_name 'solo_mse'
python3 main.py --seed 4 --seq_len 600 --pred_len 600 --window_stride 480 --experiment_name 'solo_mse'
python3 main.py --seed 5 --seq_len 600 --pred_len 600 --window_stride 480 --experiment_name 'solo_mse'
python3 results.py
#
#python3 main.py --seed 0 --seq_len 480 --pred_len 480 --window_stride 50 --gpu_ids 5
#python3 main.py --seed 1 --seq_len 480 --pred_len 480 --window_stride 50 --gpu_ids 5
#python3 main.py --seed 2 --seq_len 480 --pred_len 480 --window_stride 50 --gpu_ids 5

#python3 main.py --seed 0 --seq_len 480 --pred_len 480 --window_stride 240 --out_size 1 --appliance_names 'solar' --experiment_name 'solar' --gpu_ids 5
