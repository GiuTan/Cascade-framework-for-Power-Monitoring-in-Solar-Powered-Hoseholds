import os
import pandas as pd
import numpy as np

path_results = './results.csv'
model_name = 'CRNN'
experiment_name = 'solo_mse'

data = 'UKDALE'

appliance_names = ['active', 'solar', 'kettle', 'microwave', 'fridge', 'washing machine', 'dish washer']

seq_len = 600
pred_len = 600
window_stride = 480

seeds = [0,1,2,3,4,5]  # ,2


if __name__ == '__main__':

    per_app_metrics = dict()
    for appliance in appliance_names:
        per_app_metrics[f'{appliance}_mae_mean'] = []
        per_app_metrics[f'{appliance}_mae_std'] = []
        per_app_metrics[f'{appliance}_mre_mean'] = []
        per_app_metrics[f'{appliance}_mre_std'] = []

    avg_metrics = {
        'mae': [],
        'mre': []
    }

    mae_scores, mre_scores = dict(), dict()
    for appliance in appliance_names:
        mae_scores[appliance] = []
        mre_scores[appliance] = []

    for seed in seeds:
        path_logs = os.path.join('./logs', experiment_name)
        path_csv = os.path.join(path_logs, f'{data}_{model_name}_seq_len_{seq_len}_pred_len_{pred_len}_window_stride_{window_stride}_seed_{seed}')
        per_app_results_csv = os.path.join(path_csv, '50_per_app_results.csv')
        avg_results_csv = os.path.join(path_csv, 'avg_results.csv')
        df_per_app_results = pd.read_csv(per_app_results_csv)
        df_avg_results = pd.read_csv(avg_results_csv)

        for appliance in appliance_names:
            mae = df_per_app_results[appliance].values[0]
            mre = df_per_app_results[appliance].values[1]
            mae_scores[appliance].append(mae)
            mre_scores[appliance].append(mre)

        avg_metrics['mae'].append(df_avg_results[data][0])
        avg_metrics['mre'].append(df_avg_results[data][1])

    for appliance in appliance_names:
        mae_scores[appliance] = np.mean(mae_scores[appliance]), np.std(mae_scores[appliance])
        mre_scores[appliance] = np.mean(mre_scores[appliance]), np.std(mre_scores[appliance])

    for appliance in appliance_names:
        per_app_metrics[f'{appliance}_mae_mean'].append(mae_scores[appliance][0])
        per_app_metrics[f'{appliance}_mae_std'].append(mae_scores[appliance][1])
        per_app_metrics[f'{appliance}_mre_mean'].append(mre_scores[appliance][0])
        per_app_metrics[f'{appliance}_mre_std'].append(mre_scores[appliance][1])

    df_per_app_results = pd.DataFrame.from_dict(per_app_metrics)
    df_per_app_results.to_csv('./'+str(seq_len) +str(window_stride)+'_per_app_results.csv', index=False)
    print(df_per_app_results)

    print(f'MAE (avg): {np.mean(avg_metrics["mae"])}, MAE (std): {np.std(avg_metrics["mae"])}')
    print(f'MRE (avg): {np.mean(avg_metrics["mre"])}, MRE (std): {np.std(avg_metrics["mre"])}')
