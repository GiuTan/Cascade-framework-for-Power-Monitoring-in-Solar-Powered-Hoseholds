import os
import torch
import torch.nn as nn

from utils.config import get_args
from utils.random_seed import random_seed
from data.ukdale_parser import UK_Dale_Parser
from data.nilm_dataloader import NILMDataloader
from net.model import Transformer
from experiment import train, test


device = torch.device('cuda')


if __name__ == '__main__':
    ##### Args #####
    args = get_args()
    args.use_multiple_gpus = True if len(args.gpu_ids) > 1 else False
    print(args)

    random_seed(args.seed)
    print(f'Fixing random seed: {args.seed}')

    ##### Data loading #####
    ds_parser = UK_Dale_Parser(args)
    dataloader = NILMDataloader(args, ds_parser)
    train_loader, val_loader, test_loader, x_min, x_max, y_min, y_max = dataloader.get_dataloaders()

    ##### Experiment name, checkpoint, logging #####
    exp_name = f'{args.data}_{args.model_name}'
    file_name = f'{exp_name}_seq_len_{args.seq_len}_pred_len_{args.pred_len}_window_stride_{args.window_stride}_seed_{args.seed}'
    print(f'File name: {file_name}')
    path_ckpts = os.path.join(args.ckpt_path, args.experiment_name, file_name)
    path_logs = os.path.join(args.log_path, args.experiment_name, file_name)
    os.makedirs(path_ckpts, exist_ok=True)
    os.makedirs(path_logs, exist_ok=True)
    path_ckpts = os.path.join(path_ckpts, 'model.pth')

    ##### Model #####
    model = Transformer(in_size=args.in_size, pred_len=args.pred_len, out_size=args.out_size, dropout=args.dropout, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, activation=args.activation)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'pytorch_total_params: {pytorch_total_params}')
    model = model.to(device)
    if args.use_multiple_gpus:
        model = nn.DataParallel(model, args.gpu_ids)

    ##### Optimizer #####
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ##### Training #####
    y_min = torch.tensor(y_min).to(device)
    y_max = torch.tensor(y_max).to(device)
    #train(args, model, train_loader, val_loader, y_min, y_max, optimizer, path_ckpts, device)

    ##### Testing #####
    del train_loader
    del val_loader

    model = Transformer(in_size=args.in_size, pred_len=args.pred_len, out_size=args.out_size, dropout=args.dropout, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, activation=args.activation)

    model = model.to(device)
    model.load_state_dict(torch.load(path_ckpts))
    print(f'Loading best model from: {path_ckpts}')
    if args.use_multiple_gpus:
        model = nn.DataParallel(model, args.gpu_ids)
    per_app_results, avg_results = test(args, model, test_loader, y_min, y_max, device)

    ##### Saving results #####
    per_app_results.to_csv(os.path.join(path_logs, 'per_app_results.csv'), index=True)
    avg_results.to_csv(os.path.join(path_logs, 'avg_results.csv'), index=True)
    print(per_app_results)
    print(avg_results)