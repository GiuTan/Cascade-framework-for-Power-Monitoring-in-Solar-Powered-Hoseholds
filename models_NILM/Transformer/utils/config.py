from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--data', default='UKDALE', type=str)
    parser.add_argument('--model_name', default='Transformer', type=str)
    parser.add_argument('--experiment_name', default='default', help='Name of the experiment')
    parser.add_argument('--data_path', default='../../datasets', type=str)
    parser.add_argument('--ckpt_path', default='./ckpts', type=str)
    parser.add_argument('--log_path', default='./logs', type=str)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--early_stopping', default=10, type=int)

    parser.add_argument('--in_size', default=1, type=int)
    parser.add_argument('--out_size', default=5, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--activation', type=str, default='gelu', help='Activation')

    parser.add_argument('--appliance_names', default=['kettle', 'microwave', 'fridge', 'washing machine', 'dish washer'], nargs='+', type=str)

    parser.add_argument('--seq_len', type=int, default=480)
    parser.add_argument('--pred_len', type=int, default=480)
    parser.add_argument('--window_stride', type=int, default=480)
    parser.add_argument('--validation_size', type=float, default=0.2)

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=5, type=int, help='Seed for the reproducibility of the experiment')
    parser.add_argument('--gpu_ids', default=[0], nargs='+', type=int, help='GPU ids')

    args = parser.parse_args()
    args = update_preprocessing_parameters(args)
    return args


# IF DATASET IS UKDALE 
def update_preprocessing_parameters(args):
    args.house_indicies_train = [1,5]
    args.house_indicies_test = [2]
    args.out_size = len(args.appliance_names)
    return args

# IF DATASET IS REFIT 
def update_preprocessing_parameters(args):
    args.house_indicies_train = [3,4,5,7,8,10,12,13,15,16,17,18,19]
    args.house_indicies_test = [2,9]
    #args.out_size = len(args.appliance_names)
    return args
