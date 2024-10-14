import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import get_results_summary


def train_epoch(args, model, train_dataloader, val_dataloader, data_min, data_max, optimizer, epoch, device):
    ##### Training #####
    model.train()
    train_loss_mse_solar = []
    train_loss_mse_nilm = []
    train_loss_reg = []
    train_loss = []

    for inputs, targets in tqdm(train_dataloader, total=len(train_dataloader), leave=False):
        inputs, targets_1, targets_2 = inputs.to(device), targets[0].to(device),   targets[1].to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss_reg = F.mse_loss(logits[2], targets_2[:,:,:1])
        loss_mse_solar = F.mse_loss(logits[0], targets_1)
        loss_mse_nilm = F.mse_loss(logits[1], targets_2)
        magnitude_factor = 0.1
        
        loss_total = loss_mse_solar * magnitude_factor + loss_mse_nilm + loss_reg* 0.001
        loss_total.backward()
        optimizer.step()
        train_loss_mse_nilm.append(loss_mse_nilm.detach())
        train_loss_mse_solar.append(loss_mse_solar.detach())
        train_loss_reg.append(loss_reg.detach())
        train_loss.append(loss_total.detach())

    ##### Validation #####
    model.eval()
    val_mae_solar = []
    val_mae_nilm = []
    val_mre_solar = []
    val_mre_nilm = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_dataloader, total=len(val_dataloader), leave=False):
            inputs, targets_3, targets_4 = inputs.to(device), targets[0].to(device), targets[1].to(device)
            #print(targets_3.shape)
            #print(targets_4.shape)
            logits = model(inputs)
            #print(logits.shape)
            mae_score_solar = F.l1_loss(logits[0], targets_3)
            mae_score_nilm = F.l1_loss(logits[1], targets_4)
            mre_score_solar = mean_relative_error(logits[0], targets_3)
            mre_score_nilm = mean_relative_error(logits[1], targets_4)

            val_mae_solar.append(mae_score_solar.detach())
            val_mae_nilm.append(mae_score_nilm.detach())
            val_mre_solar.append(mre_score_solar.detach())
            val_mre_nilm.append(mre_score_nilm.detach())

    train_loss_mse_nilm = torch.stack(train_loss_mse_nilm).mean().item()
    train_loss_mse_solar = torch.stack(train_loss_mse_solar).mean().item()
    train_loss_reg = torch.stack(train_loss_reg).mean().item()
    train_loss = torch.stack(train_loss).mean().item()
    val_mae_solar = torch.stack(val_mae_solar).mean().item()
    val_mae_nilm = torch.stack(val_mae_nilm).mean().item()
    val_mre_solar = torch.stack(val_mre_solar).mean().item()
    val_mre_nilm = torch.stack(val_mre_nilm).mean().item()

    ##### Metrics #####
    print(
        f'Epoch: {epoch+1}/{args.n_epochs}',
        'Train Loss: %.4f' % train_loss,
        'Train MSE SOLAR: %.4f' % train_loss_mse_solar,
        'Train MSE NILM: %.4f' % train_loss_mse_nilm,
        'Train Loss REG: %.4f' % train_loss_reg,
        'Validation MAE SOLAR: %.4f' % val_mae_solar,
        'Validation MRE SOLAR: %.4f' % val_mre_solar,
        'Validation MAE NILM: %.4f' % val_mae_nilm,
        'Validation MRE NILM: %.4f' % val_mre_nilm
    )

    return [val_mae_solar,val_mae_nilm], [val_mre_solar,val_mae_nilm]


def train(args, model, train_dataloader, val_dataloader, data_min, data_max, optimizer, path_ckpts, device):
    best_mae_score = np.inf
    best_mre_score = np.inf
    print('Training full model ...')

    n_epochs_without_improvements = 0
    for epoch in tqdm(range(args.n_epochs), total=args.n_epochs, leave=False):

        mae_score, mre_score = train_epoch(args, model, train_dataloader, val_dataloader, data_min, data_max, optimizer, epoch, device)
        MEAN_mae_score = (mae_score[0]*0.1 + mae_score[1]) / 2
        MEAN_mre_score = (mre_score[0]*0.1 + mre_score[1]) / 2
        if MEAN_mae_score < best_mae_score:
            best_mae_score = MEAN_mae_score
            best_mre_score = MEAN_mre_score
            print('New best epoch: {}/{}, Validation MAE: {:.4f}, Validation MRE: {:.4f}.'.format(epoch+1, args.n_epochs, best_mae_score, best_mre_score))
            if args.use_multiple_gpus:
                torch.save(model.module.state_dict(), path_ckpts)
            else:
                torch.save(model.state_dict(), path_ckpts)
            print('Model saved to: {}'.format(path_ckpts))
            n_epochs_without_improvements = 0
        else:
            n_epochs_without_improvements += 1
        if n_epochs_without_improvements == args.early_stopping:
            print(f'No improvements for {args.early_stopping} epochs: end training')
            break

    return


def test(args, model, test_dataloader, data_min, data_max, device):
    ##### Testing #####
    model.eval()
    power_pred_solar = []
    power_pred_nilm = []
    power_true_solar = []
    power_true_nilm = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader, total=len(test_dataloader), leave=False):
            inputs, targets_0, targets_1 = inputs.to(device), targets[0].to(device), targets[1].to(device)
            logits = model(inputs)
            targets_0 = denormalize(targets_0, torch.from_numpy(data_min[0]).to(device), torch.from_numpy(data_max[0]).to(device))
            targets_1 = denormalize(targets_1,  torch.from_numpy(data_min[1]).to(device), torch.from_numpy(data_max[1]).to(device))
            logits_0 = denormalize(logits[0],  torch.from_numpy(data_min[0]).to(device), torch.from_numpy(data_max[0]).to(device))
            logits_1 = denormalize(logits[1],  torch.from_numpy(data_min[1]).to(device), torch.from_numpy(data_max[1]).to(device))
            power_pred_solar.append(logits_0)
            power_pred_nilm.append(logits_1)
            power_true_solar.append(targets_0)
            power_true_nilm.append(targets_1)


        power_true_solar = torch.cat(power_true_solar, dim=0).cpu().numpy()
        power_true_nilm = torch.cat(power_true_nilm, dim=0).cpu().numpy()

        power_pred_solar = torch.cat(power_pred_solar, dim=0).cpu().numpy()
        power_pred_solar = np.where(power_pred_solar < 0, 0, power_pred_solar)

        power_pred_nilm = torch.cat(power_pred_nilm, dim=0).cpu().numpy()
        power_pred_nilm = np.where(power_pred_nilm < 0, 0, power_pred_nilm)

        np.save('pred_solar.npy', power_pred_solar)
        np.save('true_solar.npy', power_true_solar)
        np.save('pred_nilm.npy', power_pred_nilm)
        np.save('true_nilm.npy', power_true_nilm)
        per_app_results_solar, avg_results_solar = get_results_summary(power_true_solar, power_pred_solar, args.appliance_names, args.data)
        per_app_results_nilm, avg_results_nilm = get_results_summary(power_true_nilm, power_pred_nilm, args.appliance_names, args.data)

        return [per_app_results_solar, avg_results_solar], [per_app_results_nilm,avg_results_nilm]


def mean_relative_error(pred, label, eps=1e-9):
    temp = torch.full_like(label, eps)
    maximum, _ = torch.max(torch.stack([label, pred, temp], dim=-1), dim=-1)
    return torch.mean(torch.nan_to_num(torch.abs(label - pred) / maximum))


def denormalize(data, data_min, data_max):
    #print(data_min)
    return (data_max - data_min) * data + data_min
