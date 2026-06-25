import os
import time
import argparse
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from config import Config
from model import Model, WPCLoss, WeightedGaussianNLLLoss
from ema import ModelEMA
from utils import weighted_pearson_correlation


def soft_winsorize(y, threshold):
    result = y.copy()
    mean = np.mean(y)
    
    y_centered = y - mean
    
    mask_high = y_centered > threshold
    mask_low = y_centered < -threshold
    
    result[mask_high] = mean + threshold + np.tanh(y_centered[mask_high] - threshold)
    result[mask_low] = mean - threshold + np.tanh(y_centered[mask_low] + threshold)
    
    return result


class DatasetWrapper(Dataset):
    def __init__(self, features, t0, t1, t0_lag, t1_lag, masks):
        self.features = features
        self.t0 = t0
        self.t1 = t1
        self.t0_lag = t0_lag
        self.t1_lag = t1_lag
        self.masks = masks
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.t0[idx]),
            torch.from_numpy(self.t1[idx]),
            torch.from_numpy(self.t0_lag[idx]),
            torch.from_numpy(self.t1_lag[idx]),
            torch.from_numpy(self.masks[idx])
        )


def load_data(path, cfg):
    table = pq.read_table(path)
    
    seq_ix = table.column('seq_ix').to_numpy()
    step_in_seq = table.column('step_in_seq').to_numpy()
    
    unique_seq_ids = np.unique(seq_ix)
    n_seq = len(unique_seq_ids)
    seq_map = {sid: i for i, sid in enumerate(unique_seq_ids)}
    seq_indices = np.array([seq_map[s] for s in seq_ix], dtype=np.int32)
    
    features = np.zeros((n_seq, cfg.seq_len, cfg.input_dim), dtype=np.float32)
    t0 = np.zeros((n_seq, cfg.seq_len), dtype=np.float32)
    t1 = np.zeros((n_seq, cfg.seq_len), dtype=np.float32)
    masks = np.zeros((n_seq, cfg.seq_len), dtype=bool)
    
    all_features = np.column_stack([
        table.column(name).to_numpy().astype(np.float32)
        for name in cfg.feature_names
    ])
    
    features[seq_indices, step_in_seq] = all_features
    
    t0_raw = table.column('t0').to_numpy().astype(np.float32)
    t1_raw = table.column('t1').to_numpy().astype(np.float32)
    need_pred = table.column('need_prediction').to_numpy().astype(bool)
    
    t0_clipped = np.clip(t0_raw, -cfg.clip_target, cfg.clip_target)
    t1_clipped = np.clip(t1_raw, -cfg.clip_target, cfg.clip_target)
    
    t0_processed = soft_winsorize(t0_clipped, cfg.soft_winsorize_threshold)
    t1_processed = soft_winsorize(t1_clipped, cfg.soft_winsorize_threshold)
    
    t0[seq_indices, step_in_seq] = t0_processed
    t1[seq_indices, step_in_seq] = t1_processed
    masks[seq_indices, step_in_seq] = need_pred
    
    t0_lag = np.zeros_like(t0)
    t1_lag = np.zeros_like(t1)
    t0_lag[:, 1:] = t0[:, :-1]
    t1_lag[:, 1:] = t1[:, :-1]
    
    return features, t0, t1, t0_lag, t1_lag, masks


def compute_sign_accuracy(pred, target):
    pred_sign = (pred >= 0)
    true_sign = (target >= 0)
    return (pred_sign == true_sign).float().mean().item()


def train_epoch(model, loader, main_criterion, aux_criterion, optimizer, scaler, epoch, cfg, ema=None, swa_model=None):
    model.train()
    total_loss = 0
    total_t0_loss = 0
    total_t1_loss = 0
    total_aux_loss = 0
    n_batches = 0
    
    device_type = 'cuda' if cfg.device == 'cuda' else 'cpu'
    
    pbar = tqdm(loader, desc=f'epoch {epoch}')
    for batch in pbar:
        batch = [b.to(cfg.device) for b in batch]
        features, t0, t1, t0_lag, t1_lag, mask = batch
        
        if mask.sum() == 0:
            continue
        
        optimizer.zero_grad()
        
        with autocast(device_type):
            t0_pred, t1_pred, t0_lag_mean, t0_lag_logvar, t1_lag_mean, t1_lag_logvar, _ = model(features)
            
            t0_loss = main_criterion(t0_pred[mask], t0[mask])
            t1_loss = main_criterion(t1_pred[mask], t1[mask])
            
            aux_loss_t0 = aux_criterion(t0_lag_mean[mask], t0_lag_logvar[mask], t0_lag[mask])
            aux_loss_t1 = aux_criterion(t1_lag_mean[mask], t1_lag_logvar[mask], t1_lag[mask])
            aux_loss = aux_loss_t0 + aux_loss_t1
            
            loss = t0_loss + cfg.t1_loss_weight * t1_loss + cfg.aux_loss_weight * aux_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        if ema is not None:
            ema.update(model)
        
        if swa_model is not None:
            swa_model.update_parameters(model)
        
        total_loss += loss.item()
        total_t0_loss += t0_loss.item()
        total_t1_loss += t1_loss.item()
        total_aux_loss += aux_loss.item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            't0': f'{t0_loss.item():.4f}',
            't1': f'{t1_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / max(n_batches, 1),
        't0_loss': total_t0_loss / max(n_batches, 1),
        't1_loss': total_t1_loss / max(n_batches, 1),
        'aux': total_aux_loss / max(n_batches, 1)
    }


@torch.no_grad()
def validate(model, loader, main_criterion, aux_criterion, cfg, detailed=False):
    model.eval()
    
    total_loss = 0
    total_t0_loss = 0
    total_t1_loss = 0
    total_aux_loss = 0
    n_batches = 0
    
    all_t0_preds = []
    all_t1_preds = []
    all_t0_targets = []
    all_t1_targets = []
    all_t0_lag_mean = []
    all_t1_lag_mean = []
    all_t0_lag_target = []
    all_t1_lag_target = []
    
    for batch in loader:
        batch = [b.to(cfg.device) for b in batch]
        features, t0, t1, t0_lag, t1_lag, mask = batch
        
        if mask.sum() == 0:
            continue
        
        t0_pred, t1_pred, t0_lag_mean, t0_lag_logvar, t1_lag_mean, t1_lag_logvar, _ = model(features)
        
        t0_loss = main_criterion(t0_pred[mask], t0[mask])
        t1_loss = main_criterion(t1_pred[mask], t1[mask])
        
        aux_loss_t0 = aux_criterion(t0_lag_mean[mask], t0_lag_logvar[mask], t0_lag[mask])
        aux_loss_t1 = aux_criterion(t1_lag_mean[mask], t1_lag_logvar[mask], t1_lag[mask])
        aux_loss = aux_loss_t0 + aux_loss_t1
        
        loss = t0_loss + cfg.t1_loss_weight * t1_loss + cfg.aux_loss_weight * aux_loss
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            total_t0_loss += t0_loss.item()
            total_t1_loss += t1_loss.item()
            total_aux_loss += aux_loss.item()
            n_batches += 1
        
        all_t0_preds.append(t0_pred[mask].cpu())
        all_t1_preds.append(t1_pred[mask].cpu())
        all_t0_targets.append(t0[mask].cpu())
        all_t1_targets.append(t1[mask].cpu())
        all_t0_lag_mean.append(t0_lag_mean[mask].cpu())
        all_t1_lag_mean.append(t1_lag_mean[mask].cpu())
        all_t0_lag_target.append(t0_lag[mask].cpu())
        all_t1_lag_target.append(t1_lag[mask].cpu())
    
    if n_batches == 0:
        return None
    
    all_t0_preds = torch.cat(all_t0_preds)
    all_t1_preds = torch.cat(all_t1_preds)
    all_t0_targets = torch.cat(all_t0_targets)
    all_t1_targets = torch.cat(all_t1_targets)
    all_t0_lag_mean = torch.cat(all_t0_lag_mean)
    all_t1_lag_mean = torch.cat(all_t1_lag_mean)
    all_t0_lag_target = torch.cat(all_t0_lag_target)
    all_t1_lag_target = torch.cat(all_t1_lag_target)
    
    t0_corr = weighted_pearson_correlation(all_t0_targets.numpy(), all_t0_preds.numpy())
    t1_corr = weighted_pearson_correlation(all_t1_targets.numpy(), all_t1_preds.numpy())
    
    result = {
        'loss': total_loss / n_batches,
        't0_loss': total_t0_loss / n_batches,
        't1_loss': total_t1_loss / n_batches,
        'aux': total_aux_loss / n_batches,
        't0_corr': t0_corr,
        't1_corr': t1_corr
    }
    
    if detailed:
        result['t0_sign_acc'] = compute_sign_accuracy(all_t0_preds, all_t0_targets)
        result['t1_sign_acc'] = compute_sign_accuracy(all_t1_preds, all_t1_targets)
        
        t0_lag_corr = weighted_pearson_correlation(all_t0_lag_target.numpy(), all_t0_lag_mean.numpy())
        t1_lag_corr = weighted_pearson_correlation(all_t1_lag_target.numpy(), all_t1_lag_mean.numpy())
        result['t0_lag_corr'] = t0_lag_corr
        result['t1_lag_corr'] = t1_lag_corr
    
    return result


def print_model_metrics(metrics):
    avg_corr = (metrics['t0_corr'] + metrics['t1_corr']) / 2
    print(f'  model: loss={metrics["loss"]:.4f} (t0={metrics["t0_loss"]:.4f}, t1={metrics["t1_loss"]:.4f}, aux={metrics["aux"]:.4f})')
    print(f'         corr: t0={metrics["t0_corr"]:.4f}, t1={metrics["t1_corr"]:.4f}, avg={avg_corr:.4f}')
    print(f'         sign_acc: t0={metrics["t0_sign_acc"]:.4f}, t1={metrics["t1_sign_acc"]:.4f}')
    print(f'         lag_corr: t0={metrics["t0_lag_corr"]:.4f}, t1={metrics["t1_lag_corr"]:.4f}')


def print_ema_metrics(metrics, lr):
    avg_corr = (metrics['t0_corr'] + metrics['t1_corr']) / 2
    print(f'  ema: corr: t0={metrics["t0_corr"]:.4f}, t1={metrics["t1_corr"]:.4f}, avg={avg_corr:.4f}, lr={lr:.2e}')


def train_model(model_idx, output_dir, cfg, train_path, valid_path):
    seed = cfg.ensemble_seeds[model_idx]
    
    model_output_dir = os.path.join(output_dir, f'model_{model_idx}')
    os.makedirs(model_output_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_output_dir, cfg.checkpoint_name)
    
    device = torch.device(cfg.device)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    
    print(f'device: {device}, model_idx: {model_idx}, seed: {seed}')
    print(f'train path: {train_path}')
    print(f'valid path: {valid_path}')
    print(f't1_loss_weight: {cfg.t1_loss_weight}')
    print(f't1_gradient_scale: {cfg.t1_gradient_scale}')
    print(f'aux_loss_weight: {cfg.aux_loss_weight}')
    print(f'soft_winsorize_threshold: {cfg.soft_winsorize_threshold}')
    
    print('loading data...')
    train_features, train_t0, train_t1, train_t0_lag, train_t1_lag, train_masks = load_data(train_path, cfg)
    valid_features, valid_t0, valid_t1, valid_t0_lag, valid_t1_lag, valid_masks = load_data(valid_path, cfg)
    
    print(f'train: {len(train_features)}, valid: {len(valid_features)}')
    
    train_dataset = DatasetWrapper(train_features, train_t0, train_t1, train_t0_lag, train_t1_lag, train_masks)
    valid_dataset = DatasetWrapper(valid_features, valid_t0, valid_t1, valid_t0_lag, valid_t1_lag, valid_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    
    model = Model(cfg).to(device)
    print(f'params: {model.count_parameters():,}')
    
    batches_per_epoch = len(train_loader)
    max_epochs_estimate = cfg.early_stop_patience * 3
    total_steps_estimate = batches_per_epoch * max_epochs_estimate
    
    main_criterion = WPCLoss()
    aux_criterion = WeightedGaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs_estimate, eta_min=cfg.lr_min)
    scaler = GradScaler()
    
    ema_tau = total_steps_estimate // 4
    ema = ModelEMA(model, decay=cfg.ema_decay, tau=ema_tau)
    print(f'ema tau: {ema_tau} (batches_per_epoch={batches_per_epoch})')
    
    best_corr = -float('inf')
    no_improve = 0
    epoch = 0
    
    print('ema phase')
    while no_improve < cfg.early_stop_patience:
        epoch += 1
        train_metrics = train_epoch(model, train_loader, main_criterion, aux_criterion, 
                                    optimizer, scaler, epoch, cfg, ema=ema)
        scheduler.step()
        
        val_metrics = validate(model, valid_loader, main_criterion, aux_criterion, cfg, detailed=True)
        ema_metrics = validate(ema.ema, valid_loader, main_criterion, aux_criterion, cfg, detailed=False)
        
        if val_metrics is None or ema_metrics is None:
            continue
        
        ema_corr = (ema_metrics['t0_corr'] + ema_metrics['t1_corr']) / 2
        lr = optimizer.param_groups[0]['lr']
        
        print(f'\nepoch {epoch}: train loss={train_metrics["loss"]:.4f} (t0={train_metrics["t0_loss"]:.4f}, t1={train_metrics["t1_loss"]:.4f}, aux={train_metrics["aux"]:.4f})')
        print_model_metrics(val_metrics)
        print_ema_metrics(ema_metrics, lr)
        
        if ema_corr > best_corr:
            best_corr = ema_corr
            no_improve = 0
            torch.save(ema.ema.state_dict(), checkpoint_path)
            print(f'  >>> saved: corr={ema_corr:.4f}')
        else:
            no_improve += 1
            print(f'  no improve ({no_improve}/{cfg.early_stop_patience})')
    
    print(f'\nbest ema corr: {best_corr:.4f}')
    
    print('\nswa phase')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    
    swa_model = AveragedModel(model, avg_fn=lambda avg, new, num: 0.9 * avg + 0.1 * new)
    swa_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.swa_lr, weight_decay=cfg.weight_decay)
    swa_scheduler = SWALR(swa_optimizer, swa_lr=cfg.swa_lr)
    swa_scaler = GradScaler()
    
    swa_best_corr = best_corr
    swa_no_improve = 0
    swa_epoch = 0
    
    while swa_no_improve < cfg.swa_patience:
        swa_epoch += 1
        train_metrics = train_epoch(model, train_loader, main_criterion, aux_criterion, 
                                    swa_optimizer, swa_scaler, swa_epoch, cfg, swa_model=swa_model)
        swa_scheduler.step()
        
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        swa_module = swa_model.module
        swa_metrics = validate(swa_module, valid_loader, main_criterion, aux_criterion, cfg, detailed=True)
        
        if swa_metrics is None:
            continue
        
        swa_corr = (swa_metrics['t0_corr'] + swa_metrics['t1_corr']) / 2
        
        print(f'\nswa {swa_epoch}: train loss={train_metrics["loss"]:.4f} (t0={train_metrics["t0_loss"]:.4f}, t1={train_metrics["t1_loss"]:.4f}, aux={train_metrics["aux"]:.4f})')
        print_model_metrics(swa_metrics)
        
        if swa_corr > swa_best_corr:
            swa_best_corr = swa_corr
            swa_no_improve = 0
            torch.save(swa_module.state_dict(), checkpoint_path)
            print(f'  >>> saved: corr={swa_corr:.4f}')
        else:
            swa_no_improve += 1
            print(f'  no improve ({swa_no_improve}/{cfg.swa_patience})')
    
    print(f'\nfinal corr: {swa_best_corr:.4f}')
    print(f'saved to {checkpoint_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./datasets/train.parquet')
    parser.add_argument('--valid_path', type=str, default='./datasets/valid.parquet')
    parser.add_argument('--output_dir', type=str, default='./weights')
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        cfg = Config.from_json(args.config)
        print(f'loaded config from {args.config}')
    else:
        cfg = Config()
        print('using default config')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    for i in range(cfg.ensemble_size):
        print(f'\ntraining model {i+1}/{cfg.ensemble_size}')
        train_model(i, args.output_dir, cfg, args.train_path, args.valid_path)
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print('\ntraining complete')
    print(f'total time: {hours}h {minutes}m {seconds}s')


if __name__ == '__main__':
    main()
