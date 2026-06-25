import torch
import torch.nn as nn
from torch.autograd import Function


class GradientScaler(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_gradient(x, scale):
    return GradientScaler.apply(x, scale)


class WPCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        pred_clipped = torch.clamp(pred, -6.0, 6.0)
        weights = torch.abs(target).clamp(min=self.eps)
        
        sum_w = weights.sum() + self.eps
        
        mean_true = (target * weights).sum() / sum_w
        mean_pred = (pred_clipped * weights).sum() / sum_w
        
        dev_true = target - mean_true
        dev_pred = pred_clipped - mean_pred
        
        cov = (weights * dev_true * dev_pred).sum() / sum_w
        
        var_true = (weights * dev_true ** 2).sum() / sum_w
        var_pred = (weights * dev_pred ** 2).sum() / sum_w
        
        corr = cov / (torch.sqrt(var_true + self.eps) * torch.sqrt(var_pred + self.eps))
        
        return 1 - corr


class WeightedGaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, mean, logvar, target):
        weights = torch.abs(target).clamp(min=self.eps)
        var = torch.exp(logvar).clamp(min=1e-6)
        
        nll = 0.5 * (logvar + (target - mean) ** 2 / var)
        
        sum_w = weights.sum() + self.eps
        weighted_nll = (weights * nll).sum() / sum_w
        
        return weighted_nll


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.t1_gradient_scale = getattr(cfg, 't1_gradient_scale', 1.0)
        
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True
        )
        
        self.gate = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Sigmoid()
        )
        
        self.transform = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout)
        )
        
        self.lag_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ffn_dim),
            nn.ReLU(),
            nn.Linear(cfg.ffn_dim, 4)
        )
        
        self.t0_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 4, cfg.ffn_dim),
            nn.ReLU(),
            nn.Linear(cfg.ffn_dim, 1)
        )
        
        self.t1_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 4, cfg.ffn_dim),
            nn.ReLU(),
            nn.Linear(cfg.ffn_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.gru.flatten_parameters()
    
    def forward(self, x, h=None):
        self.gru.flatten_parameters()
        
        gru_out, h_new = self.gru(x, h)
        
        g = self.gate(gru_out)
        t = self.transform(gru_out)
        gated = g * t
        
        lag_out = self.lag_head(gated)
        t0_lag_mean = lag_out[..., 0]
        t0_lag_logvar = lag_out[..., 1]
        t1_lag_mean = lag_out[..., 2]
        t1_lag_logvar = lag_out[..., 3]
        
        gated_with_lag = torch.cat([gated, lag_out], dim=-1)
        
        t0_pred = self.t0_head(gated_with_lag).squeeze(-1)
        
        if self.training and self.t1_gradient_scale < 1.0:
            gated_scaled = scale_gradient(gated_with_lag, self.t1_gradient_scale)
            t1_pred = self.t1_head(gated_scaled).squeeze(-1)
        else:
            t1_pred = self.t1_head(gated_with_lag).squeeze(-1)
        
        return t0_pred, t1_pred, t0_lag_mean, t0_lag_logvar, t1_lag_mean, t1_lag_logvar, h_new
    
    def forward_inference(self, x, h=None):
        t0_pred, t1_pred, _, _, _, _, h_new = self.forward(x, h)
        return t0_pred, t1_pred, h_new
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
