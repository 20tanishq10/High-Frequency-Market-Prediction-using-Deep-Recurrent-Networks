import json
import torch


class Config:
    checkpoint_name = 'model.pt'
    onnx_name = 'model.onnx'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seq_len = 1000
    warmup_steps = 99
    input_dim = 32
    clip_target = 6.0
    soft_winsorize_threshold = 3.0
    
    feature_names = [
        'p0', 'p1', 'p2', 'p3', 'p4', 'p5',
        'p6', 'p7', 'p8', 'p9', 'p10', 'p11',
        'v0', 'v1', 'v2', 'v3', 'v4', 'v5',
        'v6', 'v7', 'v8', 'v9', 'v10', 'v11',
        'dp0', 'dp1', 'dp2', 'dp3',
        'dv0', 'dv1', 'dv2', 'dv3'
    ]
    
    hidden_dim = 128
    num_layers = 2
    dropout = 0
    ffn_dim = 64
    
    batch_size = 32
    lr = 1e-3
    lr_min = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 10.0

    aux_loss_weight = 0.75
    t1_loss_weight = 0.3
    t1_gradient_scale = 0.3
    
    early_stop_patience = 7
    swa_lr = 1e-5
    swa_patience = 5
    
    use_amp = True
    ema_decay = 0.9999
    num_workers = 4
    
    ensemble_size = 6
    ensemble_seeds = [42, 123, 456, 789, 1024, 2048]
    
    @classmethod
    def from_json(cls, json_path):
        cfg = cls()
        with open(json_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
    
    def to_json(self, json_path):
        data = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                value = getattr(self, key)
                if isinstance(value, (int, float, str, bool, list)):
                    data[key] = value
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)


cfg = Config()
