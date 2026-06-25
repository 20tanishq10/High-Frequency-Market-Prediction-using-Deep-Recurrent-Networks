import os
import argparse
import torch
import torch.nn as nn

from config import Config
from model import Model


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, h):
        t0_pred, t1_pred, h_new = self.model.forward_inference(x, h)
        return t0_pred, t1_pred, h_new


def export_model(model_dir, model_idx, cfg):
    checkpoint_path = os.path.join(model_dir, cfg.checkpoint_name)
    onnx_path = os.path.join(model_dir, cfg.onnx_name)
    
    print(f'exporting model {model_idx} from {model_dir}')
    
    model = Model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.eval()
    
    wrapper = ModelWrapper(model)
    wrapper.eval()
    
    x = torch.randn(1, 1, cfg.input_dim)
    h = torch.zeros(cfg.num_layers, 1, cfg.hidden_dim)
    
    torch.onnx.export(
        wrapper,
        (x, h),
        onnx_path,
        input_names=['x', 'h'],
        output_names=['t0_pred', 't1_pred', 'h_out'],
        dynamic_axes={
            'x': {0: 'batch', 1: 'seq'},
            'h': {1: 'batch'},
            't0_pred': {0: 'batch', 1: 'seq'},
            't1_pred': {0: 'batch', 1: 'seq'},
            'h_out': {1: 'batch'}
        },
        opset_version=17
    )
    
    print(f'exported to {onnx_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--weights_dir', type=str, default='./weights')
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        cfg = Config.from_json(args.config)
        print(f'loaded config from {args.config}')
    else:
        cfg = Config()
        print('using default config')
    
    for i in range(cfg.ensemble_size):
        model_dir = os.path.join(args.weights_dir, f'model_{i}')
        if os.path.exists(os.path.join(model_dir, cfg.checkpoint_name)):
            export_model(model_dir, i, cfg)
        else:
            print(f'skip model_{i}: checkpoint not found')
    
    print('export complete')


if __name__ == '__main__':
    main()
