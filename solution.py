import os
import argparse
import numpy as np
import onnxruntime as ort

from config import Config
from utils import DataPoint

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MASK_T0 = False
MASK_T1 = False


class EnsembleModel:
    def __init__(self, cfg, weights_dir):
        self.current_seq_ix = None
        self.cfg = cfg
        
        self.sessions = []
        self.n_models = 0
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        for i in range(cfg.ensemble_size):
            model_dir = os.path.join(weights_dir, f'model_{i}')
            onnx_path = os.path.join(model_dir, cfg.onnx_name)
            
            if os.path.exists(onnx_path):
                session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
                
                self.sessions.append(session)
                self.n_models += 1
                print(f'loaded model_{i}')
        
        if self.n_models == 0:
            onnx_path = os.path.join(weights_dir, cfg.onnx_name)
            
            if os.path.exists(onnx_path):
                session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
                
                self.sessions.append(session)
                self.n_models = 1
                print('loaded single model')
        
        if self.n_models == 0:
            raise RuntimeError(f'no models found in {weights_dir}')
        
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.clip_target = cfg.clip_target
        
        self.h_shape = (self.num_layers, 1, self.hidden_dim)
        self.h = np.zeros((self.n_models,) + self.h_shape, dtype=np.float32)
        
        print(f'ensemble: {self.n_models} models')
    
    def reset(self):
        self.h.fill(0)
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.reset()
        
        x = data_point.state.astype(np.float32).reshape(1, 1, -1)
        
        t0_preds = np.zeros(self.n_models, dtype=np.float32)
        t1_preds = np.zeros(self.n_models, dtype=np.float32)
        
        for i in range(self.n_models):
            outputs = self.sessions[i].run(None, {
                'x': x,
                'h': self.h[i]
            })
            
            t0_preds[i] = outputs[0][0, 0]
            t1_preds[i] = outputs[1][0, 0]
            
            self.h[i] = outputs[2]
        
        if not data_point.need_prediction:
            return None
        
        t0_final = np.clip(np.mean(t0_preds), -self.clip_target, self.clip_target)
        t1_final = np.clip(np.mean(t1_preds), -self.clip_target, self.clip_target)
        
        if MASK_T0:
            t0_final = 0.0
        if MASK_T1:
            t1_final = 0.0
        
        return np.array([t0_final, t1_final], dtype=np.float32)


class PredictionModel:
    def __init__(self, cfg, weights_dir):
        self.model = EnsembleModel(cfg, weights_dir)
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        return self.model.predict(data_point)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--weights_dir', type=str, default='./weights')
    parser.add_argument('--test_path', type=str, default='./datasets/valid.parquet')
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        cfg = Config.from_json(args.config)
        print(f'loaded config from {args.config}')
    else:
        cfg = Config()
        print('using default config')
    
    if not os.path.exists(args.test_path):
        print(f'not found: {args.test_path}')
        return
    
    print(f'testing on {args.test_path}')
    
    from utils import ScorerStepByStep
    
    model = PredictionModel(cfg, args.weights_dir)
    scorer = ScorerStepByStep(args.test_path)
    
    print('running evaluation...')
    results = scorer.score(model)
    
    print(f"mean weighted pearson: {results['weighted_pearson']:.6f}")
    for target in scorer.targets:
        print(f"  {target}: {results[target]:.6f}")


if __name__ == '__main__':
    main()
