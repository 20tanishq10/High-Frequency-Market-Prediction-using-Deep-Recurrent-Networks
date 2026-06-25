# High Frequency Market Prediction

## 1. Source Code & Scripts

### Requirements

This solution uses Python 3.11.13 with the following dependencies:

- torch==2.4.0
- numpy==2.2.6
- onnxruntime==1.23.2
- pyarrow==21.0.0
- argparse
- tqdm

To install dependencies:

```bash
pip install -r requirements.txt
```

### Training

The model was trained on 1x RTX 4070 Ti for a total time of 0h 13m 45s.

To start training:

```bash
python train.py --train_path ./datasets/train.parquet --valid_path ./datasets/valid.parquet --output_dir ./weights --config config.json
```

### Model Export

To export models to ONNX format:

```bash
python export.py --config config.json --weights_dir ./weights
```

## 2. Technical Report

### A. Summary

The solution is primarily straightforward, relying on proper training and minimal outlier removal. It is based on a stateful GRU model with a gated transform mechanism that predicts lags (t0_lag1, t1_lag1 - targets from the previous step) as distributions, as well as targets t0 and t1 as scalars.

### B. Solution Architecture

### Model

The model is initialized with random weights. Raw input features (32 dimensions) are fed directly into a GRU (2 layers × 128 units). The GRU also receives its hidden state from the previous step. The output from the GRU is then passed through a gated transform (specifically, a Gated Linear Unit - GLU). From the gated output, the lags t0_lag1 and t1_lag1 are predicted as Gaussian distributions using corresponding heads with 2 linear layers each. The lags are then returned.

Next, the lags are concatenated with the gated output into a single tensor. This sample is fed separately into t1_head and t0_head, each consisting of 2 linear layers separated by ReLU activation.

Final output: t0_pred, t1_pred, t0_lag_mean, t0_lag_logvar, t1_lag_mean, t1_lag_logvar, h_new

### Data Preprocessing

Input features are not preprocessed. However, for targets (clipped to [-6, 6]), a soft_winsorize operation (threshold=3.0) is applied around the mean values of t0 and t1. This helps smooth outliers and improve model generalization. This operation is primarily beneficial for t0, with minimal effect on t1. No other normalization or scaling is applied to the data.

### Training Strategy

The loss function is a weighted sum of three components:

- t0_loss - WPCLoss (competition quality metric)
- t1_loss × w1 - WPCLoss
- auxiliary_loss × w2 - Uses WeightedGaussianNLLLoss (weighted by target magnitudes) where loss is calculated on non-warmup steps (loss masking is applied)

Regularization methods (dropout and weight decay) are configurable through the config file, though dropout is disabled.

During training, AMP (automatic mixed precision) and gradient scaling were used to maintain model accuracy when transitioning to lower precision arithmetic. Gradients are clipped with max_grad_clip(10). Additionally, a custom gradient scaler is implemented for scaling gradients during backward pass for t1 to reduce overfitting.

Training is divided into 2 phases:

1. The model is initialized and training begins until patience is reached, using ModelEMA (exponential moving average of weights). In this phase:
    - Optimizer: Adam
    - Scheduler: CosineAnnealingLR
2. The best weights (EMA) are fixed, and with new weights (obtained from ModelEMA), the second phase begins using SWA (Stochastic Weight Averaging) and SWALR (everything else remains unchanged). This phase completes in less than 3 epochs and provides a modest improvement in results.

The data was used as-is (train/valid split) without shuffling or cross-validation.

### C. Key boosters (critical steps)

1. Using an ensemble of 6 similar models (differing only in initial weights) - results are obtained through weighted averaging
2. Explicitly using lags for predicting t0 and t1 improves results (provided a quality metric for predicted targets is introduced - hence the importance of predicting them as distributions to assess uncertainty)
3. Applying soft_winsorize to t0 provides a good boost for the training dataset
4. Separate training of t0 and t1 (with gradient weighting for t1 during backward pass) to significantly reduce overfitting - preventing t1 from pulling gradients and weights towards itself during training and reducing its influence on t0

### D. "What didn't work"

Data normalization attempts were unsuccessful. Despite efforts to align distributions by sign or magnitude, or to isolate certain signals from t1 by focusing on the linear signal of t1, these approaches did not succeed.

Alternative architectural solutions were not successful. Experiments with separating bid/ask price/volumes into different RNN streams, using LSTM, Transformer, Retention Net, residuals, or skip-connections did not lead to improvements. The current architecture appears close to optimal.

Extrapolating t1 values with various functions was unsuccessful. The idea was to predict lags over a sufficiently large horizon where correlation > 0.9 and extrapolate smoothed values for t1 using some sine-based function.

Binning for t0, predicting binary signals, or classification approaches were unsuccessful.

Splitting the sequence into sub-signals also failed. The idea: it was discovered that sequences could be divided into 2 classes based on whether t0 is flat or not. When examining these segments separately and training a vanilla GRU on them, results showed that for slope_t0 = 0 (t0 unchanged) t1=0.30, and for segments where slope_t0 ≠ 0 (t0 moving) t1=0.17 or higher. However, a simple baseline on the full sequence achieves around 0.14. While examining these segments separately shows that t1 is significantly better predicted, the problem lies in classifying slope_t0 - whether it changed or not. Implementing a classifier solves the task reasonably well - achieving accuracy of 94% and ROC AUC of 0.985 - but this is insufficient. If the classifier makes a mistake, the selected model is also incorrect and the prediction becomes completely wrong, significantly degrading results. An attempt to implement 3 models (flat, move, baseline), where flat/move are used only when a certain threshold is exceeded (achieving 0.996% accuracy) and baseline is used otherwise, also failed. In practice, any minor error leads to catastrophic mistakes that destroy the signal. This approach remains unviable until classifier accuracy reaches 0.9999%.
