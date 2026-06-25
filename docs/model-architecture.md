## Model Architecture

The proposed model combines recurrent sequence modeling with multi-task learning to capture short-term market dynamics from streaming Limit Order Book (LOB) observations.

At its core is a multi-layer **Gated Recurrent Unit (GRU)** encoder that processes one timestep at a time while maintaining an internal hidden state. This allows the model to retain historical context without repeatedly processing the full sequence.

```text
Input Features
      │
      ▼
  GRU Encoder
      │
      ▼
 Feature Gating
      │
      ▼
 Shared Representation
      │
 ┌────┴─────┐
 │          │
 ▼          ▼
Lag Head  Prediction Heads
 │          │
 └────┬─────┘
      ▼
  t0 & t1 Predictions
```

### GRU Encoder

The GRU serves as the temporal backbone of the model, learning sequential patterns in market activity while maintaining a compact memory of previous observations.

### Feature Gating

A lightweight gating network filters the GRU output, allowing the model to emphasize informative latent features and suppress noisy representations before prediction.

### Auxiliary Lag Head

An auxiliary prediction head estimates intermediate target statistics, which are concatenated with the learned representation before the final prediction stage. This provides additional supervision during training and improves feature learning.

### Multi-task Prediction

The model jointly predicts two anonymized targets (`t0` and `t1`) using separate prediction heads built on a shared latent representation. Sharing the encoder enables both tasks to learn common market dynamics while retaining task-specific outputs.

### Uncertainty Estimation

In addition to point predictions, the auxiliary head estimates predictive uncertainty through Gaussian Negative Log-Likelihood (NLL), improving calibration and providing richer training signals.

### Ensemble Inference

To improve robustness, multiple independently trained models are exported to ONNX and combined through simple averaging during inference. This ensemble strategy reduces prediction variance and improves generalization on unseen market sequences.
