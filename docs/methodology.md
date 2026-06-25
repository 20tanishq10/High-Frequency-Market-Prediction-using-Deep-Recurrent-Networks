## Methodology

The proposed framework models market prediction as a **sequence learning** problem. Instead of treating each market observation independently, the model processes the incoming Limit Order Book (LOB) data sequentially, allowing it to capture temporal dependencies and evolving market dynamics.

The architecture is built around a multi-layer **Gated Recurrent Unit (GRU)** network, which maintains a hidden state across timesteps to summarize historical information. This enables the model to perform **streaming inference**, making predictions without repeatedly processing the entire sequence.

```text
                Input Features
                      │
                      ▼
                GRU Encoder
                      │
                      ▼
            Gating & Transformation
                      │
                      ▼
          Shared Latent Representation
                      │
          ┌───────────┴───────────┐
          │                       │
     Auxiliary Head         Prediction Heads
          │                       │
          └───────────┬───────────┘
                      ▼
               t0 and t1 Outputs
```

To improve learning, the network adopts a **multi-task architecture**, jointly predicting two target variables (`t0` and `t1`) from a shared representation. This allows both tasks to leverage common market patterns while learning task-specific behaviors through independent prediction heads.

An auxiliary prediction head estimates intermediate target statistics, which are concatenated back into the shared representation before the final prediction stage. This provides additional supervision and improves feature learning.

Training is driven by two complementary objectives:

* **Weighted Pearson Correlation Loss** to maximize the correlation between predictions and ground truth.
* **Gaussian Negative Log-Likelihood (NLL)** to estimate prediction uncertainty and improve model calibration.

To further stabilize optimization, gradient scaling is applied to balance the influence of the two prediction tasks on the shared encoder.

Finally, multiple independently trained models are exported to **ONNX** and combined through **ensemble averaging** during inference, resulting in more robust and stable predictions.
