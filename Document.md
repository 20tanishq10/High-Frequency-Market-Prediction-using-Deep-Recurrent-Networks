# High-Frequency Market Prediction using Deep Recurrent Networks

<p align="center">

**A deep learning framework for streaming high-frequency market prediction from anonymized Limit Order Book (LOB) data using recurrent neural networks, uncertainty-aware learning, multi-task optimization, and ensemble inference.**

</p>

---

## Overview

Financial markets generate enormous streams of information every second through the continuous interaction of buyers and sellers. Capturing short-term market dynamics from this data is one of the fundamental challenges in quantitative finance and algorithmic trading.

This project presents an end-to-end deep learning pipeline for predicting future market movements directly from anonymized **Limit Order Book (LOB)** snapshots and recent trading activity.

Unlike traditional regression models that treat observations independently, this framework models the market as a **temporal sequence**, allowing the neural network to learn evolving order flow patterns, liquidity shifts, and hidden market dynamics over time.

The model is specifically designed for **streaming inference**, where observations arrive sequentially and predictions must be generated online while maintaining an internal memory of previous market states.

To improve robustness and predictive performance, the framework combines:

- GRU-based sequential modeling
- Multi-task learning
- Uncertainty estimation
- Auxiliary supervision
- Gradient balancing
- Ensemble inference
- ONNX deployment for fast production inference

The complete pipeline is optimized using **Weighted Pearson Correlation**, making it particularly suitable for financial prediction tasks where directional consistency and ranking are more important than minimizing absolute prediction error.

---

# Motivation

High-frequency financial markets exhibit several characteristics that make prediction particularly challenging:

- Extremely noisy observations
- Non-stationary data distributions
- Long temporal dependencies
- Rapid changes in liquidity
- Sparse but highly informative events
- Strong interactions between order book states and executed trades

Traditional machine learning methods often struggle to capture these temporal dependencies because they treat each observation independently.

Recurrent Neural Networks (RNNs), particularly **Gated Recurrent Units (GRUs)**, provide a natural solution by maintaining an internal hidden state that evolves with every incoming market observation.

This project explores whether combining recurrent sequence modeling with probabilistic learning and multi-task optimization can improve the prediction of future market movements in a streaming environment.

---

# Key Features

## Sequence Modeling

Instead of predicting each timestep independently, the model maintains a persistent hidden state using a multi-layer GRU, enabling it to capture temporal market dynamics across long sequences.

---

## Multi-task Learning

The network simultaneously predicts two anonymized future market indicators (`t0` and `t1`) from a shared latent representation.

Sharing representations allows both prediction tasks to benefit from common market dynamics while still learning task-specific behaviors.

---

## Uncertainty-aware Prediction

Rather than predicting only the expected target value, the model also estimates the predictive uncertainty by learning the variance of each prediction.

This probabilistic formulation provides richer supervision during training and improves representation learning.

---

## Auxiliary Supervision

An intermediate prediction head estimates both target values and their uncertainties before the final prediction layers.

These intermediate predictions are then fed back into the network as additional learned features, allowing the final prediction heads to refine their forecasts.

---

## Gradient Balancing

The two prediction tasks do not necessarily contribute equally during optimization.

A custom gradient scaling mechanism selectively reduces the influence of one task on the shared encoder, preventing negative transfer during multi-task learning.

---

## Ensemble Inference

Multiple independently trained models are exported to ONNX format and combined during inference through prediction averaging.

Ensembling reduces prediction variance and consistently improves robustness compared to individual models.

---

## Production-ready Deployment

The final trained PyTorch models are exported to **ONNX Runtime**, enabling lightweight and efficient inference without requiring a PyTorch installation.

This makes the framework suitable for real-time deployment.

---

# Highlights

- Deep Recurrent Neural Network (GRU)
- Streaming sequence inference
- Hidden-state management
- Multi-task prediction
- Weighted Pearson optimization
- Gaussian Negative Log-Likelihood
- Probabilistic forecasting
- ONNX Runtime deployment
- Ensemble prediction
- Modular PyTorch implementation

---

# Project Pipeline

```text
                Market Data
                     │
                     ▼
          Feature Engineering
                     │
                     ▼
            Sequential Dataset
                     │
                     ▼
               Multi-layer GRU
                     │
                     ▼
             Gating Mechanism
                     │
                     ▼
          Shared Latent Features
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
 Auxiliary Lag Head          Prediction Heads
        │                         │
        └────────────┬────────────┘
                     ▼
          Multi-task Optimization
                     │
                     ▼
           Ensemble of Models
                     │
                     ▼
             ONNX Runtime Export
                     │
                     ▼
          Real-time Market Prediction
```

---

# Repository Structure

*(Detailed explanation in the next section.)*

---

# Documentation

The repository documentation is organized as follows:

| Document | Description |
|----------|-------------|
| README.md | Project overview |
| docs/data.md | Dataset description |
| docs/model.md | Model architecture |
| docs/training.md | Training methodology |
| docs/inference.md | Deployment pipeline |
| docs/results.md | Experimental results |
