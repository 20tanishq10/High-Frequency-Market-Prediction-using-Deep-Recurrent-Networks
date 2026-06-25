## Repository Structure

```text
.
├── datasets/              # Training and validation datasets
├── weights/               # Trained model checkpoints and ONNX models
├── config.py              # Centralized configuration and hyperparameters
├── dataset.py             # Dataset loading and preprocessing
├── model.py               # GRU-based neural network architecture
├── train.py               # Model training pipeline
├── solution.py            # ONNX inference pipeline for evaluation/submission
├── utils.py               # Utility functions, metrics, and helper classes
├── export.py              # Export trained models to ONNX
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

### Workflow

The project follows a straightforward end-to-end pipeline:

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Model Training
   │
   ▼
Validation
   │
   ▼
Export to ONNX
   │
   ▼
Streaming Inference
```

* **`dataset.py`** prepares sequential market data for training.
* **`model.py`** implements the GRU-based multi-task prediction model.
* **`train.py`** handles optimization, validation, and checkpointing.
* **`export.py`** converts trained PyTorch models into ONNX format.
* **`solution.py`** performs efficient streaming inference using ONNX Runtime and model ensembling.
