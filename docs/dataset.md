# Dataset & Competition

## Problem Statement

This project was developed around a high-frequency financial forecasting challenge based on anonymized **Limit Order Book (LOB)** data.

The objective is to predict two anonymized future market movement indicators (`t0` and `t1`) from a continuous stream of market observations. Unlike conventional supervised learning tasks where each sample is independent, the dataset is organized as a collection of **time-ordered market sequences**, requiring the model to learn temporal dependencies and evolving market dynamics.

Predictions are evaluated using **Weighted Pearson Correlation**, encouraging the model to learn the direction and relative magnitude of future movements rather than minimizing absolute prediction error.

---

# Understanding the Limit Order Book (LOB)

A **Limit Order Book (LOB)** is a real-time representation of all outstanding buy and sell orders for a financial instrument.

At any instant, the order book contains two sides:

- **Bid Side** — Buy orders waiting to be executed
- **Ask Side** — Sell orders waiting to be executed

Each side consists of multiple price levels, where every level stores both:

- Price
- Available volume

A simplified order book looks like:

```
          ASK SIDE
──────────────────────────────
 Price      Volume
101.05        35
101.04        18
101.03        12
──────────────────────────────
      Current Market Price
──────────────────────────────
100.99        25
100.98        41
100.97        20
──────────────────────────────
          BID SIDE
```

The interaction between these orders determines market liquidity, price movement, and short-term trading opportunities.

Rather than using raw prices directly, the competition provides **anonymized numerical representations** derived from the underlying order book.

This preserves the statistical structure of the market while preventing participants from exploiting knowledge about specific financial instruments.

---

# Dataset Structure

The dataset is organized into **independent sequences** of market observations.

Each sequence represents the chronological evolution of a single market episode.

Every row corresponds to **one timestep**.

```
Sequence 0

Step 0
↓

Step 1
↓

Step 2

...

↓

Step 999
```

Each sequence contains **exactly 1000 timesteps**.

The neural network processes these observations sequentially while maintaining an internal hidden state that summarizes previous market information.

---

# Dataset Statistics

| Split | Number of Sequences |
|--------|--------------------:|
| Training | 10,721 |
| Validation | 1,444 |
| Public Test | ~1,500 (hidden) |
| Private Test | ~1,500 (hidden) |

The public and private leaderboard datasets are hidden throughout development, ensuring that model performance reflects genuine generalization rather than overfitting.

---

# Dataset Columns

Each observation contains three groups of information:

1. Sequence metadata
2. Market features
3. Prediction targets

---

## Sequence Metadata

| Column | Description |
|---------|-------------|
| `seq_ix` | Unique identifier of each sequence |
| `step_in_seq` | Position of the observation within the sequence (0–999) |
| `need_prediction` | Indicates whether a prediction should be generated for the next timestep |

The sequence identifier is particularly important because **hidden states must be reset whenever a new sequence begins**.

Failing to reset the recurrent memory would leak information between unrelated market episodes.

---

# Market Features

The input features are anonymized representations of different components of the Limit Order Book and recent trading activity.

Although the actual asset is hidden, the statistical relationships between features remain realistic.

The features are divided into four categories.

---

## 1. Price Features

### Bid Prices (`p0`–`p5`)

These represent multiple price levels on the **bid** side of the order book.

They capture how aggressively buyers are willing to purchase the asset.

---

### Ask Prices (`p6`–`p11`)

These correspond to multiple price levels on the **ask** side.

Together with bid prices, they describe the instantaneous market spread and available liquidity.

---

## 2. Volume Features

### Bid Volumes (`v0`–`v5`)

Volumes available at each bid level.

Large bid volumes often indicate strong buying interest.

---

### Ask Volumes (`v6`–`v11`)

Volumes available on the ask side.

These values provide information about selling pressure and liquidity.

---

## 3. Recent Trade Features

### Trade Prices (`dp0`–`dp3`)

Derived representations of recently executed transaction prices.

These features summarize the most recent market activity.

---

### Trade Volumes (`dv0`–`dv3`)

Derived representations of executed trade sizes.

They provide information about trading intensity and order flow.

---

# Prediction Targets

The dataset contains two anonymized prediction targets.

| Target | Description |
|---------|-------------|
| `t0` | Future market movement indicator |
| `t1` | Second future market movement indicator |

The exact definitions are intentionally hidden as part of the competition.

Participants must infer meaningful relationships directly from the historical order flow rather than relying on handcrafted financial assumptions.

This project approaches the problem as a **multi-task learning** problem, where both targets are predicted simultaneously using a shared temporal representation.

---

# Sequence Organization

Unlike standard tabular datasets, observations cannot be shuffled arbitrarily.

Each sequence is processed chronologically.

```
Sequence

Step0

↓

Step1

↓

Step2

↓

...

↓

Step999
```

The hidden state of the recurrent network evolves throughout the sequence, enabling the model to accumulate information about historical market behavior.

---

# Warm-up Period

The first **99 timesteps** of every sequence are considered a **warm-up period**.

During this stage:

- The model receives market observations.
- The recurrent hidden state is updated.
- No predictions are evaluated.

This allows the GRU to build an internal representation of recent market dynamics before producing scored predictions.

Beginning from **step 99**, predictions are evaluated whenever `need_prediction=True`.

This streaming setup closely resembles real-world deployment, where a forecasting system observes historical market activity before generating actionable predictions.

---

# Streaming Inference

Unlike many deep learning models that receive an entire sequence at once, this competition requires **online prediction**.

At every timestep:

1. A new market observation arrives.
2. The model updates its hidden state.
3. A prediction is generated for the next timestep.
4. The updated hidden state is carried forward.

This design makes recurrent architectures particularly attractive because they naturally maintain memory without repeatedly processing the full historical sequence.

---

# Evaluation Metric

Model performance is measured using **Weighted Pearson Correlation**.

Instead of minimizing raw prediction error, the objective is to maximize the correlation between predicted and true target values while assigning greater importance to larger market movements.

The weighting scheme encourages the model to focus on observations with higher financial significance.

Unlike Mean Squared Error (MSE), Weighted Pearson Correlation rewards:

- Correct directional prediction
- Accurate ranking of market movements
- Strong linear agreement between predictions and targets

This makes it especially suitable for financial forecasting tasks where relative performance often matters more than absolute numerical precision.

---

# Why This Problem is Challenging

High-frequency financial prediction combines several difficult machine learning challenges:

- Highly noisy observations
- Rapidly changing market dynamics
- Long temporal dependencies
- Non-stationary data distributions
- Sparse informative events
- Real-time prediction constraints
- Hidden target definitions
- Evaluation based on correlation instead of regression error

These characteristics motivate the use of recurrent neural networks, probabilistic learning, multi-task optimization, and ensemble inference—forming the core methodology presented in this project.
