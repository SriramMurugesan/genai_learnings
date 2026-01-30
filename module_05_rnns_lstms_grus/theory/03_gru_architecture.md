# GRU Architecture

## Introduction

Gated Recurrent Unit (GRU) is a simplified version of LSTM, introduced by Cho et al. in 2014. GRUs achieve similar performance to LSTMs with fewer parameters and faster training.

## Motivation

**LSTM**: Powerful but complex (4 gates, cell state)

**Question**: Can we simplify while keeping performance?

**Answer**: GRU - fewer gates, no separate cell state!

## GRU vs LSTM

### Key Differences

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| States | 2 (hidden, cell) | 1 (hidden only) |
| Parameters | More | Fewer (~25% less) |
| Speed | Slower | Faster |
| Performance | Slightly better (usually) | Comparable |

## GRU Architecture

### Components

**Hidden state** (h_t): Single state (no separate cell state)
**Gates**:
1. **Reset gate** (r_t): How much past to forget
2. **Update gate** (z_t): How much to update

### The Gates

#### 1. Update Gate

**Decides**: How much of past information to keep

**Formula**:
```
z_t = σ(W_z · [h_(t-1), x_t] + b_z)
```

**Output**: 0 (ignore past) to 1 (keep past)

**Similar to**: LSTM's forget + input gates combined!

#### 2. Reset Gate

**Decides**: How much past information to forget when computing new content

**Formula**:
```
r_t = σ(W_r · [h_(t-1), x_t] + b_r)
```

**Output**: 0 (forget completely) to 1 (keep completely)

### Complete GRU Equations

```
z_t = σ(W_z · [h_(t-1), x_t] + b_z)        # Update gate
r_t = σ(W_r · [h_(t-1), x_t] + b_r)        # Reset gate
h̃_t = tanh(W_h · [r_t ⊙ h_(t-1), x_t] + b_h)  # Candidate hidden state
h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_(t-1)     # Final hidden state
```

**⊙**: Element-wise multiplication

### Step-by-Step Computation

**Step 1**: Compute update gate
```
z_t = σ(W_z · [h_(t-1), x_t] + b_z)
```
Decides how much to update.

**Step 2**: Compute reset gate
```
r_t = σ(W_r · [h_(t-1), x_t] + b_r)
```
Decides how much past to use.

**Step 3**: Compute candidate hidden state
```
h̃_t = tanh(W_h · [r_t ⊙ h_(t-1), x_t] + b_h)
```
New content based on reset past.

**Step 4**: Compute final hidden state
```
h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_(t-1)
```
Interpolate between new and old.

## Intuition

### Update Gate (z_t)

**z_t ≈ 1**: Keep old hidden state (ignore new input)
**z_t ≈ 0**: Use new candidate (forget old state)

**Example**:
```
"The cat, which was very fluffy, sat on the mat"
                  ↑ irrelevant detail
Update gate ≈ 1: Keep "cat" in memory, ignore "fluffy"
```

### Reset Gate (r_t)

**r_t ≈ 1**: Use full past hidden state
**r_t ≈ 0**: Ignore past (start fresh)

**Example**:
```
"I love Paris. The city is beautiful."
                ↑ new sentence
Reset gate ≈ 0: Start fresh for new sentence
```

### Combined Effect

**Update gate**: Controls **how much** to update
**Reset gate**: Controls **what** to use from past

## Visual Representation

```
        h_(t-1) ────────────────→ h_t
           │                       ↑
           │                       │
           │    ┌─────────────┐   │
           ├───→│ Update Gate │───┤ z_t
           │    └─────────────┘   │
           │                       │
           │    ┌─────────────┐   │
           ├───→│ Reset Gate  │───┤ r_t
           │    └─────────────┘   │
           │          ↓            │
           │    ┌─────────────┐   │
           └───→│  Candidate  │───┘ h̃_t
                └─────────────┘
                      ↑
                     x_t
```

## Detailed Example

**Inputs**: "The cat sat"

**Dimensions**:
- Input size: 50
- Hidden size: 128

**Time step 1** (x_1 = "The"):
```
# Update gate
z_1 = σ(W_z @ [h_0, x_1] + b_z)  # e.g., [0.3, 0.7, ...]

# Reset gate
r_1 = σ(W_r @ [h_0, x_1] + b_r)  # e.g., [0.8, 0.2, ...]

# Candidate (using reset past)
h̃_1 = tanh(W_h @ [r_1 ⊙ h_0, x_1] + b_h)  # e.g., [0.5, -0.3, ...]

# Final hidden state (interpolate)
h_1 = (1 - z_1) ⊙ h̃_1 + z_1 ⊙ h_0
    = [0.7, 0.3, ...] ⊙ [0.5, -0.3, ...] + [0.3, 0.7, ...] ⊙ h_0
```

## GRU vs LSTM Comparison

### Similarities

1. Both solve vanishing gradient problem
2. Both use gates to control information flow
3. Both can learn long-term dependencies
4. Both use sigmoid and tanh activations

### Differences

**LSTM**:
```
- 3 gates (forget, input, output)
- Separate cell state and hidden state
- More parameters
- More expressive (theoretically)
```

**GRU**:
```
- 2 gates (reset, update)
- Single hidden state
- Fewer parameters (~25% less)
- Simpler, faster
```

### When to Use Which?

**Use LSTM**:
- Very long sequences
- Complex patterns
- Have lots of data
- Computational resources available

**Use GRU**:
- Shorter sequences
- Limited data
- Need faster training
- Simpler patterns

**In practice**: Try both! GRU often works just as well.

## PyTorch Implementation

### Basic GRU

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), 
                        self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, hn = self.gru(x, h0)
        
        # Decode last time step
        out = self.fc(out[:, -1, :])
        
        return out

# Example
model = GRUModel(input_size=100, hidden_size=256,
                 num_layers=2, output_size=10)
x = torch.randn(32, 50, 100)  # batch, seq_len, features
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

### Bidirectional GRU

```python
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRU, self).__init__()
        
        # Bidirectional GRU
        self.gru = nn.GRU(input_size, hidden_size, 
                         bidirectional=True, batch_first=True)
        
        # Output layer (2*hidden_size because bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        # out: (batch, seq_len, 2*hidden_size)
        
        # Use last time step
        out = self.fc(out[:, -1, :])
        
        return out
```

### Custom GRU Cell

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Candidate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, h_prev):
        # Concatenate input and previous hidden
        combined = torch.cat([x, h_prev], dim=1)
        
        # Update gate
        z = torch.sigmoid(self.W_z(combined))
        
        # Reset gate
        r = torch.sigmoid(self.W_r(combined))
        
        # Candidate hidden state
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        
        # Final hidden state
        h = (1 - z) * h_tilde + z * h_prev
        
        return h
```

## Training GRUs

### Initialization

**Hidden state**: Zeros
```python
h0 = torch.zeros(num_layers, batch_size, hidden_size)
```

**Weights**: Xavier or orthogonal
```python
for name, param in gru.named_parameters():
    if 'weight' in name:
        nn.init.orthogonal_(param)
```

### Gradient Clipping

**Essential**: Prevent exploding gradients

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### Hyperparameters

**Hidden size**: 128-512 (typical)
**Num layers**: 1-3 (typical)
**Learning rate**: 0.001-0.01
**Dropout**: 0.2-0.5 (between layers)

## Applications

### 1. Sentiment Analysis

```python
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        output = self.fc(hidden[-1])
        return self.sigmoid(output)
```

### 2. Sequence Labeling

```python
class SequenceTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(SequenceTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, 
                         bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        tags = self.fc(output)
        return tags
```

### 3. Time Series Forecasting

```python
class TimeSeriesGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TimeSeriesGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        prediction = self.fc(out[:, -1, :])
        return prediction
```

## Performance Comparison

### Empirical Results

**General findings** (from various papers):
- GRU and LSTM perform similarly on most tasks
- GRU trains ~30% faster
- LSTM slightly better on very long sequences
- GRU better with limited data

**Recommendation**: Start with GRU, try LSTM if needed

## Summary

GRUs:
- Simplified version of LSTM
- 2 gates instead of 3
- Single hidden state (no cell state)
- Fewer parameters, faster training
- Similar performance to LSTM

**Key Components**:
- Update gate (how much to update)
- Reset gate (how much past to use)
- Candidate hidden state
- Final hidden state (interpolation)

**Advantages**:
- Simpler than LSTM
- Faster training
- Fewer parameters
- Good performance

**Disadvantages**:
- Slightly less expressive than LSTM
- Still sequential (can't parallelize)

**When to use**:
- Default choice for sequence modeling
- Limited computational resources
- Shorter sequences
- Try first before LSTM

**Next**: [Practical Notebooks](../notebooks/)
