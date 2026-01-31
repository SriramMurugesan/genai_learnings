# RNN Fundamentals

## Introduction

Recurrent Neural Networks (RNNs) are designed for sequential data where order matters. Unlike feedforward networks, RNNs have connections that loop back, allowing them to maintain a "memory" of previous inputs.

## Why RNNs?

### Sequential Data Examples

- **Text**: "I love deep learning" (word order matters!)
- **Time series**: Stock prices, weather data
- **Speech**: Audio waveforms
- **Video**: Sequence of frames

### Problem with Standard NNs

**Fixed input size**: Can't handle variable-length sequences

**No memory**: Each input processed independently

**Example**:
```
"The cat sat on the ___"
```
Need to remember "cat" to predict "mat"!

## RNN Architecture

### Basic Structure

**Key idea**: Hidden state carries information from previous time steps

**Formula**:
```
h_t = tanh(W_hh × h_(t-1) + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

Where:
- `h_t`: Hidden state at time t
- `x_t`: Input at time t
- `y_t`: Output at time t
- `W_hh`: Hidden-to-hidden weights
- `W_xh`: Input-to-hidden weights
- `W_hy`: Hidden-to-output weights

### Unrolled View

```
Time:    t=0      t=1      t=2      t=3
Input:   x_0  →  x_1  →  x_2  →  x_3
         ↓        ↓        ↓        ↓
Hidden:  h_0  →  h_1  →  h_2  →  h_3
         ↓        ↓        ↓        ↓
Output:  y_0      y_1      y_2      y_3
```

**Same weights** shared across all time steps!

### Simple Example

```python
import numpy as np

# Parameters
hidden_size = 3
input_size = 2

# Weights (random initialization)
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
W_xh = np.random.randn(hidden_size, input_size) * 0.01
W_hy = np.random.randn(1, hidden_size) * 0.01

# Initial hidden state
h = np.zeros((hidden_size, 1))

# Sequence of inputs
inputs = [np.array([[1.0], [0.5]]),
          np.array([[0.8], [0.3]]),
          np.array([[0.6], [0.9]])]

# Process sequence
for x_t in inputs:
    # Update hidden state
    h = np.tanh(W_hh @ h + W_xh @ x_t)
    # Compute output
    y_t = W_hy @ h
    print(f"Hidden: {h.T}, Output: {y_t[0,0]:.3f}")
```

## Types of RNN Architectures

### 1. One-to-One
**Standard neural network** (not really RNN)
```
Input → Network → Output
```
**Example**: Image classification

### 2. One-to-Many
**Single input** → **Sequence output**
```
Input → [RNN] → Output_1 → Output_2 → Output_3
```
**Examples**:
- Image captioning (image → text)
- Music generation (seed → melody)

### 3. Many-to-One
**Sequence input** → **Single output**
```
Input_1 → Input_2 → Input_3 → [RNN] → Output
```
**Examples**:
- Sentiment analysis (text → positive/negative)
- Video classification (frames → category)

### 4. Many-to-Many (Synced)
**Sequence input** → **Sequence output** (same length)
```
Input_1 → Input_2 → Input_3
  ↓         ↓         ↓
Output_1   Output_2   Output_3
```
**Examples**:
- Video frame labeling
- Part-of-speech tagging

### 5. Many-to-Many (Encoder-Decoder)
**Sequence input** → **Sequence output** (different lengths)
```
Encoder:  Input_1 → Input_2 → Input_3 → [Context]
Decoder:  [Context] → Output_1 → Output_2
```
**Examples**:
- Machine translation
- Text summarization

## Backpropagation Through Time (BPTT)

### The Challenge

**Problem**: How to compute gradients through time?

**Solution**: Unroll RNN and apply standard backprop

### Algorithm

1. **Forward pass**: Compute all hidden states and outputs
2. **Backward pass**: Compute gradients from last to first time step
3. **Update weights**: Using accumulated gradients

### Gradient Flow

**Chain rule through time**:
```
∂L/∂W_hh = Σ_t ∂L_t/∂W_hh
```

Sum gradients across all time steps!

### Truncated BPTT

**Problem**: Long sequences → expensive computation

**Solution**: Limit backprop to k time steps

**Trade-off**: Faster but less accurate gradients

```python
# Instead of backprop through entire sequence
# Backprop only through last k steps (e.g., k=10)
```

## Vanishing and Exploding Gradients

### Vanishing Gradients

**Problem**: Gradients become very small

**Cause**: Repeated multiplication by small numbers
```
∂h_t/∂h_0 = ∂h_t/∂h_(t-1) × ... × ∂h_1/∂h_0
```

If each term < 1, product → 0 exponentially!

**Effect**: Can't learn long-term dependencies

**Example**:
```
Gradient after 10 steps: 0.9^10 ≈ 0.35
Gradient after 50 steps: 0.9^50 ≈ 0.005
```

**Solutions**:
1. LSTM / GRU (designed to solve this!)
2. Gradient clipping
3. Better initialization
4. ReLU activation (but causes other issues)

### Exploding Gradients

**Problem**: Gradients become very large

**Cause**: Repeated multiplication by large numbers

**Effect**: Unstable training, NaN values

**Solution**: **Gradient clipping**

```python
# Clip gradients to maximum norm
max_norm = 5.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

## Training RNNs

### Loss Function

**For sequence prediction**:
```
L = Σ_t loss(y_t, ŷ_t)
```

Sum loss across all time steps.

**Example** (language modeling):
```python
criterion = nn.CrossEntropyLoss()
total_loss = 0

for t in range(seq_length):
    output = rnn(input[t], hidden)
    loss = criterion(output, target[t])
    total_loss += loss

total_loss.backward()
```

### Initialization

**Hidden state**: Usually zeros
```python
h_0 = torch.zeros(num_layers, batch_size, hidden_size)
```

**Weights**: Xavier or orthogonal initialization
```python
for name, param in rnn.named_parameters():
    if 'weight' in name:
        nn.init.orthogonal_(param)
```

### Batch Processing

**Challenge**: Variable-length sequences

**Solutions**:
1. **Padding**: Pad to max length
2. **Packing**: Pack sequences efficiently

```python
# Padding
from torch.nn.utils.rnn import pad_sequence
padded = pad_sequence(sequences, batch_first=True)

# Packing (more efficient)
from torch.nn.utils.rnn import pack_padded_sequence
packed = pack_padded_sequence(padded, lengths, batch_first=True)
```

## Applications

### 1. Language Modeling

**Task**: Predict next word

**Example**:
```
Input:  "The cat sat on"
Output: "the"
```

**Use**: Text generation, autocomplete

### 2. Sentiment Analysis

**Task**: Classify text sentiment

**Example**:
```
Input:  "This movie is amazing!"
Output: Positive (0.95)
```

### 3. Machine Translation

**Task**: Translate text

**Example**:
```
Input:  "Hello world" (English)
Output: "Bonjour le monde" (French)
```

**Architecture**: Encoder-Decoder

### 4. Time Series Forecasting

**Task**: Predict future values

**Example**:
```
Input:  Stock prices [100, 102, 101, 103]
Output: 104 (predicted next price)
```

### 5. Speech Recognition

**Task**: Convert speech to text

**Input**: Audio waveform
**Output**: Text transcription

## Limitations of Vanilla RNNs

1. **Vanishing gradients**: Can't learn long-term dependencies
2. **Sequential processing**: Can't parallelize
3. **Limited memory**: Hidden state has limited capacity
4. **Slow training**: BPTT is expensive

**Solutions**:
- **LSTM**: Addresses vanishing gradients
- **GRU**: Simpler alternative to LSTM
- **Attention**: Focus on relevant parts
- **Transformers**: Parallel processing (replaces RNNs in many tasks)

## Practical Implementation

### PyTorch RNN

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        
        # RNN forward
        out, hidden = self.rnn(x, hidden)
        # out: (batch, seq_len, hidden_size)
        
        # Apply output layer to each time step
        out = self.fc(out)
        # out: (batch, seq_len, output_size)
        
        return out, hidden

# Example usage
model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
x = torch.randn(32, 15, 10)  # batch=32, seq_len=15, features=10
output, hidden = model(x)
print(output.shape)  # torch.Size([32, 15, 5])
```

### Character-Level Language Model

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # Embed characters
        x = self.embedding(x)
        
        # RNN
        out, hidden = self.rnn(x, hidden)
        
        # Predict next character
        out = self.fc(out)
        
        return out, hidden
```

## Summary

RNNs are:
- Designed for sequential data
- Have recurrent connections (memory)
- Share weights across time steps
- Trained with BPTT

**Key Concepts**:
- Hidden state carries information
- Different architectures for different tasks
- Vanishing/exploding gradients
- Gradient clipping essential

**Limitations**:
- Vanishing gradients
- Can't learn long-term dependencies
- Sequential processing (slow)

**Solutions**:
- LSTM (next topic!)
- GRU
- Attention mechanisms

**Next**: [LSTM Architecture](./02_lstm_architecture.md)
