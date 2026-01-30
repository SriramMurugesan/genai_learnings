# LSTM Architecture

## Introduction

Long Short-Term Memory (LSTM) networks are a special type of RNN designed to solve the vanishing gradient problem and learn long-term dependencies. Introduced by Hochreiter & Schmidhuber in 1997, LSTMs have become the standard for sequence modeling.

## The Problem with Vanilla RNNs

**Vanishing gradients**: Can't remember information from many steps ago

**Example**:
```
"I grew up in France... I speak fluent ___"
```

Need to remember "France" from many words ago to predict "French"!

## LSTM Solution

**Key idea**: Separate **cell state** (long-term memory) from **hidden state** (short-term memory)

**Gates**: Control information flow
1. **Forget gate**: What to forget from cell state
2. **Input gate**: What new information to add
3. **Output gate**: What to output

## LSTM Architecture

### Components

**Cell state** (C_t): Long-term memory highway
**Hidden state** (h_t): Short-term memory / output
**Gates**: Sigmoid layers that control flow

### The Gates

#### 1. Forget Gate

**Decides**: What to forget from cell state

**Formula**:
```
f_t = σ(W_f · [h_(t-1), x_t] + b_f)
```

**Output**: 0 (forget completely) to 1 (keep completely)

**Example**:
```
"The cat, which was very cute, sat on the mat"
After "cat", might forget subject when new subject appears
```

#### 2. Input Gate

**Decides**: What new information to add to cell state

**Two parts**:
```
i_t = σ(W_i · [h_(t-1), x_t] + b_i)        # What to update
C̃_t = tanh(W_C · [h_(t-1), x_t] + b_C)    # New candidate values
```

**i_t**: Which values to update (0 to 1)
**C̃_t**: New candidate values (-1 to 1)

#### 3. Cell State Update

**Combines**: Forget old + Add new

**Formula**:
```
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t
```

**⊙**: Element-wise multiplication

**Intuition**:
- `f_t ⊙ C_(t-1)`: Forget some old information
- `i_t ⊙ C̃_t`: Add some new information

#### 4. Output Gate

**Decides**: What to output from cell state

**Formula**:
```
o_t = σ(W_o · [h_(t-1), x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

**o_t**: What parts of cell state to output
**h_t**: Final hidden state (filtered cell state)

### Complete LSTM Equations

```
f_t = σ(W_f · [h_(t-1), x_t] + b_f)        # Forget gate
i_t = σ(W_i · [h_(t-1), x_t] + b_i)        # Input gate
C̃_t = tanh(W_C · [h_(t-1), x_t] + b_C)    # Candidate cell state
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t           # Cell state update
o_t = σ(W_o · [h_(t-1), x_t] + b_o)        # Output gate
h_t = o_t ⊙ tanh(C_t)                      # Hidden state
```

### Visual Representation

```
                    C_(t-1) ────────────────→ C_t
                       │                        ↑
                       │ ×(forget)      +(add)  │
                       ↓                        │
        ┌──────────────────────────────────────┐
        │                                      │
x_t ────┤  [Forget] [Input] [Output] [Cell]  ├──→ h_t
        │    Gate     Gate    Gate    Cand.   │
h_(t-1)─┤                                      │
        └──────────────────────────────────────┘
```

## Step-by-Step Example

**Inputs**: x_1 = "The", x_2 = "cat", x_3 = "sat"

**Dimensions**:
- Input size: 50 (word embedding)
- Hidden size: 128
- Cell state size: 128 (same as hidden)

**Time step 1** (x_1 = "The"):
```
f_1 = σ(...) = [0.5, 0.3, ...]  # Forget gate
i_1 = σ(...) = [0.8, 0.6, ...]  # Input gate
C̃_1 = tanh(...) = [0.2, -0.1, ...]  # Candidate
C_1 = f_1 ⊙ C_0 + i_1 ⊙ C̃_1  # Update cell state
o_1 = σ(...) = [0.7, 0.4, ...]  # Output gate
h_1 = o_1 ⊙ tanh(C_1)  # Hidden state
```

**Time step 2** (x_2 = "cat"):
```
# Similar computation, but uses h_1 and C_1 from previous step
f_2 = σ(W_f · [h_1, x_2] + b_f)
...
```

## Why LSTMs Work

### 1. Constant Error Carousel

**Cell state**: Direct path for gradients to flow

**Gradient flow**:
```
∂C_t/∂C_(t-1) = f_t
```

If forget gate ≈ 1, gradient flows unchanged!

**No vanishing**: Gradients don't diminish exponentially

### 2. Gating Mechanism

**Learn what to remember**: Gates trained to keep relevant information

**Adaptive memory**: Different for each sequence

### 3. Additive Updates

**Cell state update**: `C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t`

**Addition** (not multiplication) prevents vanishing!

## LSTM Variants

### Peephole Connections

**Idea**: Gates can "peek" at cell state

**Modified gates**:
```
f_t = σ(W_f · [C_(t-1), h_(t-1), x_t] + b_f)
i_t = σ(W_i · [C_(t-1), h_(t-1), x_t] + b_i)
o_t = σ(W_o · [C_t, h_(t-1), x_t] + b_o)
```

**Benefit**: More information for gate decisions

### Coupled Forget and Input Gates

**Idea**: Forget and input are complementary

**Modified**:
```
f_t = σ(...)
i_t = 1 - f_t  # Coupled!
C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t
```

**Benefit**: Fewer parameters

## Bidirectional LSTM

**Idea**: Process sequence in both directions

**Architecture**:
```
Forward:  x_1 → x_2 → x_3 → ... → x_n
Backward: x_n → ... → x_3 → x_2 → x_1

Output: Concatenate forward and backward hidden states
```

**Formula**:
```
h_t = [h_forward_t ; h_backward_t]
```

**Use cases**:
- Text classification (can see full context)
- Named entity recognition
- Any task where full sequence is available

**Not for**: Real-time prediction (need future context)

## Stacked (Multi-layer) LSTM

**Idea**: Stack multiple LSTM layers

**Architecture**:
```
Input
  ↓
LSTM Layer 1
  ↓
LSTM Layer 2
  ↓
LSTM Layer 3
  ↓
Output
```

**Benefits**:
- Learn hierarchical representations
- Better performance (usually)

**Typical**: 2-4 layers

**Code**:
```python
lstm = nn.LSTM(input_size, hidden_size, num_layers=3)
```

## Training LSTMs

### Initialization

**Forget gate bias**: Initialize to 1 or 2
```python
for name, param in lstm.named_parameters():
    if 'bias' in name:
        n = param.size(0)
        param.data[n//4:n//2].fill_(1.0)  # Forget gate bias
```

**Reason**: Start by remembering everything

### Gradient Clipping

**Still needed**: Prevent exploding gradients

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### Learning Rate

**Typical**: 0.001 - 0.01

**Scheduler**: Reduce on plateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)
```

## PyTorch Implementation

### Basic LSTM

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        
        return out

# Example
model = LSTMModel(input_size=100, hidden_size=256, 
                  num_layers=2, output_size=10)
x = torch.randn(32, 50, 100)  # batch=32, seq_len=50, features=100
output = model(x)
print(output.shape)  # torch.Size([32, 10])
```

### Sequence-to-Sequence LSTM

```python
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqLSTM, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, src, tgt):
        # Encode
        _, (hidden, cell) = self.encoder(src)
        
        # Decode
        output, _ = self.decoder(tgt, (hidden, cell))
        
        # Predict
        output = self.fc(output)
        
        return output
```

## Applications

### 1. Text Generation

**Task**: Generate text character-by-character or word-by-word

**Architecture**: LSTM with softmax output

### 2. Machine Translation

**Task**: Translate sentences

**Architecture**: Encoder-Decoder LSTM

### 3. Speech Recognition

**Task**: Convert audio to text

**Architecture**: Bidirectional LSTM

### 4. Video Captioning

**Task**: Generate captions for videos

**Architecture**: CNN (frames) + LSTM (caption)

### 5. Time Series Prediction

**Task**: Forecast future values

**Architecture**: LSTM with regression output

## LSTM vs Vanilla RNN

| Aspect | Vanilla RNN | LSTM |
|--------|-------------|------|
| Memory | Short-term only | Long-term + short-term |
| Vanishing gradient | Yes | No (mostly solved) |
| Parameters | Fewer | More (4x) |
| Training time | Faster | Slower |
| Performance | Poor on long sequences | Good on long sequences |
| Use case | Simple, short sequences | Complex, long sequences |

## Summary

LSTMs:
- Solve vanishing gradient problem
- Learn long-term dependencies
- Use gates to control information flow
- Have cell state (long-term memory)

**Key Components**:
- Forget gate (what to forget)
- Input gate (what to add)
- Output gate (what to output)
- Cell state (memory highway)

**Advantages**:
- Handle long sequences
- Stable gradients
- Flexible architecture

**Disadvantages**:
- More parameters than RNN
- Slower training
- Still sequential (can't parallelize)

**Next**: [GRU Architecture](./03_gru_architecture.md)
