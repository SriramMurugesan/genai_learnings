# Neural Network Fundamentals

## Introduction

Neural networks are the foundation of deep learning. They are computational models inspired by biological neural networks in the brain, capable of learning complex patterns from data.

## What is a Neural Network?

A neural network is a series of algorithms that endeavors to recognize underlying relationships in data through a process that mimics the way the human brain operates.

### Key Components

1. **Neurons (Nodes)**: Basic computational units
2. **Layers**: Collections of neurons
3. **Weights**: Parameters that transform inputs
4. **Biases**: Offset parameters
5. **Activation Functions**: Non-linear transformations

## The Perceptron

### Single Perceptron

The simplest neural network unit, invented by Frank Rosenblatt in 1958.

**Mathematical Formula**:
```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
      = activation(w·x + b)
```

Where:
- `x`: Input vector
- `w`: Weight vector
- `b`: Bias
- `activation`: Activation function

**Example**:
```python
import numpy as np

# Inputs
x = np.array([1.0, 2.0, 3.0])
# Weights
w = np.array([0.5, -0.3, 0.8])
# Bias
b = 0.1

# Compute weighted sum
z = np.dot(w, x) + b  # 0.5*1 + (-0.3)*2 + 0.8*3 + 0.1 = 2.4

# Apply activation (step function)
output = 1 if z > 0 else 0  # output = 1
```

### Perceptron Limitations

- Can only learn **linearly separable** patterns
- Cannot solve XOR problem
- Limited to binary classification

## Multi-Layer Perceptron (MLP)

### Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
```

**Example 3-layer network**:
```
Input (3 neurons)
   ↓
Hidden (4 neurons)
   ↓
Output (2 neurons)
```

### Forward Propagation

Process of computing output from input:

**Layer 1 (Input → Hidden)**:
```
h = activation(W₁·x + b₁)
```

**Layer 2 (Hidden → Output)**:
```
y = activation(W₂·h + b₂)
```

**Complete Example**:
```python
import numpy as np

# Network architecture: 3 → 4 → 2
X = np.array([1.0, 2.0, 3.0])  # Input

# Layer 1: Input → Hidden
W1 = np.random.randn(4, 3)  # 4 neurons, 3 inputs each
b1 = np.random.randn(4)
z1 = W1 @ X + b1
h = np.maximum(0, z1)  # ReLU activation

# Layer 2: Hidden → Output
W2 = np.random.randn(2, 4)  # 2 neurons, 4 inputs each
b2 = np.random.randn(2)
z2 = W2 @ h + b2
y = 1 / (1 + np.exp(-z2))  # Sigmoid activation

print(f"Output: {y}")
```

## Activation Functions

### Why Activation Functions?

Without activation functions, neural networks would just be linear transformations, no matter how many layers!

**Linear composition**:
```
f(g(x)) = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = Wx + b
```

### Common Activation Functions

#### 1. Sigmoid
```
σ(x) = 1 / (1 + e^(-x))
```

**Properties**:
- Output range: (0, 1)
- Smooth gradient
- **Problem**: Vanishing gradients

**Use case**: Binary classification output layer

#### 2. Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Properties**:
- Output range: (-1, 1)
- Zero-centered
- **Problem**: Vanishing gradients

#### 3. ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```

**Properties**:
- Output range: [0, ∞)
- Simple and fast
- **Problem**: Dying ReLU (neurons can die)

**Use case**: Default choice for hidden layers

#### 4. Leaky ReLU
```
LeakyReLU(x) = max(0.01x, x)
```

**Properties**:
- Fixes dying ReLU problem
- Small gradient for negative values

#### 5. Softmax
```
softmax(x)ᵢ = e^(xᵢ) / Σⱼ e^(xⱼ)
```

**Properties**:
- Outputs sum to 1 (probability distribution)
- **Use case**: Multi-class classification output

**Example**:
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / exp_x.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(probs)  # [0.659, 0.242, 0.099]
print(probs.sum())  # 1.0
```

## Universal Approximation Theorem

**Statement**: A feedforward neural network with:
- Single hidden layer
- Finite number of neurons
- Appropriate activation function

can approximate **any continuous function** to arbitrary accuracy.

**Implications**:
- Neural networks are extremely powerful
- Depth (more layers) often better than width (more neurons per layer)
- Practical networks use multiple layers for efficiency

## Network Capacity

### Parameters Count

For a layer with `n_in` inputs and `n_out` outputs:
- Weights: `n_in × n_out`
- Biases: `n_out`
- **Total**: `n_in × n_out + n_out`

**Example Network**: 784 → 128 → 64 → 10
```
Layer 1: 784 × 128 + 128 = 100,480
Layer 2: 128 × 64 + 64 = 8,256
Layer 3: 64 × 10 + 10 = 650
Total: 109,386 parameters
```

### Overfitting and Underfitting

**Underfitting**:
- Model too simple
- High training error
- High test error

**Overfitting**:
- Model too complex
- Low training error
- High test error

**Solutions**:
- Regularization (L1, L2, Dropout)
- More training data
- Early stopping
- Data augmentation

## Training Process Overview

1. **Initialize** weights randomly
2. **Forward pass**: Compute predictions
3. **Compute loss**: Measure error
4. **Backward pass**: Compute gradients
5. **Update weights**: Gradient descent
6. **Repeat** until convergence

## Loss Functions

### Regression

**Mean Squared Error (MSE)**:
```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

**Mean Absolute Error (MAE)**:
```
MAE = (1/n) Σ |yᵢ - ŷᵢ|
```

### Classification

**Binary Cross-Entropy**:
```
BCE = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Categorical Cross-Entropy**:
```
CCE = -(1/n) Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)
```

## Practical Considerations

### Initialization

**Random Initialization**:
- Break symmetry
- Small random values

**Xavier/Glorot Initialization**:
```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))
```

**He Initialization** (for ReLU):
```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

### Batch Processing

Process multiple samples simultaneously:
```python
# Single sample: (features,)
# Batch: (batch_size, features)

X_batch = np.random.randn(32, 784)  # 32 samples
W = np.random.randn(784, 128)
output = X_batch @ W  # (32, 128)
```

### Normalization

**Input Normalization**:
```python
X_normalized = (X - X.mean()) / X.std()
```

**Batch Normalization**:
- Normalize activations within each layer
- Speeds up training
- Reduces sensitivity to initialization

## Summary

Neural networks are:
- Composed of layers of neurons
- Use activation functions for non-linearity
- Can approximate any function (Universal Approximation Theorem)
- Trained using forward and backward propagation
- Require careful initialization and normalization

**Key Concepts**:
- Perceptron → MLP
- Activation functions (ReLU, Sigmoid, Softmax)
- Forward propagation
- Loss functions
- Network capacity

**Next**: [Backpropagation](./02_backpropagation.md)
