# Activation Functions

## Introduction

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without activation functions, a neural network would be just a linear regression model, regardless of depth.

## Why Non-Linearity?

**Linear composition is still linear**:
```
f(g(x)) = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
```

No matter how many layers, without activation functions, the network reduces to a single linear transformation!

## Common Activation Functions

### 1. Sigmoid (Logistic)

**Formula**:
```
σ(x) = 1 / (1 + e^(-x))
```

**Range**: (0, 1)

**Derivative**:
```
σ'(x) = σ(x) × (1 - σ(x))
```

**Pros**:
- Smooth gradient
- Output interpretable as probability
- Historically important

**Cons**:
- Vanishing gradient problem
- Not zero-centered
- Computationally expensive (exp)

**Use Cases**:
- Binary classification output layer
- Gate mechanisms (LSTM)

**Code**:
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

### 2. Tanh (Hyperbolic Tangent)

**Formula**:
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
       = 2σ(2x) - 1
```

**Range**: (-1, 1)

**Derivative**:
```
tanh'(x) = 1 - tanh²(x)
```

**Pros**:
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid

**Cons**:
- Still suffers from vanishing gradients
- Computationally expensive

**Use Cases**:
- Hidden layers (before ReLU became popular)
- RNN/LSTM cells

**Code**:
```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

### 3. ReLU (Rectified Linear Unit)

**Formula**:
```
ReLU(x) = max(0, x) = {x if x > 0, 0 otherwise}
```

**Range**: [0, ∞)

**Derivative**:
```
ReLU'(x) = {1 if x > 0, 0 otherwise}
```

**Pros**:
- Simple and fast
- No vanishing gradient for positive values
- Sparse activation (many zeros)
- Biologically plausible

**Cons**:
- Dying ReLU problem
- Not zero-centered
- Unbounded output

**Use Cases**:
- **Default choice for hidden layers**
- Most CNNs and deep networks

**Code**:
```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

### 4. Leaky ReLU

**Formula**:
```
LeakyReLU(x) = {x if x > 0, αx otherwise}
```
where α is a small constant (e.g., 0.01)

**Range**: (-∞, ∞)

**Derivative**:
```
LeakyReLU'(x) = {1 if x > 0, α otherwise}
```

**Pros**:
- Fixes dying ReLU problem
- Small gradient for negative values

**Cons**:
- Inconsistent performance
- Extra hyperparameter (α)

**Code**:
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

### 5. ELU (Exponential Linear Unit)

**Formula**:
```
ELU(x) = {x if x > 0, α(e^x - 1) otherwise}
```

**Range**: (-α, ∞)

**Pros**:
- Smooth everywhere
- Negative saturation pushes mean activation closer to zero
- Reduces bias shift

**Cons**:
- Computationally expensive (exp)
- Extra hyperparameter

**Code**:
```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

### 6. GELU (Gaussian Error Linear Unit)

**Formula**:
```
GELU(x) = x × Φ(x)
```
where Φ(x) is the cumulative distribution function of the standard normal distribution

**Approximation**:
```
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Pros**:
- Smooth, non-monotonic
- Used in BERT, GPT
- State-of-the-art in transformers

**Code**:
```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

### 7. Swish / SiLU

**Formula**:
```
Swish(x) = x × σ(x) = x / (1 + e^(-x))
```

**Pros**:
- Smooth, non-monotonic
- Self-gated
- Better than ReLU in some deep networks

**Code**:
```python
def swish(x):
    return x * sigmoid(x)
```

### 8. Softmax

**Formula**:
```
softmax(x)ᵢ = e^(xᵢ) / Σⱼ e^(xⱼ)
```

**Range**: (0, 1) with Σ softmax(x)ᵢ = 1

**Pros**:
- Outputs probability distribution
- Differentiable

**Use Cases**:
- **Multi-class classification output layer**

**Code**:
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / exp_x.sum(axis=-1, keepdims=True)
```

## Comparison Table

| Function | Range | Zero-Centered | Monotonic | Computational Cost | Dying Neurons |
|----------|-------|---------------|-----------|-------------------|---------------|
| Sigmoid | (0, 1) | No | Yes | High | No |
| Tanh | (-1, 1) | Yes | Yes | High | No |
| ReLU | [0, ∞) | No | Yes | Low | Yes |
| Leaky ReLU | (-∞, ∞) | No | Yes | Low | No |
| ELU | (-α, ∞) | ~Yes | Yes | Medium | No |
| GELU | (-∞, ∞) | No | No | High | No |
| Swish | (-∞, ∞) | No | No | Medium | No |
| Softmax | (0, 1) | No | No | High | No |

## Choosing Activation Functions

### Hidden Layers

**Default choice**: **ReLU**
- Fast, simple, works well in practice
- Good starting point

**Alternatives**:
- **Leaky ReLU / ELU**: If dying ReLU is a problem
- **GELU / Swish**: For transformers and very deep networks
- **Tanh**: For RNNs/LSTMs

### Output Layer

**Regression**: 
- Linear (no activation) or ReLU (if output ≥ 0)

**Binary Classification**:
- **Sigmoid** (outputs probability)

**Multi-class Classification**:
- **Softmax** (outputs probability distribution)

## Dying ReLU Problem

**Problem**: Neurons can "die" during training

**Cause**: 
- Large negative bias → always outputs 0
- No gradient → no learning

**Example**:
```python
# Neuron with large negative bias
x = np.array([1, 2, 3])
w = np.array([-1, -1, -1])
b = -10

z = np.dot(w, x) + b  # -16
output = max(0, z)  # 0
gradient = 0 if z <= 0 else 1  # 0 → no learning!
```

**Solutions**:
1. Use Leaky ReLU
2. Lower learning rate
3. Proper initialization
4. Batch normalization

## Vanishing Gradient Problem

**Problem**: Gradients become very small in deep networks

**Cause**: 
- Sigmoid/Tanh saturate (gradient → 0)
- Many layers multiply small gradients

**Example**:
```
Sigmoid max gradient = 0.25
10 layers: 0.25^10 ≈ 0.000001
```

**Solutions**:
1. Use ReLU (gradient = 1 for x > 0)
2. Batch normalization
3. Residual connections (ResNet)
4. Proper initialization

## Practical Tips

1. **Start with ReLU** for hidden layers
2. **Use Softmax** for multi-class classification output
3. **Use Sigmoid** for binary classification output
4. **Try Leaky ReLU** if ReLU doesn't work
5. **Use GELU** for transformers
6. **Avoid Sigmoid/Tanh** in hidden layers of deep networks
7. **Combine with Batch Normalization** for better training

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(14, 10))

# Plot activation functions
activations = {
    'Sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'Tanh': np.tanh,
    'ReLU': lambda x: np.maximum(0, x),
    'Leaky ReLU': lambda x: np.where(x > 0, x, 0.01 * x),
    'ELU': lambda x: np.where(x > 0, x, np.exp(x) - 1),
    'GELU': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
}

for i, (name, func) in enumerate(activations.items(), 1):
    plt.subplot(2, 3, i)
    plt.plot(x, func(x), linewidth=2)
    plt.title(name, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

## Summary

Activation functions:
- Introduce non-linearity
- Enable learning complex patterns
- Different functions for different use cases
- ReLU is the default for hidden layers
- Softmax/Sigmoid for output layers

**Key Takeaways**:
- **ReLU**: Default choice, fast, effective
- **Leaky ReLU**: Fixes dying ReLU
- **GELU/Swish**: State-of-the-art for transformers
- **Softmax**: Multi-class classification
- **Sigmoid**: Binary classification

**Next**: [Practical Notebooks](../notebooks/)
