# Backpropagation

## Introduction

Backpropagation (backward propagation of errors) is the algorithm used to train neural networks. It efficiently computes gradients of the loss function with respect to all weights in the network.

## The Learning Problem

**Goal**: Minimize loss function L(w) by adjusting weights w

**Method**: Gradient Descent
```
w_new = w_old - η * ∇L(w)
```

Where:
- `η` (eta): Learning rate
- `∇L(w)`: Gradient of loss with respect to weights

## Gradient Descent

### Intuition

Imagine you're on a mountain in fog and want to reach the valley:
- You can't see the valley
- You feel the slope under your feet
- You take steps in the steepest downward direction

### Mathematical Formulation

**Update rule**:
```
w := w - η * ∂L/∂w
```

**Example** (1D):
```python
# Function: f(x) = x²
# Derivative: f'(x) = 2x

x = 3.0  # Starting point
learning_rate = 0.1

for i in range(10):
    gradient = 2 * x
    x = x - learning_rate * gradient
    print(f"Step {i}: x = {x:.3f}, f(x) = {x**2:.3f}")

# Converges to x = 0 (minimum)
```

### Types of Gradient Descent

#### 1. Batch Gradient Descent
- Use **all** training samples
- Accurate gradient
- Slow for large datasets

```python
for epoch in range(num_epochs):
    gradient = compute_gradient(X_train, y_train, weights)
    weights = weights - learning_rate * gradient
```

#### 2. Stochastic Gradient Descent (SGD)
- Use **one** sample at a time
- Fast, noisy updates
- Can escape local minima

```python
for epoch in range(num_epochs):
    for x, y in zip(X_train, y_train):
        gradient = compute_gradient(x, y, weights)
        weights = weights - learning_rate * gradient
```

#### 3. Mini-Batch Gradient Descent
- Use **small batches** (e.g., 32, 64, 128)
- Balance between speed and accuracy
- **Most common in practice**

```python
batch_size = 32
for epoch in range(num_epochs):
    for batch_x, batch_y in get_batches(X_train, y_train, batch_size):
        gradient = compute_gradient(batch_x, batch_y, weights)
        weights = weights - learning_rate * gradient
```

## The Chain Rule

Backpropagation relies on the **chain rule** from calculus.

### Single Variable

If `y = f(g(x))`, then:
```
dy/dx = (dy/dg) * (dg/dx)
```

**Example**:
```
y = (3x + 2)²

Let g = 3x + 2, then y = g²

dy/dx = (dy/dg) * (dg/dx)
      = 2g * 3
      = 2(3x + 2) * 3
      = 6(3x + 2)
```

### Multiple Variables

For `z = f(x, y)`:
```
dz/dt = (∂z/∂x) * (dx/dt) + (∂z/∂y) * (dy/dt)
```

## Backpropagation Algorithm

### Forward Pass

Compute outputs layer by layer:

```
Layer 1: z₁ = W₁x + b₁,  a₁ = σ(z₁)
Layer 2: z₂ = W₂a₁ + b₂, a₂ = σ(z₂)
...
Output: ŷ = aₗ
Loss: L = loss(y, ŷ)
```

### Backward Pass

Compute gradients layer by layer (in reverse):

**Output layer**:
```
δₗ = ∂L/∂zₗ = (ŷ - y) ⊙ σ'(zₗ)
```

**Hidden layers** (layer l):
```
δₗ = (Wₗ₊₁ᵀ δₗ₊₁) ⊙ σ'(zₗ)
```

**Weight gradients**:
```
∂L/∂Wₗ = δₗ aₗ₋₁ᵀ
∂L/∂bₗ = δₗ
```

### Complete Example (2-layer network)

**Network**: Input (2) → Hidden (3) → Output (1)

**Forward Pass**:
```python
# Input
x = np.array([1.0, 2.0])

# Layer 1
W1 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])
z1 = W1 @ x + b1  # [0.6, 1.3, 2.0]
a1 = 1 / (1 + np.exp(-z1))  # Sigmoid

# Layer 2
W2 = np.array([[0.7, 0.8, 0.9]])
b2 = np.array([0.4])
z2 = W2 @ a1 + b2
a2 = 1 / (1 + np.exp(-z2))  # Output

# Loss (MSE)
y_true = 1.0
loss = 0.5 * (a2 - y_true)**2
```

**Backward Pass**:
```python
# Output layer gradient
dL_da2 = a2 - y_true
da2_dz2 = a2 * (1 - a2)  # Sigmoid derivative
delta2 = dL_da2 * da2_dz2

# Gradients for W2, b2
dL_dW2 = delta2 * a1
dL_db2 = delta2

# Hidden layer gradient
delta1 = (W2.T @ delta2) * (a1 * (1 - a1))

# Gradients for W1, b1
dL_dW1 = np.outer(delta1, x)
dL_db1 = delta1

# Update weights
learning_rate = 0.1
W2 -= learning_rate * dL_dW2
b2 -= learning_rate * dL_db2
W1 -= learning_rate * dL_dW1
b1 -= learning_rate * dL_db1
```

## Activation Function Derivatives

### Sigmoid
```
σ(x) = 1 / (1 + e^(-x))
σ'(x) = σ(x) * (1 - σ(x))
```

### Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

### ReLU
```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0 else 0
```

### Softmax (with Cross-Entropy)
```
Combined derivative: ŷ - y
```

## Computational Graph

Visualizing computation as a graph helps understand backpropagation.

**Example**: `L = (wx + b)²`

```
x → [×w] → [+b] → [²] → L
     ↓      ↓      ↓
     w      b      
```

**Forward**: Compute left to right
**Backward**: Compute right to left (chain rule)

## Common Challenges

### 1. Vanishing Gradients

**Problem**: Gradients become very small in deep networks

**Causes**:
- Sigmoid/Tanh activation (gradients < 1)
- Deep networks (many multiplications)

**Solutions**:
- Use ReLU activation
- Batch normalization
- Residual connections (ResNet)
- Careful initialization

**Example**:
```
Sigmoid derivative: max value = 0.25
After 10 layers: 0.25^10 ≈ 0.000001
```

### 2. Exploding Gradients

**Problem**: Gradients become very large

**Solutions**:
- Gradient clipping
- Proper initialization
- Batch normalization

```python
# Gradient clipping
max_norm = 5.0
if np.linalg.norm(gradient) > max_norm:
    gradient = gradient * (max_norm / np.linalg.norm(gradient))
```

### 3. Dying ReLU

**Problem**: ReLU neurons output 0 for all inputs

**Cause**: Large negative bias
**Solution**: Use Leaky ReLU or other variants

## Optimization Algorithms

### Momentum

Accelerates gradient descent by adding "velocity":

```python
velocity = 0.9 * velocity - learning_rate * gradient
weights = weights + velocity
```

**Benefits**:
- Faster convergence
- Reduces oscillations

### RMSprop

Adapts learning rate for each parameter:

```python
cache = 0.9 * cache + 0.1 * gradient**2
weights = weights - learning_rate * gradient / (np.sqrt(cache) + 1e-8)
```

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSprop:

```python
m = beta1 * m + (1 - beta1) * gradient  # Momentum
v = beta2 * v + (1 - beta2) * gradient**2  # RMSprop

m_hat = m / (1 - beta1**t)  # Bias correction
v_hat = v / (1 - beta2**t)

weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
```

**Default values**: β₁=0.9, β₂=0.999, ε=1e-8

**Most popular optimizer in practice!**

## Learning Rate Scheduling

### Fixed Learning Rate
```python
lr = 0.001  # Constant
```

### Step Decay
```python
lr = initial_lr * (decay_rate ** (epoch // step_size))
```

### Exponential Decay
```python
lr = initial_lr * np.exp(-decay_rate * epoch)
```

### Cosine Annealing
```python
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(epoch / total_epochs * np.pi))
```

## Training Loop

```python
def train(model, X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            y_pred = model.forward(X_batch)
            loss = compute_loss(y_pred, y_batch)
            
            # Backward pass
            gradients = model.backward(y_batch)
            
            # Update weights
            model.update_weights(gradients, lr)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## Summary

Backpropagation is:
- The algorithm for training neural networks
- Based on the chain rule from calculus
- Computes gradients efficiently
- Combined with gradient descent for optimization

**Key Concepts**:
- Forward pass: Compute predictions
- Backward pass: Compute gradients
- Chain rule: Propagate errors backward
- Gradient descent: Update weights
- Optimization algorithms: Adam, RMSprop, Momentum

**Common Issues**:
- Vanishing/exploding gradients
- Dying ReLU
- Learning rate selection

**Next**: [Activation Functions](./03_activation_functions.md)
