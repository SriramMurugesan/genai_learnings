# Linear Algebra for AI

## Introduction

Linear algebra is the mathematics of data. In AI and machine learning, almost everything is represented as vectors and matrices, making linear algebra absolutely essential.

## Why Linear Algebra Matters in AI

- **Data Representation**: Images, text, audio all represented as vectors/matrices
- **Neural Networks**: Weights, activations, gradients are all matrices
- **Transformations**: Linear algebra describes how data transforms through layers
- **Efficiency**: Matrix operations are highly optimized (GPUs excel at this)

## Vectors

### Definition

A vector is an ordered list of numbers.

**Example**:
```
v = [1, 2, 3]  # 3-dimensional vector
```

**In AI**:
- Feature vector: [age, height, weight]
- Word embedding: 300-dimensional vector
- Image pixel: [R, G, B]

### Vector Operations

**Addition**:
```
[1, 2] + [3, 4] = [4, 6]
```

**Scalar Multiplication**:
```
2 * [1, 2] = [2, 4]
```

**Dot Product**:
```
[1, 2] · [3, 4] = 1*3 + 2*4 = 11
```

**Magnitude (Length)**:
```
||v|| = √(v₁² + v₂² + ... + vₙ²)
||[3, 4]|| = √(9 + 16) = 5
```

### Vector Properties

**Unit Vector**: Vector with magnitude 1
```
v̂ = v / ||v||
```

**Orthogonal Vectors**: Dot product = 0
```
[1, 0] · [0, 1] = 0  # Perpendicular
```

## Matrices

### Definition

A matrix is a 2D array of numbers.

**Example**:
```
A = [[1, 2, 3],
     [4, 5, 6]]  # 2×3 matrix
```

**In AI**:
- Image: Height × Width × Channels
- Weight matrix in neural network
- Batch of data: Batch Size × Features

### Matrix Operations

**Addition**:
```
[[1, 2]] + [[3, 4]] = [[4, 6]]
```

**Scalar Multiplication**:
```
2 * [[1, 2],     [[2, 4],
     [3, 4]]  =   [6, 8]]
```

**Matrix Multiplication**:
```
A (m×n) × B (n×p) = C (m×p)

C[i,j] = Σ A[i,k] * B[k,j]

Example:
[[1, 2],  ×  [[5, 6],  =  [[19, 22],
 [3, 4]]      [7, 8]]      [43, 50]]
```

**Transpose**:
```
A^T: Flip rows and columns

[[1, 2, 3],^T    [[1, 4],
 [4, 5, 6]]   =   [2, 5],
                  [3, 6]]
```

### Special Matrices

**Identity Matrix (I)**:
```
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

A × I = A
```

**Zero Matrix**:
```
All elements are 0
```

**Diagonal Matrix**:
```
Non-zero only on diagonal
```

## Matrix Multiplication in Neural Networks

**Forward Pass**:
```
Input (batch_size × input_dim)
Weights (input_dim × hidden_dim)
Output = Input × Weights

Example:
[1, 2, 3] × [[0.1, 0.2],  = [2.2, 2.8]
            [0.3, 0.4],
            [0.5, 0.6]]
```

**Why it works**:
- Each neuron computes weighted sum of inputs
- Matrix multiplication does this for all neurons simultaneously
- Highly efficient on GPUs

## Systems of Linear Equations

**Form**:
```
2x + 3y = 8
4x + 5y = 14

Matrix form: Ax = b
[[2, 3],  [x]   [8]
 [4, 5]]  [y] = [14]
```

**Solution**: x = A⁻¹b (if A is invertible)

**In ML**: Solving for optimal parameters

## Matrix Inverse

**Definition**: A⁻¹ such that A × A⁻¹ = I

**Properties**:
- Only square matrices can have inverses
- Not all square matrices are invertible
- (AB)⁻¹ = B⁻¹A⁻¹

**Computing Inverse**:
- Gaussian elimination
- LU decomposition
- In practice: Use library functions

## Determinant

**Definition**: Scalar value that encodes properties of matrix

**2×2 Matrix**:
```
det([[a, b],  = ad - bc
     [c, d]])
```

**Properties**:
- det(A) = 0 ⟺ A is not invertible
- det(AB) = det(A) × det(B)
- det(A^T) = det(A)

**Geometric Interpretation**: Volume scaling factor

## Eigenvalues and Eigenvectors

**Definition**:
```
Av = λv

v: eigenvector
λ: eigenvalue
```

**Meaning**: Direction that only gets scaled, not rotated

**Example**:
```
A = [[2, 0],
     [0, 3]]

Eigenvectors: [1, 0] and [0, 1]
Eigenvalues: 2 and 3
```

**Applications in AI**:
- Principal Component Analysis (PCA)
- Understanding neural network dynamics
- Spectral clustering
- PageRank algorithm

## Singular Value Decomposition (SVD)

**Formula**: A = UΣV^T

Where:
- U: Left singular vectors
- Σ: Singular values (diagonal)
- V: Right singular vectors

**Applications**:
- Dimensionality reduction
- Image compression
- Recommender systems
- Latent Semantic Analysis

## Norms

**Vector Norms**:

**L1 Norm** (Manhattan):
```
||v||₁ = |v₁| + |v₂| + ... + |vₙ|
```

**L2 Norm** (Euclidean):
```
||v||₂ = √(v₁² + v₂² + ... + vₙ²)
```

**L∞ Norm** (Maximum):
```
||v||∞ = max(|v₁|, |v₂|, ..., |vₙ|)
```

**In ML**:
- L1 regularization: Sparse solutions
- L2 regularization: Weight decay
- Distance metrics

## Linear Transformations

**Definition**: Function that preserves vector addition and scalar multiplication

**Examples**:
- Rotation
- Scaling
- Shear
- Projection

**Matrix Representation**: Every linear transformation can be represented as matrix multiplication

## Applications in Deep Learning

### 1. Fully Connected Layers

```python
output = input @ weights + bias
# @ is matrix multiplication
```

### 2. Convolutional Layers

Convolution can be expressed as matrix multiplication (Toeplitz matrix)

### 3. Attention Mechanism

```python
Attention(Q, K, V) = softmax(QK^T / √d)V
```

### 4. Batch Normalization

```python
x_normalized = (x - μ) / σ
```

### 5. Gradient Descent

```python
weights = weights - learning_rate * gradient
```

## Computational Considerations

### Broadcasting

NumPy/PyTorch automatically expand dimensions:
```python
[1, 2, 3] + 5 = [6, 7, 8]  # 5 broadcast to [5, 5, 5]
```

### Memory Layout

- Row-major (C): [[1,2,3], [4,5,6]] stored as [1,2,3,4,5,6]
- Column-major (Fortran): stored as [1,4,2,5,3,6]

### GPU Acceleration

- Matrix operations parallelize well
- Batch operations for efficiency
- Use cuBLAS, cuDNN for optimized operations

## Common Operations in PyTorch/NumPy

```python
import numpy as np

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)  # 32

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Matrix multiplication

# Transpose
A_T = A.T

# Inverse
A_inv = np.linalg.inv(A)

# Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, S, Vt = np.linalg.svd(A)

# Norms
l1_norm = np.linalg.norm(v1, ord=1)
l2_norm = np.linalg.norm(v1, ord=2)
```

## Key Formulas Summary

**Dot Product**:
```
a · b = Σ aᵢbᵢ = ||a|| ||b|| cos(θ)
```

**Matrix Multiplication**:
```
(AB)ᵢⱼ = Σ Aᵢₖ Bₖⱼ
```

**Inverse**:
```
AA⁻¹ = I
```

**Eigenvalue Equation**:
```
Av = λv
```

## Practical Tips

1. **Vectorize operations**: Avoid loops, use matrix operations
2. **Check dimensions**: Most bugs are dimension mismatches
3. **Use libraries**: NumPy, PyTorch handle optimization
4. **Batch processing**: Process multiple examples simultaneously
5. **Numerical stability**: Watch for very large/small numbers

## Common Pitfalls

1. **Dimension mismatch** in matrix multiplication
2. **Confusing element-wise and matrix multiplication**
3. **Forgetting to transpose** when needed
4. **Numerical instability** with matrix inversion
5. **Memory issues** with large matrices

## Summary

Linear algebra is the foundation of AI:
- Vectors and matrices represent data
- Matrix operations transform data
- Efficient computation on GPUs
- Essential for understanding deep learning

**Key Concepts**:
- Vector and matrix operations
- Matrix multiplication
- Eigenvalues and eigenvectors
- Linear transformations
- Applications in neural networks

---

**Next**: [Practical Notebooks](../notebooks/)
