# Convolution Operations

## Introduction

Convolution is the fundamental operation in CNNs. Understanding how convolution works mathematically and computationally is essential for working with CNNs.

## What is Convolution?

**Mathematical definition**: Convolution is an operation on two functions that produces a third function.

**In CNNs**: Sliding a filter (kernel) over an input to produce a feature map.

## 2D Convolution

### Basic Operation

**Input**: Image I (H × W)
**Filter**: Kernel K (F × F)
**Output**: Feature map O

**Formula**:
```
O[i,j] = Σₘ Σₙ I[i+m, j+n] × K[m,n]
```

### Step-by-Step Example

**Input** (5×5):
```
1 2 3 4 5
2 3 4 5 6
3 4 5 6 7
4 5 6 7 8
5 6 7 8 9
```

**Filter** (3×3 edge detector):
```
-1 -1 -1
 0  0  0
 1  1  1
```

**Computation** at position (1,1):
```
1×(-1) + 2×(-1) + 3×(-1) +
2×0    + 3×0    + 4×0    +
3×1    + 4×1    + 5×1    = -3 + 0 + 12 = 9
```

**Output** (3×3):
```
 9  9  9
 9  9  9
 9  9  9
```

## Convolution Parameters

### 1. Filter Size (Kernel Size)

**Common sizes**: 1×1, 3×3, 5×5, 7×7

**3×3 filters** (most popular):
- Good balance between receptive field and computation
- Can stack multiple 3×3 to get larger receptive field
- Two 3×3 filters = one 5×5 receptive field (fewer parameters!)

**1×1 filters**:
- Change number of channels
- Add non-linearity
- Reduce/increase dimensions

### 2. Stride

**Definition**: Step size when sliding filter

**Stride = 1**: Slide one pixel at a time
**Stride = 2**: Skip every other pixel

**Output size with stride**:
```
Output = ⌊(Input - Filter) / Stride⌋ + 1
```

**Example**:
- Input: 7×7
- Filter: 3×3
- Stride: 2
- Output: ⌊(7-3)/2⌋ + 1 = 3×3

### 3. Padding

**Purpose**: Control output size, preserve border information

**Types**:

**Valid (No padding)**:
```
Input: 5×5, Filter: 3×3
Output: 3×3
```

**Same (Zero padding)**:
```
Input: 5×5, Filter: 3×3, Padding: 1
Output: 5×5 (same as input)
```

**Padding formula** for same output:
```
Padding = (Filter - 1) / 2
```

**Example with padding**:
```
Input (3×3):        Padded (5×5):
1 2 3               0 0 0 0 0
4 5 6      →        0 1 2 3 0
7 8 9               0 4 5 6 0
                    0 7 8 9 0
                    0 0 0 0 0
```

### 4. Dilation

**Purpose**: Increase receptive field without increasing parameters

**Dilated convolution**: Insert spaces between filter elements

**Example** (3×3 filter, dilation=2):
```
Normal:     Dilated:
1 1 1       1 0 1 0 1
1 1 1       0 0 0 0 0
1 1 1       1 0 1 0 1
            0 0 0 0 0
            1 0 1 0 1
```

**Effective receptive field**: 5×5 with 3×3 parameters!

## Multi-Channel Convolution

### RGB Images (3 channels)

**Input**: H × W × 3
**Filter**: F × F × 3 (must match input channels)
**Output**: H' × W' × 1 (single feature map)

**Operation**:
```
Output[i,j] = Σₘ Σₙ Σc Input[i+m, j+n, c] × Filter[m,n,c] + bias
```

Sum across all channels!

**Example**:
```
Input: 5×5×3 (RGB)
Filter: 3×3×3
Output: 3×3×1
```

### Multiple Filters

**K filters** → **K feature maps**

**Input**: H × W × C
**Filters**: K filters of size F × F × C
**Output**: H' × W' × K

**Example**:
```
Input: 32×32×3
64 filters of 3×3×3
Output: 32×32×64 (with same padding)
```

## Convolution as Matrix Multiplication

Convolution can be expressed as matrix multiplication using **Toeplitz matrices**.

**Example**:
```
Input (flattened): [1,2,3,4,5,6,7,8,9]
Filter: [a,b,c]

Toeplitz matrix:
[a b c 0 0 0 0 0 0]
[0 a b c 0 0 0 0 0]
[0 0 a b c 0 0 0 0]
...

Output = Toeplitz × Input
```

**Benefits**:
- Leverage optimized matrix multiplication
- GPU acceleration
- Framework implementation

## Computational Complexity

### FLOPs (Floating Point Operations)

**For one convolution**:
```
FLOPs = Output_H × Output_W × Filter_H × Filter_W × Input_Channels × Output_Channels
```

**Example**:
- Input: 32×32×3
- Filter: 3×3
- Output: 32×32×64

FLOPs = 32 × 32 × 3 × 3 × 3 × 64 = **17,694,720**

### Memory Requirements

**Activations**:
```
Memory = Batch_size × H × W × Channels × 4 bytes (float32)
```

**Example** (batch=32, 224×224×64):
```
Memory = 32 × 224 × 224 × 64 × 4 = 411 MB
```

## Transposed Convolution

**Purpose**: Upsampling (increase spatial dimensions)

**Also called**: Deconvolution, fractionally-strided convolution

**Use cases**:
- Semantic segmentation
- GANs (generators)
- Autoencoders

**Example**:
```
Input: 2×2
Filter: 3×3, stride=2
Output: 5×5
```

**Operation**: Insert zeros between input pixels, then convolve

## Depthwise Separable Convolution

**Purpose**: Reduce parameters and computation

**Standard convolution**:
- Input: H×W×C
- K filters of F×F×C
- Parameters: F×F×C×K

**Depthwise separable**:
1. **Depthwise**: One filter per input channel (F×F×1)
2. **Pointwise**: 1×1 convolution to combine

**Parameters**:
- Depthwise: F×F×C
- Pointwise: 1×1×C×K
- **Total**: F×F×C + C×K

**Reduction**:
```
Standard: F×F×C×K
Separable: F×F×C + C×K

Ratio ≈ 1/K + 1/(F×F)
```

For F=3, K=64: **8-9x fewer parameters!**

**Used in**: MobileNet, EfficientNet

## Grouped Convolution

**Purpose**: Reduce parameters, increase efficiency

**Operation**: Split channels into groups, convolve separately

**Example**:
- Input: 64 channels
- Groups: 4
- Each group: 16 channels

**Parameters reduction**: 4x fewer

**Used in**: ResNeXt, MobileNet

## Practical Implementation

### NumPy (Educational)

```python
import numpy as np

def convolve2d(image, kernel, stride=1, padding=0):
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    
    h, w = image.shape
    kh, kw = kernel.shape
    
    # Output dimensions
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    # Convolve
    for i in range(0, out_h):
        for j in range(0, out_w):
            region = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Example
image = np.array([[1,2,3,4,5],
                  [2,3,4,5,6],
                  [3,4,5,6,7],
                  [4,5,6,7,8],
                  [5,6,7,8,9]])

kernel = np.array([[-1,-1,-1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]])

result = convolve2d(image, kernel)
print(result)
```

### PyTorch

```python
import torch
import torch.nn as nn

# Single convolution
conv = nn.Conv2d(in_channels=3, out_channels=64, 
                 kernel_size=3, stride=1, padding=1)

# Input: batch_size=1, channels=3, height=32, width=32
x = torch.randn(1, 3, 32, 32)
output = conv(x)
print(output.shape)  # torch.Size([1, 64, 32, 32])
```

## Common Filter Types

### Edge Detection

**Horizontal edges**:
```
-1 -1 -1
 0  0  0
 1  1  1
```

**Vertical edges**:
```
-1  0  1
-1  0  1
-1  0  1
```

**Sobel (horizontal)**:
```
-1 -2 -1
 0  0  0
 1  2  1
```

### Blur

**Box blur**:
```
1/9 1/9 1/9
1/9 1/9 1/9
1/9 1/9 1/9
```

**Gaussian blur**:
```
1/16 2/16 1/16
2/16 4/16 2/16
1/16 2/16 1/16
```

### Sharpen

```
 0 -1  0
-1  5 -1
 0 -1  0
```

## Summary

Convolution is:
- The core operation in CNNs
- Sliding a filter over input
- Controlled by size, stride, padding, dilation

**Key Concepts**:
- Filter size (3×3 most common)
- Stride (controls output size)
- Padding (preserves dimensions)
- Multi-channel convolution
- Depthwise separable (efficient)

**Computational aspects**:
- Can be expressed as matrix multiplication
- GPU-accelerated
- Memory and compute intensive

**Next**: [Pooling and Features](./03_pooling_and_features.md)
