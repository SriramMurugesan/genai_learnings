# Pooling and Features

## Introduction

Pooling layers are a critical component of CNNs that reduce spatial dimensions while retaining important information. This document covers pooling operations and how CNNs learn hierarchical features.

## Pooling Operations

### Purpose of Pooling

1. **Reduce spatial dimensions**: Decrease computation and memory
2. **Translation invariance**: Small shifts don't affect output
3. **Increase receptive field**: See larger context
4. **Prevent overfitting**: Reduce parameters

### Max Pooling

**Most common** pooling operation.

**Operation**: Take maximum value in each region

**Example** (2×2 pooling, stride=2):
```
Input (4×4):        Output (2×2):
1  3  2  4          
5  6  7  8    →     6  8
9  2  3  1          9  5
4  5  2  3
```

**Properties**:
- Preserves strongest activations
- No learnable parameters
- Typically 2×2 with stride 2 (halves dimensions)

**Code**:
```python
import torch.nn as nn

maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
# Input: (batch, channels, 32, 32)
# Output: (batch, channels, 16, 16)
```

### Average Pooling

**Operation**: Take average value in each region

**Example** (2×2 pooling):
```
Input (4×4):        Output (2×2):
1  3  2  4          
5  6  7  8    →     3.75  5.25
9  2  3  1          4.5   2.25
4  5  2  3
```

**When to use**:
- Smoother downsampling
- Final layers (global average pooling)

### Global Average Pooling (GAP)

**Operation**: Average entire feature map to single value

**Example**:
```
Input (4×4×512):    Output (512):
Each channel → 1 value
```

**Benefits**:
- Replaces fully connected layers
- Fewer parameters
- More robust to spatial translations
- Used in modern architectures (ResNet, MobileNet)

**Code**:
```python
gap = nn.AdaptiveAvgPool2d((1, 1))
# Input: (batch, 512, 7, 7)
# Output: (batch, 512, 1, 1)
```

### Comparison

| Type | Pros | Cons | Use Case |
|------|------|------|----------|
| Max | Preserves strong features | Loses information | Most common |
| Average | Smoother | Dilutes features | Less common |
| Global Average | No parameters | Aggressive reduction | Final layer |

## Feature Hierarchies

CNNs learn features at multiple levels of abstraction.

### Layer 1 (Low-level features)

**Learns**:
- Edges (horizontal, vertical, diagonal)
- Colors
- Simple textures
- Gradients

**Receptive field**: Small (3×3 to 7×7)

**Example filters**:
```
Horizontal edge:    Vertical edge:
-1 -1 -1            -1  0  1
 0  0  0            -1  0  1
 1  1  1            -1  0  1
```

### Layer 2-3 (Mid-level features)

**Learns**:
- Corners
- Patterns
- Simple shapes
- Textures (stripes, dots)

**Receptive field**: Medium (11×11 to 27×27)

**Combines**: Multiple low-level features

### Layer 4-5 (High-level features)

**Learns**:
- Object parts (eyes, wheels, windows)
- Complex patterns
- Semantic concepts

**Receptive field**: Large (51×51 to 99×99)

### Final Layers (Very high-level)

**Learns**:
- Complete objects
- Faces
- Animals
- Scenes

**Receptive field**: Very large (entire image)

## Receptive Field

### Definition

**Receptive field**: Region of input image that affects a particular neuron's activation.

### Calculation

**For stacked layers**:

**Single layer**:
```
RF = filter_size
```

**Two layers** (both 3×3):
```
RF = 3 + (3-1) = 5
```

**Three layers** (all 3×3):
```
RF = 3 + 2×(3-1) = 7
```

**General formula** (same filter size F, stride 1):
```
RF = F + (n-1) × (F-1)
```
where n = number of layers

**With pooling** (2×2, stride 2):
```
RF doubles after each pooling layer
```

### Effective Receptive Field

**Theoretical vs Effective**:
- Theoretical: Mathematical calculation
- Effective: Actual influence (center pixels matter more)

**Insight**: Not all pixels in receptive field contribute equally!

## Feature Visualization

### Activation Maximization

**Goal**: Find input that maximizes a neuron's activation

**Method**:
1. Start with random image
2. Compute gradient of neuron w.r.t. image
3. Update image to increase activation
4. Repeat

**Reveals**: What the neuron is looking for

### Deconvolution / Guided Backprop

**Goal**: Visualize which input regions activate neurons

**Method**: Backpropagate activations to input space

**Shows**: Important regions for classification

### Class Activation Mapping (CAM)

**Goal**: Highlight discriminative regions for a class

**Method**:
```
CAM = Σ wc × feature_maps
```
where wc = weights for class c

**Example**:
```
Input: Dog image
CAM: Highlights dog's face and body
```

**Code**:
```python
# Get feature maps from last conv layer
features = model.features(image)  # (1, 512, 7, 7)

# Get weights for class
weights = model.fc.weight[class_idx]  # (512,)

# Compute CAM
cam = torch.sum(weights.view(-1, 1, 1) * features, dim=1)
cam = F.relu(cam)  # Only positive contributions
```

### Grad-CAM

**Improvement over CAM**: Works with any architecture

**Method**:
1. Compute gradients of class score w.r.t. feature maps
2. Global average pool gradients
3. Weight feature maps by gradients
4. ReLU to get positive contributions

**More flexible** than CAM!

## Spatial Pyramid Pooling

**Problem**: CNNs require fixed input size

**Solution**: Spatial Pyramid Pooling (SPP)

**Operation**:
1. Divide feature map into fixed number of bins
2. Pool within each bin
3. Concatenate

**Example** (3-level pyramid):
```
Level 1: 1×1 = 1 bin
Level 2: 2×2 = 4 bins
Level 3: 4×4 = 16 bins
Total: 21 bins per channel
```

**Benefit**: Accept any input size!

## Feature Maps

### What are Feature Maps?

**Output** of convolutional layer = feature maps

**Each filter** produces one feature map

**Example**:
```
Input: 32×32×3
64 filters (3×3)
Output: 32×32×64 (64 feature maps)
```

### Interpreting Feature Maps

**Early layers**:
- Visualizable (edges, colors)
- Interpretable

**Deep layers**:
- Abstract
- Hard to interpret
- Semantic meaning

### Number of Feature Maps

**Typical pattern**:
```
Input: 224×224×3
Conv1: 224×224×64
Pool1: 112×112×64
Conv2: 112×112×128
Pool2: 56×56×128
Conv3: 56×56×256
Pool3: 28×28×256
...
```

**Rule**: Double channels when halving spatial dimensions

## Downsampling Strategies

### 1. Max Pooling
- Most common
- 2×2 with stride 2

### 2. Strided Convolution
- Convolution with stride > 1
- Learnable (vs pooling)
- Modern architectures prefer this

### 3. Average Pooling
- Smoother downsampling
- Less common

### Comparison

**Max Pooling**:
- ✓ Simple, fast
- ✓ Translation invariant
- ✗ Not learnable

**Strided Convolution**:
- ✓ Learnable
- ✓ More flexible
- ✗ More parameters

## Practical Tips

### 1. Pooling Size and Stride

**Standard**: 2×2 with stride 2
- Halves dimensions
- Non-overlapping

**Overlapping**: 3×3 with stride 2
- Slight overlap
- Better performance (sometimes)

### 2. When to Pool

**Common pattern**:
```
Conv → ReLU → Conv → ReLU → Pool
```

**Don't pool too early**: Lose information

**Don't pool too much**: Need spatial resolution for some tasks

### 3. Alternatives to Pooling

**Modern trend**: Replace pooling with strided convolutions

**Reason**: Learnable downsampling

**Example** (ResNet):
```
# Instead of:
Conv(3×3) → MaxPool(2×2)

# Use:
Conv(3×3, stride=2)
```

## Feature Extraction vs Fine-tuning

### Feature Extraction

**Use pre-trained CNN** as fixed feature extractor

**Method**:
1. Remove final layer
2. Freeze all weights
3. Add new classifier
4. Train only new layer

**When**: Small dataset, similar domain

### Fine-tuning

**Adjust pre-trained weights** for new task

**Method**:
1. Load pre-trained weights
2. Replace final layer
3. Train with small learning rate

**When**: Medium dataset, related domain

### From Scratch

**Train entire network** from random initialization

**When**: Large dataset, different domain

## Summary

Pooling:
- Reduces spatial dimensions
- Provides translation invariance
- Max pooling most common
- Global average pooling for final layer

Feature Hierarchies:
- Low-level: Edges, colors
- Mid-level: Patterns, shapes
- High-level: Object parts, semantics

Receptive Field:
- Grows with depth
- Important for context
- Effective RF ≠ theoretical RF

Visualization:
- CAM/Grad-CAM for interpretability
- Shows what network learned
- Useful for debugging

**Next**: [Practical Notebooks](../notebooks/)
