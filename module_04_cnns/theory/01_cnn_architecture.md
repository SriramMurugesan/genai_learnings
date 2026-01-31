# CNN Architecture

## Introduction

Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data, particularly images. They have revolutionized computer vision and are the foundation of modern image recognition systems.

## Why CNNs for Images?

### Problems with Fully Connected Networks

**Image as input**: 28×28 grayscale image = 784 pixels

**Fully connected layer**: 784 → 128 neurons
- Parameters: 784 × 128 + 128 = **100,480**

**For RGB image** (224×224×3):
- 150,528 pixels → 128 neurons = **19,267,712 parameters!**

**Problems**:
1. Too many parameters → overfitting
2. No spatial structure preserved
3. Not translation invariant
4. Computationally expensive

### CNN Solution

**Key ideas**:
1. **Local connectivity**: Neurons connect to small regions
2. **Parameter sharing**: Same weights across image
3. **Spatial hierarchy**: Learn features at multiple scales

## CNN Architecture Components

### 1. Convolutional Layer

**Core building block** of CNNs.

**Operation**: Slide a filter (kernel) across the input

**Formula**:
```
Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Filter[m,n] + bias
```

**Example** (3×3 filter on 5×5 input):
```
Input:          Filter:         Output:
1 2 3 4 5       1 0 -1          
2 3 4 5 6       1 0 -1          -8 -8 -8
3 4 5 6 7   ×   1 0 -1    =     -8 -8 -8
4 5 6 7 8                       -8 -8 -8
5 6 7 8 9
```

**Properties**:
- **Input**: H × W × C (height, width, channels)
- **Filter**: F × F × C (typically 3×3 or 5×5)
- **Output**: H' × W' × K (K = number of filters)

**Parameters**: F × F × C × K + K (bias)

### 2. Pooling Layer

**Purpose**: Reduce spatial dimensions, increase receptive field

**Max Pooling** (most common):
```
Input (4×4):        Output (2×2):
1  3  2  4          
5  6  7  8    →     6  8
9  2  3  1          9  5
4  5  2  3
```
Takes maximum value in each 2×2 region.

**Average Pooling**:
Takes average instead of maximum.

**Properties**:
- No learnable parameters
- Reduces spatial size
- Provides translation invariance
- Common: 2×2 with stride 2

### 3. Fully Connected Layer

**Purpose**: Final classification/regression

**Operation**: Standard neural network layer
```
output = activation(W × flattened_features + b)
```

**Typically used**:
- At the end of CNN
- After feature extraction
- For final prediction

### 4. Activation Functions

**ReLU** (most common):
```
ReLU(x) = max(0, x)
```

Applied after convolutional layers.

### 5. Batch Normalization

**Purpose**: Normalize activations, speed up training

**Operation**:
```
x_normalized = (x - μ) / √(σ² + ε)
x_scaled = γ × x_normalized + β
```

**Benefits**:
- Faster training
- Higher learning rates
- Less sensitive to initialization

### 6. Dropout

**Purpose**: Regularization, prevent overfitting

**Operation**: Randomly set activations to 0 during training
```python
if training:
    mask = np.random.rand(*x.shape) > dropout_rate
    x = x * mask / (1 - dropout_rate)
```

## Complete CNN Architecture Example

### LeNet-5 (1998)

**Architecture**:
```
Input (32×32×1)
    ↓
Conv1: 6 filters, 5×5 → (28×28×6)
    ↓ ReLU
MaxPool: 2×2 → (14×14×6)
    ↓
Conv2: 16 filters, 5×5 → (10×10×16)
    ↓ ReLU
MaxPool: 2×2 → (5×5×16)
    ↓
Flatten → 400
    ↓
FC1: 120 neurons
    ↓ ReLU
FC2: 84 neurons
    ↓ ReLU
FC3: 10 neurons (output)
    ↓ Softmax
```

**Parameters**:
- Conv1: 5×5×1×6 + 6 = 156
- Conv2: 5×5×6×16 + 16 = 2,416
- FC1: 400×120 + 120 = 48,120
- FC2: 120×84 + 84 = 10,164
- FC3: 84×10 + 10 = 850
- **Total: 61,706 parameters**

## Popular CNN Architectures

### AlexNet (2012)

**Key innovations**:
- ReLU activation
- Dropout
- Data augmentation
- GPU training

**Architecture**: 5 conv layers + 3 FC layers
**Parameters**: ~60 million
**ImageNet top-5 error**: 15.3%

### VGGNet (2014)

**Key idea**: Deeper networks with small filters

**VGG-16**:
- 13 conv layers (all 3×3 filters)
- 3 FC layers
- **Parameters**: 138 million

**Insight**: Stack of 3×3 filters = larger receptive field

### GoogLeNet / Inception (2014)

**Key innovation**: Inception module

**Inception module**:
```
Input
  ↓ ↓ ↓ ↓
  1×1 3×3 5×5 MaxPool
  ↓ ↓ ↓ ↓
  Concatenate
  ↓
Output
```

**Benefits**:
- Multiple filter sizes
- Fewer parameters than VGG
- **Parameters**: 6.8 million

### ResNet (2015)

**Key innovation**: Residual connections (skip connections)

**Residual block**:
```
Input (x)
  ↓
Conv → ReLU → Conv
  ↓
  + ← x (skip connection)
  ↓
ReLU
  ↓
Output
```

**Formula**: `F(x) + x` instead of just `F(x)`

**Benefits**:
- Train very deep networks (50, 101, 152 layers)
- Solves vanishing gradient problem
- Better accuracy

**ResNet-50**: 25.6 million parameters

### MobileNet (2017)

**Key innovation**: Depthwise separable convolutions

**Purpose**: Efficient for mobile devices

**Benefits**:
- Fewer parameters
- Faster inference
- Good accuracy

### EfficientNet (2019)

**Key innovation**: Compound scaling

**Scales**:
- Depth (number of layers)
- Width (number of channels)
- Resolution (input size)

**State-of-the-art** accuracy with fewer parameters.

## Receptive Field

**Definition**: Region of input that affects a neuron's output

**Example**:
```
Layer 1 (3×3 filter): Receptive field = 3×3
Layer 2 (3×3 filter): Receptive field = 5×5
Layer 3 (3×3 filter): Receptive field = 7×7
```

**Formula** (for stacked 3×3 filters):
```
Receptive field = 1 + 2 × n
```
where n = number of layers

**Importance**: Larger receptive field → see more context

## Feature Hierarchy

CNNs learn hierarchical features:

**Early layers** (low-level features):
- Edges
- Corners
- Colors
- Textures

**Middle layers** (mid-level features):
- Shapes
- Patterns
- Object parts

**Deep layers** (high-level features):
- Complete objects
- Faces
- Scenes

## Calculating Output Dimensions

**Formula**:
```
Output size = (Input size - Filter size + 2×Padding) / Stride + 1
```

**Example**:
- Input: 32×32
- Filter: 5×5
- Padding: 0
- Stride: 1

Output = (32 - 5 + 0) / 1 + 1 = **28×28**

**With padding**:
- Padding: 2
Output = (32 - 5 + 4) / 1 + 1 = **32×32** (same size!)

## Padding Types

### Valid Padding (No padding)
- Output smaller than input
- Lose border information

### Same Padding
- Pad to keep output size = input size
- Preserve spatial dimensions

**Formula** for same padding:
```
Padding = (Filter size - 1) / 2
```

For 3×3 filter: Padding = 1
For 5×5 filter: Padding = 2

## Stride

**Stride**: Step size when sliding filter

**Stride = 1**: Slide one pixel at a time
**Stride = 2**: Slide two pixels (reduces output by half)

**Effect**: Larger stride → smaller output

## Design Principles

### 1. Start Simple
- Few layers
- Standard architecture (e.g., VGG-style)
- Gradually increase complexity

### 2. Use Proven Architectures
- ResNet for general tasks
- MobileNet for mobile/edge
- EfficientNet for best accuracy

### 3. Common Patterns
```
Conv → ReLU → Conv → ReLU → MaxPool
```
Repeat this pattern multiple times.

### 4. Increase Depth Gradually
- Double channels when halving spatial size
```
64 channels (32×32) → 128 channels (16×16) → 256 channels (8×8)
```

### 5. Use Batch Normalization
- After conv layers
- Before or after activation

### 6. Add Dropout
- In FC layers (0.5)
- Sometimes in conv layers (0.2-0.3)

## Practical Example

**Image Classification CNN**:
```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
```

## Summary

CNNs are:
- Specialized for grid-like data (images)
- Built from conv, pooling, and FC layers
- Learn hierarchical features
- Much more efficient than fully connected networks

**Key Components**:
- Convolutional layers (feature extraction)
- Pooling layers (dimensionality reduction)
- Fully connected layers (classification)
- Batch normalization (training stability)
- Dropout (regularization)

**Popular Architectures**:
- LeNet, AlexNet (historical)
- VGG (simple, deep)
- ResNet (skip connections)
- MobileNet (efficient)
- EfficientNet (state-of-the-art)

**Next**: [Convolution Operations](./02_convolution_operations.md)
