# Datasets

This directory contains scripts to download all datasets used in the course.

## Quick Start

```bash
python download_datasets.py
```

## Datasets Included

### Scikit-learn
- **Iris**: Flower classification (150 samples, 4 features, 3 classes)
- **Diabetes**: Disease progression prediction (442 samples, 10 features)
- **California Housing**: House price prediction

### TensorFlow/Keras
- **MNIST**: Handwritten digits (60k train, 10k test, 28x28 grayscale)
- **CIFAR-10**: Object recognition (50k train, 10k test, 32x32 color, 10 classes)
- **IMDB**: Movie review sentiment (25k train, 25k test)

### PyTorch
- **MNIST**: Same as above, PyTorch format
- **CIFAR-10**: Same as above, PyTorch format

## Storage

Datasets are automatically cached in:
- `~/.keras/datasets/` (Keras)
- `./datasets/` (PyTorch)
- Memory cache (scikit-learn)

## Manual Download

If automatic download fails, you can manually download from:
- [Kaggle](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
