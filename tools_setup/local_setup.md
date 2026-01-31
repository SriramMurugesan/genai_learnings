# Local Setup Guide

## Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- GPU (optional but recommended for deep learning)

## Installation Steps

### 1. Install Python

**Check if Python is installed:**
```bash
python --version
# or
python3 --version
```

**If not installed:**
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3`
- **Linux**: `sudo apt-get install python3 python3-pip`

### 2. Clone Repository

```bash
git clone <repository-url>
cd genai_learnings
```

### 3. Create Virtual Environment

**Why?** Isolates project dependencies.

```bash
# Create environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will install:
- TensorFlow
- PyTorch
- HuggingFace Transformers
- Jupyter
- NumPy, Pandas, Matplotlib
- And more...

**Note:** Installation may take 10-30 minutes depending on your internet speed.

### 5. Download Datasets

```bash
python datasets/download_datasets.py
```

### 6. Start Jupyter

```bash
# Jupyter Notebook
jupyter notebook

# or Jupyter Lab (recommended)
jupyter lab
```

Your browser will open automatically at `http://localhost:8888`

## GPU Setup (Optional but Recommended)

### NVIDIA GPU (CUDA)

**1. Check GPU:**
```bash
nvidia-smi
```

**2. Install CUDA Toolkit:**
- Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- Follow installation instructions for your OS

**3. Install cuDNN:**
- Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
- Requires NVIDIA account (free)

**4. Verify PyTorch GPU:**
```python
import torch
print(torch.cuda.is_available()) # Should be True
print(torch.cuda.get_device_name(0))
```

**5. Verify TensorFlow GPU:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### AMD GPU (ROCm) - Linux only

Follow [ROCm installation guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

### Apple Silicon (M1/M2/M3)

PyTorch and TensorFlow support Metal acceleration:

```bash
# Already included in requirements.txt
pip install torch torchvision
pip install tensorflow-macos tensorflow-metal
```

## IDE Setup (Optional)

### VS Code

**Install:**
- Download from [code.visualstudio.com](https://code.visualstudio.com/)

**Extensions:**
- Python
- Jupyter
- Pylance

**Open project:**
```bash
code .
```

### PyCharm

**Install:**
- Download from [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)

**Configure interpreter:**
1. Settings → Project → Python Interpreter
2. Add → Existing environment
3. Select `venv/bin/python`

## Troubleshooting

### Issue: `pip` not found

**Solution:**
```bash
python -m pip install --upgrade pip
```

### Issue: Permission denied

**Solution:**
```bash
# Don't use sudo! Use virtual environment instead
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Out of memory

**Solution:**
- Close other applications
- Use smaller batch sizes in code
- Upgrade RAM

### Issue: CUDA not found

**Solution:**
1. Verify CUDA installation: `nvcc --version`
2. Reinstall PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Jupyter kernel not found

**Solution:**
```bash
python -m ipykernel install --user --name=venv
```

## Updating Dependencies

```bash
# Activate environment
source venv/bin/activate # or venv\Scripts\activate on Windows

# Update all packages
pip install --upgrade -r requirements.txt
```

## Uninstallation

```bash
# Deactivate environment
deactivate

# Remove virtual environment
rm -rf venv # Linux/macOS
# or
rmdir /s venv # Windows
```

## Performance Tips

1. **Use GPU** for deep learning (10-100x faster)
2. **Close unused applications** to free RAM
3. **Use SSD** for faster data loading
4. **Batch processing** for large datasets
5. **Monitor resources** with `htop` (Linux/macOS) or Task Manager (Windows)

## Resources

- [Python Documentation](https://docs.python.org/3/)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

---

**Happy Learning! **
