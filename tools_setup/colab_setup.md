# Google Colab Setup Guide

## What is Google Colab?

Google Colab (Colaboratory) is a free cloud-based Jupyter notebook environment that provides:
- Free GPU access
- Pre-installed libraries
- No setup required
- Easy sharing and collaboration

## Getting Started

### 1. Access Colab

Visit: [https://colab.research.google.com/](https://colab.research.google.com/)

### 2. Open a Notebook

**Option A: From GitHub**
1. Click `File` → `Open notebook`
2. Select `GitHub` tab
3. Enter repository URL
4. Select notebook

**Option B: Upload**
1. Click `File` → `Upload notebook`
2. Select `.ipynb` file from your computer

**Option C: From Google Drive**
1. Save notebooks to Google Drive
2. Right-click → `Open with` → `Google Colaboratory`

### 3. Enable GPU (Recommended)

1. Click `Runtime` → `Change runtime type`
2. Select `Hardware accelerator` → `GPU` or `TPU`
3. Click `Save`

**Check GPU:**
```python
import torch
print(torch.cuda.is_available()) # Should print True
print(torch.cuda.get_device_name(0)) # GPU name
```

### 4. Install Additional Packages

Most packages are pre-installed. For others:

```python
!pip install package-name
```

Example:
```python
!pip install transformers diffusers
```

### 5. Mount Google Drive (Optional)

To access files from Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Your Drive files will be in `/content/drive/MyDrive/`

## Tips and Tricks

### Keyboard Shortcuts

- `Ctrl/Cmd + Enter`: Run cell
- `Shift + Enter`: Run cell and move to next
- `Ctrl/Cmd + M B`: Insert cell below
- `Ctrl/Cmd + M A`: Insert cell above
- `Ctrl/Cmd + M D`: Delete cell

### GPU Runtime Limits

- **Free tier**: ~12 hours per session
- **Colab Pro**: Longer sessions, faster GPUs
- Save your work frequently!

### Saving Work

**Auto-save:**
- Colab auto-saves to Google Drive
- Check `File` → `Locate in Drive`

**Manual save:**
- `File` → `Save a copy in Drive`
- `File` → `Download` → `.ipynb`

### Running Shell Commands

Prefix with `!`:
```python
!ls # List files
!pwd # Current directory
!nvidia-smi # Check GPU usage
```

### Upload/Download Files

**Upload:**
```python
from google.colab import files
uploaded = files.upload()
```

**Download:**
```python
from google.colab import files
files.download('filename.txt')
```

## Common Issues

### Issue: GPU not available

**Solution:**
1. Check runtime type (Runtime → Change runtime type)
2. GPU quota may be exhausted (wait or use Colab Pro)

### Issue: Session disconnected

**Solution:**
- Colab disconnects after inactivity
- Re-run cells from top
- Use `Runtime` → `Run all`

### Issue: Package not found

**Solution:**
```python
!pip install package-name
```

### Issue: Out of memory

**Solution:**
- Restart runtime: `Runtime` → `Restart runtime`
- Use smaller batch sizes
- Clear variables: `del variable_name`

## Best Practices

1. **Save frequently** - Don't lose your work!
2. **Use GPU wisely** - Only when needed (training, inference)
3. **Comment your code** - Help others (and future you)
4. **Clear outputs** - Before saving (reduces file size)
5. **Organize with sections** - Use markdown headers

## Resources

- [Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab Pro](https://colab.research.google.com/signup) - Paid tier with more resources

## Example: Complete Workflow

```python
# 1. Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# 2. Install packages
!pip install -q transformers

# 3. Mount Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

# 4. Your code here
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
print(generator("Hello, I am", max_length=30))

# 5. Save results
with open('output.txt', 'w') as f:
 f.write("Results...")

# 6. Download
from google.colab import files
files.download('output.txt')
```

---

**Happy Learning with Colab! **
