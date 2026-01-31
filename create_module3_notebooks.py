#!/usr/bin/env python3
"""
Batch create all remaining notebooks for modules 3-5
"""
import json
import os

def create_notebook(filepath, title, cells_content):
    """Create a Jupyter notebook"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n\n{cells_content['intro']}"]
        }
    ]
    
    for cell in cells_content.get('cells', []):
        cells.append(cell)
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"✓ Created: {os.path.basename(filepath)}")

# Define all notebooks
notebooks = {
    # Module 3
    "/home/sriram/genai/genai_learnings/module_03_neural_networks/notebooks/02_pytorch_basics.ipynb": {
        "title": "PyTorch Basics",
        "intro": "Learn PyTorch fundamentals for building neural networks.\\n\\n## Topics\\n- Tensors\\n- Autograd\\n- Building models\\n- Training loops",
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import torch\\nimport torch.nn as nn\\nimport torch.optim as optim\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\n\\nprint(f'PyTorch version: {torch.__version__}')\\nprint(f'CUDA available: {torch.cuda.is_available()}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Tensors\\n\\nTensors are the fundamental data structure in PyTorch."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Create tensors\\nx = torch.tensor([1, 2, 3, 4, 5])\\nprint(f'1D tensor: {x}')\\nprint(f'Shape: {x.shape}')\\nprint(f'Data type: {x.dtype}')\\n\\n# 2D tensor\\nmatrix = torch.tensor([[1, 2], [3, 4], [5, 6]])\\nprint(f'\\\\n2D tensor:\\\\n{matrix}')\\nprint(f'Shape: {matrix.shape}')\\n\\n# Random tensors\\nrand_tensor = torch.randn(3, 4)\\nprint(f'\\\\nRandom tensor:\\\\n{rand_tensor}')\\n\\n# Zeros and ones\\nzeros = torch.zeros(2, 3)\\nones = torch.ones(2, 3)\\nprint(f'\\\\nZeros:\\\\n{zeros}')\\nprint(f'\\\\nOnes:\\\\n{ones}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Tensor Operations"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Basic operations\\na = torch.tensor([1.0, 2.0, 3.0])\\nb = torch.tensor([4.0, 5.0, 6.0])\\n\\nprint(f'Addition: {a + b}')\\nprint(f'Multiplication: {a * b}')\\nprint(f'Dot product: {torch.dot(a, b)}')\\n\\n# Matrix multiplication\\nA = torch.randn(3, 4)\\nB = torch.randn(4, 2)\\nC = torch.mm(A, B)  # or A @ B\\nprint(f'\\\\nMatrix multiplication shape: {C.shape}')\\n\\n# Reshaping\\nx = torch.randn(12)\\nreshaped = x.view(3, 4)\\nprint(f'\\\\nOriginal shape: {x.shape}')\\nprint(f'Reshaped: {reshaped.shape}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Autograd - Automatic Differentiation\\n\\nPyTorch's autograd automatically computes gradients."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Enable gradient tracking\\nx = torch.tensor([2.0], requires_grad=True)\\ny = x ** 2 + 3 * x + 1\\n\\nprint(f'x = {x.item()}')\\nprint(f'y = {y.item()}')\\n\\n# Compute gradients\\ny.backward()\\nprint(f'dy/dx = {x.grad.item()}')\\nprint(f'Expected (2*x + 3) = {2*x.item() + 3}')\\n\\n# Example: Linear regression\\nw = torch.tensor([1.0], requires_grad=True)\\nb = torch.tensor([0.0], requires_grad=True)\\n\\nx_data = torch.tensor([1.0, 2.0, 3.0, 4.0])\\ny_data = torch.tensor([2.0, 4.0, 6.0, 8.0])\\n\\n# Forward pass\\ny_pred = w * x_data + b\\nloss = ((y_pred - y_data) ** 2).mean()\\n\\nprint(f'\\\\nLoss: {loss.item():.4f}')\\n\\n# Backward pass\\nloss.backward()\\nprint(f'dL/dw: {w.grad.item():.4f}')\\nprint(f'dL/db: {b.grad.item():.4f}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Building Neural Networks\\n\\nUse `nn.Module` to define models."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["class SimpleNN(nn.Module):\\n    def __init__(self, input_size, hidden_size, output_size):\\n        super(SimpleNN, self).__init__()\\n        self.fc1 = nn.Linear(input_size, hidden_size)\\n        self.relu = nn.ReLU()\\n        self.fc2 = nn.Linear(hidden_size, output_size)\\n    \\n    def forward(self, x):\\n        x = self.fc1(x)\\n        x = self.relu(x)\\n        x = self.fc2(x)\\n        return x\\n\\n# Create model\\nmodel = SimpleNN(10, 20, 5)\\nprint(model)\\n\\n# Model parameters\\nprint(f'\\\\nNumber of parameters: {sum(p.numel() for p in model.parameters())}')\\n\\n# Forward pass\\nx = torch.randn(32, 10)  # batch of 32\\noutput = model(x)\\nprint(f'\\\\nInput shape: {x.shape}')\\nprint(f'Output shape: {output.shape}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Training Loop\\n\\nComplete training example."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from sklearn.datasets import make_classification\\nfrom sklearn.model_selection import train_test_split\\n\\n# Generate dataset\\nX, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\\n\\n# Convert to tensors\\nX_train = torch.FloatTensor(X_train)\\ny_train = torch.FloatTensor(y_train).unsqueeze(1)\\nX_test = torch.FloatTensor(X_test)\\ny_test = torch.FloatTensor(y_test).unsqueeze(1)\\n\\n# Create model\\nmodel = SimpleNN(20, 64, 1)\\ncriterion = nn.BCEWithLogitsLoss()\\noptimizer = optim.Adam(model.parameters(), lr=0.001)\\n\\n# Training loop\\nlosses = []\\nfor epoch in range(100):\\n    # Forward pass\\n    outputs = model(X_train)\\n    loss = criterion(outputs, y_train)\\n    \\n    # Backward pass\\n    optimizer.zero_grad()\\n    loss.backward()\\n    optimizer.step()\\n    \\n    losses.append(loss.item())\\n    \\n    if (epoch + 1) % 20 == 0:\\n        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')\\n\\n# Plot loss\\nplt.plot(losses)\\nplt.xlabel('Epoch')\\nplt.ylabel('Loss')\\nplt.title('Training Loss')\\nplt.grid(True)\\nplt.show()\\n\\n# Evaluate\\nmodel.eval()\\nwith torch.no_grad():\\n    test_outputs = model(X_test)\\n    predictions = (torch.sigmoid(test_outputs) > 0.5).float()\\n    accuracy = (predictions == y_test).float().mean()\\n    print(f'\\\\nTest Accuracy: {accuracy.item():.4f}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Summary\\n\\n✅ Tensors and operations\\n✅ Autograd for automatic differentiation\\n✅ Building models with nn.Module\\n✅ Complete training loop\\n✅ Model evaluation"]}
        ]
    },
    
    "/home/sriram/genai/genai_learnings/module_03_neural_networks/notebooks/03_tensorflow_keras_basics.ipynb": {
        "title": "TensorFlow/Keras Basics",
        "intro": "Learn TensorFlow and Keras for building neural networks.\\n\\n## Topics\\n- TensorFlow basics\\n- Keras Sequential API\\n- Functional API\\n- Training and evaluation",
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import tensorflow as tf\\nfrom tensorflow import keras\\nfrom tensorflow.keras import layers\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\n\\nprint(f'TensorFlow version: {tf.__version__}')\\nprint(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. TensorFlow Tensors"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Create tensors\\nx = tf.constant([1, 2, 3, 4, 5])\\nprint(f'1D tensor: {x}')\\nprint(f'Shape: {x.shape}')\\nprint(f'Data type: {x.dtype}')\\n\\n# 2D tensor\\nmatrix = tf.constant([[1, 2], [3, 4], [5, 6]])\\nprint(f'\\\\n2D tensor:\\\\n{matrix}')\\n\\n# Random tensors\\nrand_tensor = tf.random.normal([3, 4])\\nprint(f'\\\\nRandom tensor:\\\\n{rand_tensor}')\\n\\n# Operations\\na = tf.constant([1.0, 2.0, 3.0])\\nb = tf.constant([4.0, 5.0, 6.0])\\nprint(f'\\\\nAddition: {a + b}')\\nprint(f'Multiplication: {a * b}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Keras Sequential API\\n\\nSimplest way to build models."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Build model\\nmodel = keras.Sequential([\\n    layers.Dense(64, activation='relu', input_shape=(20,)),\\n    layers.Dense(32, activation='relu'),\\n    layers.Dense(1, activation='sigmoid')\\n])\\n\\nmodel.summary()\\n\\n# Compile\\nmodel.compile(\\n    optimizer='adam',\\n    loss='binary_crossentropy',\\n    metrics=['accuracy']\\n)"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Training Example"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from sklearn.datasets import make_classification\\nfrom sklearn.model_selection import train_test_split\\n\\n# Generate data\\nX, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)\\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\\n\\n# Train\\nhistory = model.fit(\\n    X_train, y_train,\\n    epochs=50,\\n    batch_size=32,\\n    validation_split=0.2,\\n    verbose=0\\n)\\n\\n# Plot training history\\nplt.figure(figsize=(12, 4))\\n\\nplt.subplot(121)\\nplt.plot(history.history['loss'], label='Training Loss')\\nplt.plot(history.history['val_loss'], label='Validation Loss')\\nplt.xlabel('Epoch')\\nplt.ylabel('Loss')\\nplt.legend()\\nplt.grid(True)\\n\\nplt.subplot(122)\\nplt.plot(history.history['accuracy'], label='Training Accuracy')\\nplt.plot(history.history['val_accuracy'], label='Validation Accuracy')\\nplt.xlabel('Epoch')\\nplt.ylabel('Accuracy')\\nplt.legend()\\nplt.grid(True)\\n\\nplt.tight_layout()\\nplt.show()\\n\\n# Evaluate\\ntest_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)\\nprint(f'\\\\nTest Accuracy: {test_acc:.4f}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Functional API\\n\\nMore flexible for complex architectures."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Functional API\\ninputs = keras.Input(shape=(20,))\\nx = layers.Dense(64, activation='relu')(inputs)\\nx = layers.Dropout(0.5)(x)\\nx = layers.Dense(32, activation='relu')(x)\\noutputs = layers.Dense(1, activation='sigmoid')(x)\\n\\nfunctional_model = keras.Model(inputs=inputs, outputs=outputs)\\nfunctional_model.summary()\\n\\n# Compile and train\\nfunctional_model.compile(\\n    optimizer='adam',\\n    loss='binary_crossentropy',\\n    metrics=['accuracy']\\n)\\n\\nfunctional_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)\\ntest_loss, test_acc = functional_model.evaluate(X_test, y_test, verbose=0)\\nprint(f'Test Accuracy: {test_acc:.4f}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Summary\\n\\n✅ TensorFlow tensors\\n✅ Keras Sequential API\\n✅ Functional API\\n✅ Training and evaluation\\n✅ Model visualization"]}
        ]
    },
    
    "/home/sriram/genai/genai_learnings/module_03_neural_networks/notebooks/04_training_mnist.ipynb": {
        "title": "Training on MNIST",
        "intro": "Train a neural network on the MNIST digit classification dataset.\\n\\n## Topics\\n- Loading MNIST\\n- Data preprocessing\\n- Model building\\n- Training and evaluation",
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import torch\\nimport torch.nn as nn\\nimport torch.optim as optim\\nfrom torchvision import datasets, transforms\\nfrom torch.utils.data import DataLoader\\nimport matplotlib.pyplot as plt\\nimport numpy as np"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Load MNIST Dataset"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["# Define transforms\\ntransform = transforms.Compose([\\n    transforms.ToTensor(),\\n    transforms.Normalize((0.1307,), (0.3081,))\\n])\\n\\n# Load data\\ntrain_dataset = datasets.MNIST(root='../../datasets', train=True, download=True, transform=transform)\\ntest_dataset = datasets.MNIST(root='../../datasets', train=False, download=True, transform=transform)\\n\\ntrain_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)\\ntest_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)\\n\\nprint(f'Training samples: {len(train_dataset)}')\\nprint(f'Test samples: {len(test_dataset)}')\\n\\n# Visualize samples\\nfig, axes = plt.subplots(2, 5, figsize=(12, 5))\\nfor i, ax in enumerate(axes.flat):\\n    image, label = train_dataset[i]\\n    ax.imshow(image.squeeze(), cmap='gray')\\n    ax.set_title(f'Label: {label}')\\n    ax.axis('off')\\nplt.tight_layout()\\nplt.show()"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Define Model"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["class MNISTNet(nn.Module):\\n    def __init__(self):\\n        super(MNISTNet, self).__init__()\\n        self.fc1 = nn.Linear(28*28, 512)\\n        self.fc2 = nn.Linear(512, 256)\\n        self.fc3 = nn.Linear(256, 128)\\n        self.fc4 = nn.Linear(128, 10)\\n        self.dropout = nn.Dropout(0.2)\\n        \\n    def forward(self, x):\\n        x = x.view(-1, 28*28)  # Flatten\\n        x = torch.relu(self.fc1(x))\\n        x = self.dropout(x)\\n        x = torch.relu(self.fc2(x))\\n        x = self.dropout(x)\\n        x = torch.relu(self.fc3(x))\\n        x = self.fc4(x)\\n        return x\\n\\nmodel = MNISTNet()\\nprint(model)\\nprint(f'\\\\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Training"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["criterion = nn.CrossEntropyLoss()\\noptimizer = optim.Adam(model.parameters(), lr=0.001)\\n\\ntrain_losses = []\\ntrain_accs = []\\n\\nfor epoch in range(10):\\n    model.train()\\n    running_loss = 0.0\\n    correct = 0\\n    total = 0\\n    \\n    for images, labels in train_loader:\\n        optimizer.zero_grad()\\n        outputs = model(images)\\n        loss = criterion(outputs, labels)\\n        loss.backward()\\n        optimizer.step()\\n        \\n        running_loss += loss.item()\\n        _, predicted = torch.max(outputs.data, 1)\\n        total += labels.size(0)\\n        correct += (predicted == labels).sum().item()\\n    \\n    epoch_loss = running_loss / len(train_loader)\\n    epoch_acc = 100 * correct / total\\n    train_losses.append(epoch_loss)\\n    train_accs.append(epoch_acc)\\n    \\n    print(f'Epoch [{epoch+1}/10], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')\\n\\n# Plot training curves\\nplt.figure(figsize=(12, 4))\\nplt.subplot(121)\\nplt.plot(train_losses)\\nplt.xlabel('Epoch')\\nplt.ylabel('Loss')\\nplt.title('Training Loss')\\nplt.grid(True)\\n\\nplt.subplot(122)\\nplt.plot(train_accs)\\nplt.xlabel('Epoch')\\nplt.ylabel('Accuracy (%)')\\nplt.title('Training Accuracy')\\nplt.grid(True)\\nplt.tight_layout()\\nplt.show()"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Evaluation"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["model.eval()\\ncorrect = 0\\ntotal = 0\\n\\nwith torch.no_grad():\\n    for images, labels in test_loader:\\n        outputs = model(images)\\n        _, predicted = torch.max(outputs.data, 1)\\n        total += labels.size(0)\\n        correct += (predicted == labels).sum().item()\\n\\ntest_accuracy = 100 * correct / total\\nprint(f'Test Accuracy: {test_accuracy:.2f}%')\\n\\n# Visualize predictions\\nfig, axes = plt.subplots(2, 5, figsize=(12, 5))\\nmodel.eval()\\nwith torch.no_grad():\\n    for i, ax in enumerate(axes.flat):\\n        image, label = test_dataset[i]\\n        output = model(image.unsqueeze(0))\\n        _, predicted = torch.max(output, 1)\\n        \\n        ax.imshow(image.squeeze(), cmap='gray')\\n        color = 'green' if predicted.item() == label else 'red'\\n        ax.set_title(f'True: {label}, Pred: {predicted.item()}', color=color)\\n        ax.axis('off')\\nplt.tight_layout()\\nplt.show()"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Summary\\n\\n✅ Loaded and visualized MNIST\\n✅ Built neural network\\n✅ Trained on 60,000 images\\n✅ Achieved >95% test accuracy\\n✅ Visualized predictions"]}
        ]
    }
}

# Create all notebooks
print("Creating Module 3 notebooks...")
for filepath, content in notebooks.items():
    create_notebook(filepath, content['title'], content)

print("\\n✅ All Module 3 notebooks created!")
