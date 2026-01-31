#!/usr/bin/env python3
"""
Download all datasets required for the course
"""

import os
from pathlib import Path

def download_datasets():
 print("=" * 60)
 print("Downloading Datasets for GenAI Course")
 print("=" * 60)

 # Most datasets are downloaded automatically by libraries
 # This script ensures they're cached locally

 print("\n Downloading scikit-learn datasets...")
 from sklearn.datasets import load_iris, load_diabetes, fetch_california_housing

 # Download and cache
 iris = load_iris()
 print(" Iris dataset")

 diabetes = load_diabetes()
 print(" Diabetes dataset")

 try:
 housing = fetch_california_housing()
 print(" California housing dataset")
 except Exception as e:
 print(f" California housing: {e}")

 print("\n Downloading TensorFlow/Keras datasets...")
 try:
 import tensorflow as tf

 # MNIST
 (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 print(f" MNIST dataset ({len(x_train)} train, {len(x_test)} test)")

 # CIFAR-10
 (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
 print(f" CIFAR-10 dataset ({len(x_train)} train, {len(x_test)} test)")

 # IMDB
 (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
 print(f" IMDB dataset ({len(x_train)} train, {len(x_test)} test)")

 except Exception as e:
 print(f" TensorFlow datasets: {e}")
 print("Install TensorFlow: pip install tensorflow")

 print("\n Downloading PyTorch datasets...")
 try:
 import torchvision
 from torchvision import datasets, transforms

 # Create data directory
 data_dir = Path(__file__).parent

 # MNIST
 datasets.MNIST(root=data_dir, train=True, download=True)
 datasets.MNIST(root=data_dir, train=False, download=True)
 print(" PyTorch MNIST dataset")

 # CIFAR-10
 datasets.CIFAR10(root=data_dir, train=True, download=True)
 datasets.CIFAR10(root=data_dir, train=False, download=True)
 print(" PyTorch CIFAR-10 dataset")

 except Exception as e:
 print(f" PyTorch datasets: {e}")
 print("Install PyTorch: pip install torch torchvision")

 print("\n" + "=" * 60)
 print(" Dataset download complete!")
 print("=" * 60)
 print("\nDatasets are cached and ready to use.")
 print("You can now run the notebooks without internet connection.")

if __name__ == "__main__":
 download_datasets()
